// testing instance creation
import fs from 'fs';
import { Database } from '../config/firebase';
const sw = require('stopword');
import stemmer from 'stemmer';
import { crossValidationModel } from '../models/utils/evaluation';
import { MultinomialNB } from '../models';
import { createJsonFile } from '../utils/files';

export const createModelData = (modelId, modelNumber) => {
	let words = [];
	let counter = {};
	const minCount = 20;
	const instancesPath = './uploads/models/' + modelId + '/emails.csv';
	const separator = '/---/';

	// ----------- CREATE ATTRIBUTES FILE -----------------
	fs.readFile(instancesPath, 'utf8', (err, data) => {
		if (err) throw err;
		const instances = data.split('\n');
		instances.map((i) => {
			let attributes = i
				.split(separator)[0]
				.toLowerCase()
				.match(/([a-z]+)/g);
			if (attributes) {
				attributes = sw.removeStopwords(attributes, sw.en);
				attributes = attributes.map((w) => stemmer(w));
				attributes.filter((a) => a.length > 1).map((a) => {
					if (words.indexOf(a) === -1) {
						words.push(a);
						counter[a] = 1;
					} else {
						counter[a] = counter[a] + 1;
					}
				});
			}
		});
		const attrs = Object.keys(counter)
			.filter((word) => counter[word] >= minCount)
			.join('\n');
		fs.writeFileSync(`./uploads/models/${modelId}/attributes.txt`, attrs);

		// ----------- CREATE DATASET FILE -----------------
		const instancesProcessed = [];
		instances.map((i) => {
			if (i.split(separator)[1]) {
				let attributes = i
					.split(separator)[0]
					.toLowerCase()
					.match(/([a-z]+)/g);
				let clas = i
					.split(separator)[1]
					.toLowerCase()
					.match(/([a-z]+)/g)[0];
				if (attributes && i.split(separator).length === 2) {
					attributes = sw.removeStopwords(attributes, sw.en);
					attributes = attributes.map((w) => stemmer(w));
					const auxInstance = getInstanceFromAttributes(
						attributes,
						attrs.split('\n'),
					);
					instancesProcessed.push({
						instance: auxInstance,
						class: clas,
					});
					fs.appendFileSync(
						`./uploads/models/${modelId}/dataset.csv`,
						auxInstance + ',' + clas + '\n',
					);
				}
			}
		});

		// ----------- CREATE MODEL METRICS --------------------
		var metrics = crossValidationModel(
			MultinomialNB,
			instancesProcessed.map((r) => r.instance),
			instancesProcessed.map((r) => r.class),
		);

		Database.child('models/' + modelId).set({
			metrics,
			uid: modelId,
			number: modelNumber,
			instancesCount: instancesProcessed.map((r) => r.instance).length,
			attributesCount: attrs.split('\n').length,
		});

		// ---------- SAVE MODEL -------------
		const model = new MultinomialNB();
		model.train(
			instancesProcessed.map((r) => r.instance),
			instancesProcessed.map((r) => r.class),
		);
		createJsonFile(model, './uploads/models/' + modelId, 'model');
	});
};

export const getInstanceFromAttributes = (wordsList, attributes) => {
	const instance = attributes.map((a) =>
		wordsList.reduce((total, x) => (x == a ? total + 1 : total), 0),
	);
	return instance;
};

export const createInitialDatasetFile = () => {
	const modelId = new Date().getTime();
	// Get current model and current count
	Database.child('modelData').once('value', function(data) {
		if (data.val()) {
			const { modelsCount, currentModel } = data.val();
			if (!fs.existsSync('./uploads/models/' + modelId)) {
				fs.mkdirSync('./uploads/models/' + modelId);
			}
			// destination.txt will be created or overwritten by default.
			fs.copyFile(
				'./uploads/models/' + currentModel + '/emails.csv',
				'./uploads/models/' + modelId + '/emails.csv',
				(err) => {
					if (err) throw err;

					Database.child('verifiedMessages')
						.once('value', (newMessages) => {
							if (newMessages.val()) {
								//Add Verified Messages to previous messages
								Object.keys(newMessages.val()).map((msg) => {
									fs.appendFileSync(
										`./uploads/models/${modelId}/emails.csv`,
										newMessages.val()[msg].content +
											'/---/' +
											newMessages.val()[msg]
												.classification +
											'\n',
									);
								});
							}
						})
						.then(() => {
							createModelData(modelId, modelsCount + 1);

							// Set current model and current count
							Database.child('modelData').set({
								modelsCount: modelsCount + 1,
								currentModel: modelId,
							});

							//Remove verified Messages
							Database.child('verifiedMessages').remove();
						});
				},
			);
		}
	});
};
