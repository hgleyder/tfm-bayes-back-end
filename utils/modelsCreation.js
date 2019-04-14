// testing instance creation
import fs from 'fs';
import { Database } from '../config/firebase';
const sw = require('stopword');
import stemmer from 'stemmer';
import { crossValidationModel } from '../models/utils/evaluation';
import { MultinomialNB } from '../models';
import { createJsonFile } from '../utils/files';

export const createManualModelData = () => {
	let words = [];
	let wordsCounter = {};
	const minCount = 40;
	const modelId = new Date().getTime();
	const instancesPath = './uploads/manual/emails.csv';
	const separator = '/---/';
	fs.mkdirSync(`./uploads/manual/models/${modelId}`);

	// ----------- CREATE ATTRIBUTES FILE -----------------
	fs.readFile(instancesPath, 'utf8', (err, data) => {
		if (err) throw err;
		const instances = data.split('\n');
		instances.map((i) => {
			// Replace email addresses with 'emailaddr'
			// Replace URLs with 'httpaddr'
			// Replace money symbols with 'moneysymb'
			// Replace phone numbers with 'phonenumbr'
			// Replace numbers with 'numbr'
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
						wordsCounter[a] = 1;
					} else {
						wordsCounter[a] = wordsCounter[a] + 1;
					}
				});
			}

			// emails
			attributes = i
				.split(separator)[0]
				.toLowerCase()
				.match(
					/[a-zA-Z0-9.!#$%&’*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*/g,
				);
			if (attributes) {
				if (words.indexOf('emailaddr') === -1) {
					words.push('emailaddr');
					wordsCounter['emailaddr'] = 1;
				} else {
					wordsCounter['emailaddr'] = wordsCounter['emailaddr'] + 1;
				}
			}

			// urls
			attributes = i
				.split(separator)[0]
				.toLowerCase()
				.match(/(((https?:\/\/)|(www\.))[^\s]+)/g);
			if (attributes) {
				if (words.indexOf('urladdrs') === -1) {
					words.push('urladdrs');
					wordsCounter['urladdrs'] = 1;
				} else {
					wordsCounter['urladdrs'] = wordsCounter['urladdrs'] + 1;
				}
			}

			// phone numbers
			attributes = i
				.split(separator)[0]
				.toLowerCase()
				.match(/[+]*[(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]*/g);
			if (attributes) {
				if (words.indexOf('phonenumbr') === -1) {
					words.push('phonenumbr');
					wordsCounter['phonenumbr'] = 1;
				} else {
					wordsCounter['phonenumbr'] = wordsCounter['phonenumbr'] + 1;
				}
			}

			// numbers
			attributes = i
				.split(separator)[0]
				.toLowerCase()
				.match(/-?\d+\.?\d*$/g);
			if (attributes) {
				if (words.indexOf('numbr') === -1) {
					words.push('numbr');
					wordsCounter['numbr'] = 1;
				} else {
					wordsCounter['numbr'] = wordsCounter['numbr'] + 1;
				}
			}

			// currency symbol
			attributes = i
				.split(separator)[0]
				.toLowerCase()
				.match(/(kr|$|£|€)/g);
			if (attributes) {
				if (words.indexOf('currencysymbol') === -1) {
					words.push('currencysymbol');
					wordsCounter['currencysymbol'] = 1;
				} else {
					wordsCounter['currencysymbol'] =
						wordsCounter['currencysymbol'] + 1;
				}
			}
		});

		const importantAttributes = Object.keys(wordsCounter).filter(
			(word) => wordsCounter[word] >= minCount,
		);

		const attrFrequencies = importantAttributes
			.map((attr) => `${attr},${wordsCounter[attr]}`)
			.join('\n');

		fs.writeFileSync(
			`./uploads/manual/models/${modelId}/attributes.txt`,
			attrFrequencies,
		);

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
					// emails
					let aux =
						i
							.split(separator)[0]
							.toLowerCase()
							.match(
								/[a-zA-Z0-9.!#$%&’*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*/g,
							) || [];
					aux.map((word) => {
						attributes.push('emailaddr');
					});

					// urls
					aux =
						i
							.split(separator)[0]
							.toLowerCase()
							.match(/(((https?:\/\/)|(www\.))[^\s]+)/g) || [];
					aux.map((word) => {
						attributes.push('urladdrs');
					});

					// phone numbers
					aux =
						i
							.split(separator)[0]
							.toLowerCase()
							.match(
								/[+]*[(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]*/g,
							) || [];
					aux.map((word) => {
						attributes.push('phonenumbr');
					});

					// numbers
					aux =
						i
							.split(separator)[0]
							.toLowerCase()
							.match(/-?\d+\.?\d*$/g) || [];
					aux.map((word) => {
						attributes.push('numbr');
					});

					// currency symbol
					aux = i
						.split(separator)[0]
						.toLowerCase()
						.match(/(kr|$|£|€)/g);
					aux.map((word) => {
						attributes.push('currencysymbol');
					});
					attributes = sw.removeStopwords(attributes, sw.en);
					attributes = attributes.map((w) => stemmer(w));
					const auxInstance = getInstanceFromAttributes(
						attributes,
						importantAttributes,
						wordsCounter,
						instances.length,
					);
					instancesProcessed.push({
						instance: auxInstance,
						class: clas,
					});
					fs.appendFileSync(
						`./uploads/manual/models/${modelId}/dataset.csv`,
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

		// ---------- SAVE MODEL -------------
		const model = new MultinomialNB();
		model.train(
			instancesProcessed.map((r) => r.instance),
			instancesProcessed.map((r) => r.class),
		);
		createJsonFile(model, `./uploads/manual/models/${modelId}`, 'model');
	});
};

export const createModelData = (modelId, modelNumber) => {
	let words = [];
	let wordsCounter = {};
	const minCount = 40;
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
						wordsCounter[a] = 1;
					} else {
						wordsCounter[a] = wordsCounter[a] + 1;
					}
				});
				// emails
				attributes = i
					.split(separator)[0]
					.toLowerCase()
					.match(
						/[a-zA-Z0-9.!#$%&’*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*/g,
					);
				if (attributes) {
					if (words.indexOf('emailaddr') === -1) {
						words.push('emailaddr');
						counter['emailaddr'] = 1;
					} else {
						counter['emailaddr'] = counter['emailaddr'] + 1;
					}
				}

				// urls
				attributes = i
					.split(separator)[0]
					.toLowerCase()
					.match(/(((https?:\/\/)|(www\.))[^\s]+)/g);
				if (attributes) {
					if (words.indexOf('urladdrs') === -1) {
						words.push('urladdrs');
						counter['urladdrs'] = 1;
					} else {
						counter['urladdrs'] = counter['urladdrs'] + 1;
					}
				}

				// phone numbers
				attributes = i
					.split(separator)[0]
					.toLowerCase()
					.match(/[+]*[(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]*/g);
				if (attributes) {
					if (words.indexOf('phonenumbr') === -1) {
						words.push('phonenumbr');
						counter['phonenumbr'] = 1;
					} else {
						counter['phonenumbr'] = counter['phonenumbr'] + 1;
					}
				}

				// numbers
				attributes = i
					.split(separator)[0]
					.toLowerCase()
					.match(/-?\d+\.?\d*$/g);
				if (attributes) {
					if (words.indexOf('numbr') === -1) {
						words.push('numbr');
						counter['numbr'] = 1;
					} else {
						counter['numbr'] = counter['numbr'] + 1;
					}
				}

				// currency symbol
				attributes = i
					.split(separator)[0]
					.toLowerCase()
					.match(/(kr|$|£|€)/g);
				if (attributes) {
					if (words.indexOf('currencysymbol') === -1) {
						words.push('currencysymbol');
						counter['currencysymbol'] = 1;
					} else {
						counter['currencysymbol'] =
							counter['currencysymbol'] + 1;
					}
				}
			}
		});
		let attrs = Object.keys(wordsCounter)
			.filter((word) => wordsCounter[word] >= minCount)
			.join('\n');
		fs.writeFileSync(
			`./uploads/models/${modelId}/attributes.txt`,
			attrs + '\n',
		);

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
					// emails
					let aux =
						i
							.split(separator)[0]
							.toLowerCase()
							.match(
								/[a-zA-Z0-9.!#$%&’*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*/g,
							) || [];
					aux.map((word) => {
						attributes.push('emailaddr');
					});

					// urls
					aux =
						i
							.split(separator)[0]
							.toLowerCase()
							.match(/(((https?:\/\/)|(www\.))[^\s]+)/g) || [];
					aux.map((word) => {
						attributes.push('urladdrs');
					});

					// phone numbers
					aux =
						i
							.split(separator)[0]
							.toLowerCase()
							.match(
								/[+]*[(]{0,1}[0-9]{1,4}[)]{0,1}[-\s\./0-9]*/g,
							) || [];
					aux.map((word) => {
						attributes.push('phonenumbr');
					});

					// numbers
					aux =
						i
							.split(separator)[0]
							.toLowerCase()
							.match(/-?\d+\.?\d*$/g) || [];
					aux.map((word) => {
						attributes.push('numbr');
					});

					// currency symbol
					aux = i
						.split(separator)[0]
						.toLowerCase()
						.match(/(kr|$|£|€)/g);
					aux.map((word) => {
						attributes.push('currencysymbol');
					});

					const auxInstance = getInstanceFromAttributes(
						attributes,
						importantAttributes,
						wordsCounter,
						instances.length,
					);
					instancesProcessed.push({
						instance: auxInstance,
						class: clas,
					});
					fs.appendFileSync(
						`./uploads/manual/models/${modelId}/dataset.csv`,
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

export const deleteFolderRecursive = function(path) {
	if (fs.existsSync(path)) {
		fs.readdirSync(path).forEach(function(file, index) {
			var curPath = path + '/' + file;
			if (fs.lstatSync(curPath).isDirectory()) {
				// recurse
				deleteFolderRecursive(curPath);
			} else {
				// delete file
				fs.unlinkSync(curPath);
			}
		});
		fs.rmdirSync(path);
	}
};

export const deleteModel = (uid) => {
	Database.child('models/' + uid).remove();
	const path = './uploads/models/' + uid;
	deleteFolderRecursive(path);
};

export const createModel = (uid) => {
	let rawdata = fs.readFileSync(`./uploads/models/${uid}/model.json`);
	let data = JSON.parse(rawdata);
	let model = new MultinomialNB(data);
	return model;
};

export const preprocessInstances = (instances, modelUid) => {
	let auxInstances = instances.map((inst) =>
		removeStopwordsAndApplyStemmer(inst),
	);
	const emailsCount = fs
		.readFileSync(`/uploads/models/${modelUid}/emails.csv`, 'utf8')
		.split('\n').length;

	const attributes = readAttributesFromFile(
		`./uploads/models/${modelUid}/attributes.txt`,
	);

	return auxInstances.map((instanceWordsList) =>
		getInstanceFromAttributes(instanceWordsList, attributes, emailsCount),
	);
};

export const setMessagesClassification = (
	userUid,
	messages,
	classifications,
) => {
	messages.map((message, index) => {
		Database.child(
			`messages/${userUid}/received/${message.uid}/classification`,
		).set(classifications[index]);
		Database.child(
			`messages/${userUid}/received/${message.uid}/originalClassification`,
		).set(classifications[index]);
	});
};

/////////// USEFULL FUNCTIONS ////////////////////////////////////////
export const getInstanceFromAttributes = (
	wordsList,
	attributes,
	wordsCounter,
	emailsCount,
) => {
	const docWordsCount = wordsList.length;
	const instance = attributes.map((a) => {
		const attrDocCount = wordsList.reduce(
			(total, x) => (x == a ? total + 1 : total),
			0,
		);

		return calculateTfIdf(
			docWordsCount,
			attrDocCount,
			wordsCounter[a],
			emailsCount,
		);
	});
	return instance;
};

export const removeStopwordsAndApplyStemmer = (text) => {
	let newInstanceWords = text.toLowerCase().split(' ');
	newInstanceWords = sw.removeStopwords(newInstanceWords, sw.en);
	newInstanceWords = newInstanceWords.map((w) => stemmer(w));
	return newInstanceWords;
};

export const readAttributesFromFile = (attributesPath) => {
	let rawdata = fs.readFileSync(attributesPath, 'utf8');
	const attributes = rawdata.split('\n');
	attributes.splice(attributes.length - 1, 1);
	const aux = {};
	attributes.map((attribute) => {
		const info = attribute.split(',');
		aux[info[0]] = parseInt(info[1]);
	});
	return attributes;
};

export const calculateTfIdf = (
	totalDocWords,
	wordCount,
	docsWithWord,
	docsCount,
) => {
	// WITH TF-DIF
	const TF = Math.log10(wordCount + 1);
	const IDF = Math.log10(docsCount / docsWithWord);
	return TF * IDF;

	// WITHOUT TF-IDF
	// return wordCount;
};
////////////////////////////////////////////////////////////////////////
