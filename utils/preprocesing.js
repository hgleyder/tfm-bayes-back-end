// testing instance creation
import fs from 'fs';
const sw = require('stopword');
import stemmer from 'stemmer';

export const removeStopwordsAndApplyStemmer = (text) => {
	let newInstanceWords = text.toLowerCase().split(' ');
	newInstanceWords = sw.removeStopwords(newInstanceWords, sw.en);
	newInstanceWords = newInstanceWords.map((w) => stemmer(w));
	return newInstanceWords;
};

export const readAttributesFromFile = (attributesPath) => {
	fs.readFile(attributesPath, 'utf8', (err, data) => {
		if (err) throw err;
		const attributes = data.split('\n');
		attributes.splice(attributes.length - 1, 1);
		return attributes;
	});
};

export const createAttributesFile = (instancesPath, separator) => {
	let words = [];
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
					}
				});
			}
		});
		const data2 = words.join('\n');
		fs.writeFileSync(`uploads/attributes2.txt`, data2);
	});
};

export const createDatasetFile = (instancesPath, separator, attributesPath) => {
	fs.readFile(attributesPath, 'utf8', (e, attributesData) => {
		fs.readFile(instancesPath, 'utf8', (err, data) => {
			if (err) throw err;
			const instances = data.split('\n');
			instances.map((i) => {
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
						attributesData.split('\n'),
					);
					fs.appendFileSync(
						`uploads/dataset.csv`,
						auxInstance + ',' + clas + '\n',
					);
				}
			});
		});
	});
};

export const getInstanceFromAttributes = (wordsList, attributes) => {
	const instance = attributes.map((a) =>
		wordsList.reduce((total, x) => (x == a ? total + 1 : total), 0),
	);
	return instance;
};
