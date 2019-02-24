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
		return attributes;
	});
};

export const getInstanceFromAttributes = (wordsList, attributes) => {
	const instance = attributes.map((a) =>
		wordsList.reduce((total, x) => (x == a ? total + 1 : total), 0),
	);
	return instance;
};
