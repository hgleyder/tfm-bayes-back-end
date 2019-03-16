import express from 'express';
import ErrorMessages from '../utils/errorMessages';
import fs from 'fs';
import { loadModelFromImportedData } from '../models/utils/import';
import {
	removeStopwordsAndApplyStemmer,
	readAttributesFromFile,
	getInstanceFromAttributes,
	createAttributesFile,
	createDatasetFile,
} from '../utils/preprocesing';

var router = express.Router();

/* Load model from json */
router.post('/predict', function(req, res, next) {
	var modelData = req.body.modelData;
	var text = req.body.text;
	const wordList = removeStopwordsAndApplyStemmer(text);
	fs.readFile('./uploads/attributes.txt', 'utf8', (err, data) => {
		if (err) throw err;
		let attributes = data.split('\n');
		attributes.splice(attributes.length - 1, 1);
		const instance = getInstanceFromAttributes(wordList, attributes);
		try {
			const model = loadModelFromImportedData(modelData);
			const predictions = model.predict([ instance ]);
			res.json(predictions.map((p) => model.classes[parseInt(p)]));
		} catch (e) {
			res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED_LOAD });
		}
	});
});

/* Load model from json */
router.get('/predict', function(req, res, next) {
	var modelData = req.body.modelData;
	var text =
		'you just won an increadible prize, please click here to claim your reward!';
	const wordList = removeStopwordsAndApplyStemmer(text);
	fs.readFile('./uploads/attributes2.txt', 'utf8', (err, data) => {
		if (err) throw err;
		let attributes = data.split('\n');
		const instance = getInstanceFromAttributes(wordList, attributes);
		try {
			res.json(instance);
		} catch (e) {
			res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED_LOAD });
		}
	});
});

// create attibutes file from instances data
router.get('/create-attributes', function(req, res, next) {
	createAttributesFile('./uploads/spamData.csv', '/---/');
	res.send('attributes file created');
});

// create Dataset
router.get('/create-dataset', function(req, res, next) {
	createDatasetFile(
		'./uploads/spamData.csv',
		'/---/',
		'./uploads/attributes3.txt',
	);
	res.send('dataset file created');
});

module.exports = router;
