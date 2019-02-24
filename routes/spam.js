import express from 'express';
import ErrorMessages from '../utils/errorMessages';
import { loadModelFromImportedData } from '../models/utils/import';
import {
	removeStopwordsAndApplyStemmer,
	readAttributesFromFile,
	getInstanceFromAttributes,
} from '../utils/preprocesing';

var router = express.Router();

/* Load model from json */
router.post('/predict', function(req, res, next) {
	var modelData = req.body.modelData;
	var text = req.body.text;
	const wordList = removeStopwordsAndApplyStemmer(text);
	const attributes = readAttributesFromFile('./uploads/attributes.txt');
	const instance = getInstanceFromAttributes(wordList, attributes);

	try {
		const model = loadModelFromImportedData(modelData);
		const predictions = model.predict(instance);
		res.json(predictions.map((p) => model.classes[parseInt(p)]));
	} catch (e) {
		res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED_LOAD });
	}
});

module.exports = router;
