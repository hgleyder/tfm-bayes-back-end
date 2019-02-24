import express from 'express';
import { MultinomialNB, GaussianNB, BernoulliNB, NaiveBayes } from '../models';
import Matrix from 'ml-matrix';

import {
	createJsonFile,
	downloadAFileResponse,
	modelsDirectory,
	deleteAllFilesFromDir,
} from '../utils/files';
import API_URL from '../config/apiUrl';

var router = express.Router();

/* Save Model */
router.post('/saveModel', function(req, res, next) {
	var data = {
		modelName: req.body.data.modelName,
		instances: req.body.data.instances,
		classes: req.body.data.classes,
	};
	let model;
	if (data.modelName === 'NaiveBayes') model = new NaiveBayes();
	if (data.modelName === 'MultinomialNB') model = new MultinomialNB();
	if (data.modelName === 'GaussianNB') model = new GaussianNB();
	if (data.modelName === 'BernoulliNB') model = new BernoulliNB();
	deleteAllFilesFromDir(modelsDirectory);
	model.train(data.instances, data.classes);
	const uuid = new Date().getTime();
	const fileName = `${data.modelName}-${uuid}`;
	createJsonFile(model, modelsDirectory, fileName);
	res.json({ url: `${API_URL}/files/saveModel/${fileName}` });
});

/* Save Model */
router.get('/saveModel/:fileName', function(req, res, next) {
	const fileName = req.params.fileName;
	downloadAFileResponse(res, fileName + '.json', modelsDirectory);
});

module.exports = router;
