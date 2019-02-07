import express from 'express';
import { MultinomialNB, GaussianNB, BernoulliNB, NaiveBayes } from '../models';
import Matrix from 'ml-matrix';
import {
	createJsonFile,
	downloadAFileResponse,
	modelsDirectory,
	deleteAllFilesFromDir,
} from '../utils/files';
var router = express.Router();

/* Save Model */
router.post('/saveModel', function(req, res, next) {
	var data = {
		modelName: req.body.data.modelName,
		instances: req.body.data.instances,
		classes: req.body.data.classes,
	};
	const model = new MultinomialNB();
	model.train(data.instances, data.classes);
	const uuid = new Date().getTime();
	const modelName = '';
	const fileName = `${data.modelName}-${uuid}`;
	createJsonFile(model, modelsDirectory, fileName);
	downloadAFileResponse(res, fileName + '.json', modelsDirectory);
});

module.exports = router;
