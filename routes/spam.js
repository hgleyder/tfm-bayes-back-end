import express from 'express';
import ErrorMessages from '../utils/errorMessages';
import fs from 'fs';
import {
	createInitialDatasetFile,
	deleteModel,
	createModel,
	preprocessInstances,
	setMessagesClassification,
	createManualModelData,
} from '../utils/modelsCreation';

var router = express.Router();

router.get('/new/manual', function(req, res, next) {
	createManualModelData();
	res.send('dataset file created');
});

// NEEEEEEEEEEEWWWWW FROM HERE

router.get('/new', function(req, res, next) {
	var modelData = req.body.model;
	createInitialDatasetFile();
	res.send('dataset file created');
});

router.post('/remove', function(req, res, next) {
	var modelData = req.body.data.model;
	deleteModel(modelData);
	res.send('model Deteled');
});

/* Load model from json */
router.post('/predict', function(req, res, next) {
	var instances = req.body.data.instances;
	var modelData = req.body.data.modelUid;
	var userUid = req.body.data.userUid;
	try {
		const model = createModel(modelData);
		const instancesContent = preprocessInstances(
			instances.map((inst) => inst.content),
			modelData,
		);
		const values = model.predict(instancesContent);
		var predictionsProbs = model.predict_proba(instancesContent);
		const valsPreds = predictionsProbs.map((pred) =>
			pred.map((v) => Math.pow(10, v)),
		);
		const probs = valsPreds.map((p) => {
			const total = p[0] + p[1];
			return [ p[0] / total, p[1] / total ];
		});

		const predictions = values.map(
			(p, index) =>
				parseInt(p) === 1
					? probs[index][1] >= 0.7
						? model.classes[1]
						: model.classes[0]
					: model.classes[parseInt(p)],
		);

		setMessagesClassification(userUid, instances, predictions);
		res.send('classification of the messages is in process');
	} catch (e) {
		res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED_LOAD });
	}
});
module.exports = router;
