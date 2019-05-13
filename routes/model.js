import express from 'express';
import { MultinomialNB, GaussianNB, BernoulliNB, NaiveBayes } from '../models';
import Matrix from 'ml-matrix';
import {
	getDatasetNumericalRepresentation,
	getNumerialMatrixFromRepresentation,
} from '../models/utils/models';
import ErrorMessages from '../utils/errorMessages';
import {
	crossValidationModel,
	crossValidationModelSpam,
} from '../models/utils/evaluation';
import { loadModelFromImportedData } from '../models/utils/import';

var router = express.Router();

/* Cross Validation Metrics from data */
router.post('/multinomial/create/cv', function(req, res, next) {
	var data = {
		instances: req.body.data.instances,
		classes: req.body.data.classes,
	};

	// inspect if there is at least an string attribute
	const stringAttr = data.instances[0].find(
		(attr) => typeof attr === 'string' || attr instanceof String,
	);
	if (stringAttr) {
		res
			.status(500)
			.send({ error: ErrorMessages.NUMERIC_ATTRIBUTES_REQUIRED });
	}

	try {
		var metrics = crossValidationModel(
			MultinomialNB,
			data.instances,
			data.classes,
		);
		const model = new MultinomialNB();
		model.train(data.instances, data.classes);
		res.json({ model, metrics });
	} catch (e) {
		res
			.status(500)
			.send({ error: ErrorMessages.ERROR_OCCURRED_EVALUATIONS });
	}
});

/* Cross Validation Metrics from data */
router.post('/multinomial/create/spam/cv', function(req, res, next) {
	var data = {
		instances: req.body.data.instances,
		classes: req.body.data.classes,
	};

	// inspect if there is at least an string attribute
	const stringAttr = data.instances[0].find(
		(attr) => typeof attr === 'string' || attr instanceof String,
	);
	if (stringAttr) {
		res
			.status(500)
			.send({ error: ErrorMessages.NUMERIC_ATTRIBUTES_REQUIRED });
	}

	try {
		var metrics = crossValidationModelSpam(
			MultinomialNB,
			data.instances,
			data.classes,
		);
		const model = new MultinomialNB();
		model.train(data.instances, data.classes);
		res.json({ model, metrics });
	} catch (e) {
		res
			.status(500)
			.send({ error: ErrorMessages.ERROR_OCCURRED_EVALUATIONS });
	}
});

/* Cross Validation Metrics from data */
router.post('/gaussian/create/cv', function(req, res, next) {
	var data = {
		instances: req.body.data.instances,
		classes: req.body.data.classes,
	};

	// inspect if there is at least an string attribute
	const stringAttr = data.instances[0].find(
		(attr) => typeof attr === 'string' || attr instanceof String,
	);
	if (stringAttr) {
		res
			.status(500)
			.send({ error: ErrorMessages.NUMERIC_ATTRIBUTES_REQUIRED });
	}

	try {
		var metrics = crossValidationModel(
			GaussianNB,
			data.instances,
			data.classes,
		);
		const model = new GaussianNB();
		model.train(data.instances, data.classes);
		res.json({ model, metrics });
	} catch (e) {
		res
			.status(500)
			.send({ error: ErrorMessages.ERROR_OCCURRED_EVALUATIONS });
	}
});

/* Cross Validation Metrics from data */
router.post('/bernoulli/create/cv', function(req, res, next) {
	var data = {
		instances: req.body.data.instances,
		classes: req.body.data.classes,
	};

	// inspect if there is at least an string attribute
	const stringAttr = data.instances[0].find(
		(attr) => typeof attr === 'string' || attr instanceof String,
	);
	if (stringAttr) {
		res
			.status(500)
			.send({ error: ErrorMessages.NUMERIC_ATTRIBUTES_REQUIRED });
	}

	try {
		var metrics = crossValidationModel(
			BernoulliNB,
			data.instances,
			data.classes,
		);
		const model = new BernoulliNB();
		model.train(data.instances, data.classes);
		res.json({ model, metrics });
	} catch (e) {
		res
			.status(500)
			.send({ error: ErrorMessages.ERROR_OCCURRED_EVALUATIONS });
	}
});

/* Cross Validation Metrics from data */
router.post('/naivebayes/create/cv', function(req, res, next) {
	var data = {
		instances: req.body.data.instances,
		classes: req.body.data.classes,
	};
	try {
		var metrics = crossValidationModel(
			NaiveBayes,
			data.instances,
			data.classes,
		);
		const model = new NaiveBayes();
		model.train(data.instances, data.classes);
		res.json({ model, metrics });
	} catch (e) {
		res
			.status(500)
			.send({ error: ErrorMessages.ERROR_OCCURRED_EVALUATIONS });
	}
});

/* Save model to json */
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
	res.json({ url: `${API_URL}/model/saveModel/${fileName}` });
});

/* Download saved model */
router.get('/saveModel/:fileName', function(req, res, next) {
	const fileName = req.params.fileName;
	downloadAFileResponse(res, fileName + '.json', modelsDirectory);
});

/* Load model from json */
router.post('/load', function(req, res, next) {
	var data = req.body.data;
	try {
		const model = loadModelFromImportedData(data);
		res.json(model);
	} catch (e) {
		res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED_LOAD });
	}
});

/* Load model from json */
router.post('/predict', function(req, res, next) {
	var modelData = req.body.modelData;
	var instances = req.body.instances;
	try {
		const model = loadModelFromImportedData(modelData);
		const predictions = model.predict(instances);

		res.json(predictions.map((p, index) => model.classes[parseInt(p)]));
	} catch (e) {
		res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED_LOAD });
	}
});

/* Load model from json */
router.post('/predict-spam', function(req, res, next) {
	var modelData = req.body.modelData;
	var instances = req.body.instances;
	try {
		const model = loadModelFromImportedData(modelData);
		const predictions = model.predict(instances);

		var predictionsProbs = model.predict_proba(instances);
		const valsPreds = predictionsProbs.map((pred) =>
			pred.map((v) => Math.pow(10, v)),
		);
		const probs = valsPreds.map((p) => {
			const total = p[0] + p[1];
			return [ p[0] / total, p[1] / total ];
		});

		res.json(
			predictions.map(
				(p, index) =>
					parseInt(p) === 1
						? probs[index][1] >= 0.7
							? model.classes[1]
							: model.classes[0]
						: model.classes[parseInt(p)],
			),
		);
	} catch (e) {
		res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED_LOAD });
	}
});

module.exports = router;
