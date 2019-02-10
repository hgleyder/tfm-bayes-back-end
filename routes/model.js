import express from 'express';
import { MultinomialNB, GaussianNB, BernoulliNB, NaiveBayes } from '../models';
import Matrix from 'ml-matrix';
import {
	getDatasetNumericalRepresentation,
	getNumerialMatrixFromRepresentation,
} from '../models/utils/models';
import ErrorMessages from '../utils/errorMessages';
import { crossValidationModel } from '../models/utils/evaluation';
import { loadModelFromImportedData } from '../models/utils/import';

var router = express.Router();

/* Cross Validation Metrics from data */
router.post('/multinomial/create/cv', function(req, res, next) {
	console.log(req);
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

module.exports = router;
