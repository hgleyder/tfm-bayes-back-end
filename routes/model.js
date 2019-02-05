import express from 'express';
import { MultinomialNB, GaussianNB, BernoulliNB, NaiveBayes } from '../models';
import Matrix from 'ml-matrix';
import {
	getDatasetNumericalRepresentation,
	getNumerialMatrixFromRepresentation,
} from '../models/utils/models';
import ErrorMessages from '../utils/errorMessages';
import { crossValidationModel } from '../models/utils/evaluation';

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
		res.json(metrics);
	} catch (e) {
		res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED });
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
		res.json(metrics);
	} catch (e) {
		res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED });
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
		res.json(metrics);
	} catch (e) {
		res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED });
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
		res.json(metrics);
	} catch (e) {
		res.status(500).send({ error: ErrorMessages.ERROR_OCCURRED });
	}
});

/* Cross Validation Metrics from data */
router.get('/bernoulli/create/cv', function(req, res, next) {
	var X = [
		[ 2, 1, 0, 0, 0, 0 ],
		[ 2, 0, 1, 0, 0, 0 ],
		[ 1, 0, 0, 1, 0, 0 ],
		[ 1, 0, 0, 0, 1, 1 ],
	];

	var y = [ 0, 0, 0, 1 ];

	var model = new BernoulliNB();
	model.train(X, y);
	// var metrics = crossValidationModel(BernoulliNB, X, y);
	res.json(model);
});

/* Cross Validation Metrics from data */
router.get('/nb', function(req, res, next) {
	var X = [
		[ 'silver', 'silver', 'silver', 'silver', 'silver', 'gray' ],
		[ 'silver', 'gray', 'silver', 'gray', 'silver', 'gray' ],
		[ 'silver', 'silver', 'silver', 'silver', 'silver', 'gray' ],
		[ 'gray', 'silver', 'silver', 'gray', 'silver', 'gray' ],
		[ 'silver', 'gray', 'silver', 'gray', 'silver', 'gray' ],
	];

	var y = [ 'invierno1', 'invierno2', 'invierno1', 'invierno2', 'invierno1' ];

	var model = new NaiveBayes();
	model.train(X, y);
	const pred = model.predict(X);
	res.json(pred);
});

module.exports = router;
