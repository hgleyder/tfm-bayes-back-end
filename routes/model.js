import express from 'express';
import { MultinomialNB, GaussianNB, BernoulliNB } from '../models';
import Matrix from 'ml-matrix';
import {
	getDatasetNumericalRepresentation,
	getNumerialMatrixFromRepresentation,
} from '../models/utils/models';
import { crossValidationModel } from '../models/utils/evaluation';

var router = express.Router();

/* Cross Validation Metrics from data */
router.post('/multinomial/create/cv', function(req, res, next) {
	console.log(req);
	var data = {
		instances: req.body.data.instances,
		classes: req.body.data.classes,
	};

	var metrics = crossValidationModel(
		MultinomialNB,
		data.instances,
		data.classes,
	);
	res.json(metrics);
});

/* Cross Validation Metrics from data */
router.post('/gaussian/create/cv', function(req, res, next) {
	console.log(req);
	var data = {
		instances: req.body.data.instances,
		classes: req.body.data.classes,
	};

	var metrics = crossValidationModel(
		GaussianNB,
		data.instances,
		data.classes,
	);
	res.json(metrics);
});

/* Cross Validation Metrics from data */
router.get('/bernoulli/create/cv', function(req, res, next) {
	var X = [
		[ 1, 1 ],
		[ 1, 2 ],
		[ 1, 3 ],
		[ 1, 4 ],
		[ 1, 5 ],
		[ 2, 1 ],
		[ 2, 2 ],
		[ 2, 3 ],
		[ 2, 4 ],
		[ 2, 5 ],
	];

	var y = [ 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 ];

	var model = new BernoulliNB();
	model.train(X, y);
	// var metrics = crossValidationModel(BernoulliNB, X, y);
	res.json(model);
});

module.exports = router;
