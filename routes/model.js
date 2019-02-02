import express from 'express';
import { MultinomialNB, GaussianNB } from '../models';
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

module.exports = router;
