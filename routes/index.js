import express from 'express';
import { MultinomialNB, GaussianNB } from '../models';
import Matrix from 'ml-matrix';
import {
	getDatasetNumericalRepresentation,
	getNumerialMatrixFromRepresentation,
} from '../models/utils/models';
import { crossValidationModel } from '../models/utils/evaluation';

var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
	const matrix = new Matrix([
		[ -1, -1 ],
		[ -2, -1 ],
		[ -3, -2 ],
		[ 1, 1 ],
		[ 2, 1 ],
		[ 3, 2 ],
	]);
	var model = new GaussianNB();
	model.train(matrix, [ 1, 1, 1, 2, 2, 2 ]);

	const matrixTest = [
		[ -1, -1 ],
		[ -2, -1 ],
		[ -3, -2 ],
		[ 1, 1 ],
		[ 2, 1 ],
		[ 3, 2 ],
	];
	var predictions = model.predict(matrixTest);
	console.log(predictions);
	res.json(predictions);
});

/* Create model from text attributes example */
router.get('/text', function(req, res, next) {
	let matrix = [ [ 'calido', 'seco' ], [ 'frio', 'seco' ] ];
	const attributesRepresentations = getDatasetNumericalRepresentation(matrix);
	matrix = getNumerialMatrixFromRepresentation(
		matrix,
		attributesRepresentations,
	);
	var model = new MultinomialNB();
	model.train(matrix, [ 'verano', 'invierno' ]);

	const matrixTest = getNumerialMatrixFromRepresentation(
		[ [ 'frio', 'seco' ] ],
		attributesRepresentations,
	);
	var predictions = model.predict(matrixTest);
	res.json(predictions);
});

/* Cross Validation Metrics */
router.get('/cross', function(req, res, next) {
	const X = [
		[ 'segundo', 1 ],
		[ 'segundo', 2 ],
		[ 'segundo', 3 ],
		[ 'segundo', 4 ],
		[ 'segundo', 5 ],
		[ 'primero', 1 ],
		[ 'primero', 2 ],
		[ 'primero', 3 ],
		[ 'primero', 4 ],
		[ 'primero', 5 ],
		[ 'tercero', 5 ],
		[ 'cuarto', 5 ],
		[ 'quinto', 5 ],
		[ 'primero', 3 ],
		[ 'primero', 4 ],
	];
	const y = [
		'Up',
		'Up',
		'Up',
		'Up',
		'Up',
		'Down',
		'Down',
		'Down',
		'Down',
		'Down',
		'Up',
		'Down',
		'Down',
		'Down',
		'Down',
	];
	var metrics = crossValidationModel(MultinomialNB, X, y);
	res.json(metrics);
});

module.exports = router;
