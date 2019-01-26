import express from 'express';
import { MultinomialNB, GaussianNB } from '../models';
import Matrix from 'ml-matrix';
import {
	getDatasetNumericalRepresentation,
	getNumerialMatrixFromRepresentation,
} from '../models/utils';

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

/* GET home page. */
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

module.exports = router;
