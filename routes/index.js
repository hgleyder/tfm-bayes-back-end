import express from 'express';
import { MultinomialNB, GaussianNB } from '../models';
import Matrix from 'ml-matrix';
import fs from 'fs';
import path from 'path';

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

// define a route to download a file
router.get('/download/:file(*)', (req, res) => {
	var file = req.params.file;
	var fileLocation = path.join('./temp/models', file);
	console.log(fileLocation);
	res.download(fileLocation, file);
});

/* GET home page. */
router.get('/delete', function(req, res, next) {
	const directory = './temp/models';

	fs.readdir(directory, (err, files) => {
		if (err) throw err;
		for (const file of files) {
			fs.unlink(path.join(directory, file), (err) => {
				if (err) throw err;
			});
		}
	});
	res.send('HOLA');
});

module.exports = router;
