import express from 'express'
import { MultinomialNB, GaussianNB } from '../models';
import Matrix from 'ml-matrix';


var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  const matrix = new Matrix([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]);
  var model = new GaussianNB();
  model.train(matrix, [1, 1, 1, 2, 2, 2]);

  const matrixTest = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]];
  var predictions = model.predict(matrixTest);
  console.log(predictions)
  res.json(predictions);
});

module.exports = router;
