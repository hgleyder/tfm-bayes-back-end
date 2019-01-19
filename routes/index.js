import express from 'express'
const { MultinomialNB } = require('../models');
var router = express.Router();
import Matrix from 'ml-matrix';

/* GET home page. */
router.get('/', function(req, res, next) {
  const matrix = new Matrix([[1,1], [2,1]]);
  var model = new MultinomialNB();
  model.train(matrix, [1, 2]);

  const matrixTest = new Matrix([[1,1], [2,1]]);
var predictions = model.predict(matrixTest);
  console.log(predictions)
  res.json(predictions);
});

module.exports = router;
