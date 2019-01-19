'use strict';

var _express = require('express');

var _express2 = _interopRequireDefault(_express);

var _mlMatrix = require('ml-matrix');

var _mlMatrix2 = _interopRequireDefault(_mlMatrix);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

var _require = require('../models'),
    MultinomialNB = _require.MultinomialNB;

var router = _express2.default.Router();


/* GET home page. */
router.get('/', function (req, res, next) {
  var matrix = _mlMatrix2.default.ones(5, 5);
  var model = new MultinomialNB.MultinomialNB();
  model.train(matrix, [1, 1, 1, 1, 1]);
  console.log(matrix);
  res.json(matrix);
});

module.exports = router;