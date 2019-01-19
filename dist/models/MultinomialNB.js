'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.MultinomialNB = undefined;

var _createClass = function () { function defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if ("value" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } } return function (Constructor, protoProps, staticProps) { if (protoProps) defineProperties(Constructor.prototype, protoProps); if (staticProps) defineProperties(Constructor, staticProps); return Constructor; }; }();

var _mlMatrix = require('ml-matrix');

var _mlMatrix2 = _interopRequireDefault(_mlMatrix);

var _utils = require('./utils');

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError("Cannot call a class as a function"); } }

var MultinomialNB = exports.MultinomialNB = function () {
  /**
   * Constructor for Multinomial Naive Bayes, the model parameter is for load purposes.
   * @constructor
   * @param {object} model - for load purposes.
   */
  function MultinomialNB(model) {
    _classCallCheck(this, MultinomialNB);

    if (model) {
      this.conditionalProbability = _mlMatrix2.default.checkMatrix(model.conditionalProbability);
      this.priorProbability = _mlMatrix2.default.checkMatrix(model.priorProbability);
    }
  }

  /**
   * Train the classifier with the current training set and labels, the labels must be numbers between 0 and n.
   * @param {Matrix|Array} trainingSet
   * @param {Array} trainingLabels
   */


  _createClass(MultinomialNB, [{
    key: 'train',
    value: function train(trainingSet, trainingLabels) {
      trainingSet = _mlMatrix2.default.checkMatrix(trainingSet);

      if (trainingSet.rows !== trainingLabels.length) {
        throw new RangeError('the size of the training set and the training labels must be the same.');
      }

      var separateClass = (0, _utils.separateClasses)(trainingSet, trainingLabels);
      this.priorProbability = new _mlMatrix2.default(separateClass.length, 1);

      for (var i = 0; i < separateClass.length; ++i) {
        this.priorProbability[i][0] = Math.log(separateClass[i].length / trainingSet.rows);
      }

      var features = trainingSet.columns;
      this.conditionalProbability = new _mlMatrix2.default(separateClass.length, features);
      for (i = 0; i < separateClass.length; ++i) {
        var classValues = _mlMatrix2.default.checkMatrix(separateClass[i]);
        var total = classValues.sum();
        var divisor = total + features;
        this.conditionalProbability.setRow(i, classValues.sum('column').add(1).div(divisor).apply(matrixLog));
      }
    }

    /**
     * Retrieves the predictions for the dataset with the current model.
     * @param {Matrix|Array} dataset
     * @return {Array} - predictions from the dataset.
     */

  }, {
    key: 'predict',
    value: function predict(dataset) {
      dataset = _mlMatrix2.default.checkMatrix(dataset);
      var predictions = new Array(dataset.rows);
      for (var i = 0; i < dataset.rows; ++i) {
        var currentElement = dataset.getRowVector(i);
        predictions[i] = this.conditionalProbability.clone().mulRowVector(currentElement).sum('row').add(this.priorProbability).maxIndex()[0];
      }

      return predictions;
    }

    /**
     * Function that saves the current model.
     * @return {object} - model in JSON format.
     */

  }, {
    key: 'toJSON',
    value: function toJSON() {
      return {
        name: 'MultinomialNB',
        priorProbability: this.priorProbability,
        conditionalProbability: this.conditionalProbability
      };
    }

    /**
     * Creates a new MultinomialNB from the given model
     * @param {object} model
     * @return {MultinomialNB}
     */

  }], [{
    key: 'load',
    value: function load(model) {
      if (model.name !== 'MultinomialNB') {
        throw new RangeError(model.name + ' is not a Multinomial Naive Bayes');
      }

      return new MultinomialNB(model);
    }
  }]);

  return MultinomialNB;
}();

function matrixLog(i, j) {
  this[i][j] = Math.log(this[i][j]);
}