'use strict';

Object.defineProperty(exports, '__esModule', {
	value: true,
});
exports.GaussianNB = undefined;

var _createClass = (function() {
	function defineProperties(target, props) {
		for (var i = 0; i < props.length; i++) {
			var descriptor = props[i];
			descriptor.enumerable = descriptor.enumerable || false;
			descriptor.configurable = true;
			if ('value' in descriptor) descriptor.writable = true;
			Object.defineProperty(target, descriptor.key, descriptor);
		}
	}
	return function(Constructor, protoProps, staticProps) {
		if (protoProps) defineProperties(Constructor.prototype, protoProps);
		if (staticProps) defineProperties(Constructor, staticProps);
		return Constructor;
	};
})();

var _mlMatrix = require('ml-matrix');

var _mlMatrix2 = _interopRequireDefault(_mlMatrix);

var _mlStat = require('ml-stat');

var _mlStat2 = _interopRequireDefault(_mlStat);

var _utils = require('./utils');

function _interopRequireDefault(obj) {
	return obj && obj.__esModule ? obj : { default: obj };
}

function _classCallCheck(instance, Constructor) {
	if (!(instance instanceof Constructor)) {
		throw new TypeError('Cannot call a class as a function');
	}
}

var GaussianNB = (exports.GaussianNB = (function() {
	/**
   * Constructor for the Gaussian Naive Bayes classifier, the parameters here is just for loading purposes.
   * @constructor
   * @param {boolean} reload
   * @param {object} model
   */
	function GaussianNB(reload, model) {
		_classCallCheck(this, GaussianNB);

		if (reload) {
			this.means = model.means;
			this.calculateProbabilities = model.calculateProbabilities;
		}
	}

	/**
   * Function that trains the classifier with a matrix that represents the training set and an array that
   * represents the label of each row in the training set. the labels must be numbers between 0 to n-1 where
   * n represents the number of classes.
   *
   * WARNING: in the case that one class, all the cases in one or more features have the same value, the
   * Naive Bayes classifier will not work well.
   * @param {Matrix|Array} trainingSet
   * @param {Matrix|Array} trainingLabels
   */

	_createClass(
		GaussianNB,
		[
			{
				key: 'train',
				value: function train(trainingSet, trainingLabels) {
					var C1 = Math.sqrt(2 * Math.PI); // constant to precalculate the squared root
					trainingSet = _mlMatrix2.default.checkMatrix(trainingSet);

					if (trainingSet.rows !== trainingLabels.length) {
						throw new RangeError(
							'the size of the training set and the training labels must be the same.',
						);
					}

					var separatedClasses = (0, _utils.separateClasses)(
						trainingSet,
						trainingLabels,
					);
					var calculateProbabilities = new Array(
						separatedClasses.length,
					);
					this.means = new Array(separatedClasses.length);
					for (var i = 0; i < separatedClasses.length; ++i) {
						var means = _mlStat2.default.matrix.mean(
							separatedClasses[i],
						);
						var std = _mlStat2.default.matrix.standardDeviation(
							separatedClasses[i],
							means,
						);

						var logPriorProbability = Math.log10(
							separatedClasses[i].rows / trainingSet.rows,
						);
						calculateProbabilities[i] = new Array(means.length + 1);

						calculateProbabilities[i][0] = logPriorProbability;
						for (var j = 1; j < means.length + 1; ++j) {
							var currentStd = std[j - 1];
							calculateProbabilities[i][j] = [
								1 / (C1 * currentStd),
								-2 * currentStd * currentStd,
							];
						}

						this.means[i] = means;
					}

					this.calculateProbabilities = calculateProbabilities;
				},

				/**
     * function that predicts each row of the dataset (must be a matrix).
     *
     * @param {Matrix|Array} dataset
     * @return {Array}
     */
			},
			{
				key: 'predict',
				value: function predict(dataset) {
					if (
						dataset[0].length ===
						this.calculateProbabilities[0].length
					) {
						throw new RangeError(
							'the dataset must have the same features as the training set',
						);
					}

					var predictions = new Array(dataset.length);

					for (var i = 0; i < predictions.length; ++i) {
						predictions[i] = getCurrentClass(
							dataset[i],
							this.means,
							this.calculateProbabilities,
						);
					}

					return predictions;
				},

				/**
     * Function that export the NaiveBayes model.
     * @return {object}
     */
			},
			{
				key: 'toJSON',
				value: function toJSON() {
					return {
						modelName: 'NaiveBayes',
						means: this.means,
						calculateProbabilities: this.calculateProbabilities,
					};
				},

				/**
     * Function that create a GaussianNB classifier with the given model.
     * @param {object} model
     * @return {GaussianNB}
     */
			},
		],
		[
			{
				key: 'load',
				value: function load(model) {
					if (model.modelName !== 'NaiveBayes') {
						throw new RangeError(
							'The current model is not a Multinomial Naive Bayes, current model:',
							model.name,
						);
					}

					return new GaussianNB(true, model);
				},
			},
		],
	);

	return GaussianNB;
})());

/**
 * @private
 * Function the retrieves a prediction with one case.
 *
 * @param {Array} currentCase
 * @param {Array} mean - Precalculated means of each class trained
 * @param {Array} classes - Precalculated value of each class (Prior probability and probability function of each feature)
 * @return {number}
 */

function getCurrentClass(currentCase, mean, classes) {
	var maxProbability = 0;
	var predictedClass = -1;

	// going through all precalculated values for the classes
	for (var i = 0; i < classes.length; ++i) {
		var currentProbability = classes[i][0]; // initialize with the prior probability
		for (var j = 1; j < classes[0][1].length + 1; ++j) {
			currentProbability += calculateLogProbability(
				currentCase[j - 1],
				mean[i][j - 1],
				classes[i][j][0],
				classes[i][j][1],
			);
		}

		currentProbability = Math.exp(currentProbability);
		if (currentProbability > maxProbability) {
			maxProbability = currentProbability;
			predictedClass = i;
		}
	}

	return predictedClass;
}

/**
 * @private
 * function that retrieves the probability of the feature given the class.
 * @param {number} value - value of the feature.
 * @param {number} mean - mean of the feature for the given class.
 * @param {number} C1 - precalculated value of (1 / (sqrt(2*pi) * std)).
 * @param {number} C2 - precalculated value of (2 * std^2) for the denominator of the exponential.
 * @return {number}
 */
function calculateLogProbability(value, mean, C1, C2) {
	value = value - mean;
	return Math.log10(C1 * Math.exp(value * value / C2));
}
