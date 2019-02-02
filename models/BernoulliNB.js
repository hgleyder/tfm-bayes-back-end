import Matrix from 'ml-matrix';

import { separateClasses } from './utils/models';

export class BernoulliNB {
	/**
   * Constructor for Multinomial Naive Bayes, the model parameter is for load purposes.
   * @constructor
   * @param {object} model - for load purposes.
   */
	constructor(model) {
		if (model) {
			this.conditionalProbability = Matrix.checkMatrix(
				model.conditionalProbability,
			);
			this.priorProbability = Matrix.checkMatrix(model.priorProbability);
		}
	}

	/**
   * Train the classifier with the current training set and labels, the labels must be numbers between 0 and n.
   * @param {Matrix|Array} trainingSet
   * @param {Array} trainingLabels
   */
	train(trainingSet, trainingLabels) {
		trainingSet = Matrix.checkMatrix(trainingSet);

		if (trainingSet.rows !== trainingLabels.length) {
			throw new RangeError(
				'the size of the training set and the training labels must be the same.',
			);
		}

		// inspect if there is at least an string attribute
		const stringAttr = trainingSet[0].find(
			(attr) => typeof attr === 'string' || attr instanceof String,
		);

		if (stringAttr) {
			throw new RangeError('the attributes should be numeric');
		}

		let auxTraniningSet = trainingSet;
		auxTraniningSet.map((instance) =>
			instance.map((attr) => (attr > 0 ? 1 : 0)),
		);

		var separateClass = separateClasses(auxTraniningSet, trainingLabels);
		this.priorProbability = new Matrix(separateClass.length, 1);

		for (var i = 0; i < separateClass.length; ++i) {
			this.priorProbability[i][0] = Math.log(
				separateClass[i].length / auxTraniningSet.rows,
			);
		}

		var features = auxTraniningSet.columns;
		this.conditionalProbability = new Matrix(
			separateClass.length,
			features,
		);
		for (i = 0; i < separateClass.length; ++i) {
			var classValues = Matrix.checkMatrix(separateClass[i]);
			var total = classValues.sum();
			var divisor = total + features;
			this.conditionalProbability.setRow(
				i,
				classValues.sum('column').add(1).div(divisor).apply(matrixLog),
			);
		}
	}

	/**
   * Retrieves the predictions for the dataset with the current model.
   * @param {Matrix|Array} dataset
   * @return {Array} - predictions from the dataset.
   */
	predict(dataset) {
		dataset = Matrix.checkMatrix(dataset);
		var predictions = new Array(dataset.rows);
		for (var i = 0; i < dataset.rows; ++i) {
			var currentElement = dataset.getRowVector(i);
			predictions[i] = this.conditionalProbability
				.clone()
				.mulRowVector(currentElement)
				.sum('row')
				.add(this.priorProbability)
				.maxIndex()[0];
		}

		return predictions;
	}

	/**
   * Function that saves the current model.
   * @return {object} - model in JSON format.
   */
	toJSON() {
		return {
			name: 'BernoulliNB',
			priorProbability: this.priorProbability,
			conditionalProbability: this.conditionalProbability,
		};
	}

	/**
   * Creates a new BernoulliNB from the given model
   * @param {object} model
   * @return {BernoulliNB}
   */
	static load(model) {
		if (model.name !== 'BernoulliNB') {
			throw new RangeError(
				`${model.name} is not a Multinomial Naive Bayes`,
			);
		}

		return new BernoulliNB(model);
	}
}

function matrixLog(i, j) {
	this[i][j] = Math.log(this[i][j]);
}
