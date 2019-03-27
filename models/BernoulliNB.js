import Matrix from 'ml-matrix';

import { separateClasses } from './utils/models';
import { getClassesList } from './utils/evaluation';

export class BernoulliNB {
	/**
   * Constructor for Bernoulli Naive Bayes.
   * @constructor
   * @param {object} model
   */
	constructor(model) {
		if (model) {
			this.conditionalProbability = Matrix.checkMatrix(
				model.conditionalProbability,
			);
			this.priorProbability = Matrix.checkMatrix(model.priorProbability);
			this.classes = model.classes;
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
		auxTraniningSet.map((instance, ind) => {
			auxTraniningSet[ind].map((attr, ind2) => {
				auxTraniningSet[ind][ind2] = attr > 0 ? 1 : 0;
			});
		});

		var separateClass = separateClasses(trainingSet, trainingLabels);
		this.priorProbability = new Matrix(separateClass.length, 1);

		for (var i = 0; i < separateClass.length; ++i) {
			this.priorProbability[i][0] = Math.log10(
				separateClass[i].length / trainingSet.rows,
			);
		}

		var features = trainingSet.columns;
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
				classValues.sum('column').add(2).div(divisor),
			);
		}

		this.classes = getClassesList(trainingLabels);
	}

	/**
   * Retrieves the predictions for the dataset with the current model.
   * @param {Matrix|Array} dataset
   * @return {Array} - predictions from the dataset.
   */
	predict(dataset) {
		dataset = Matrix.checkMatrix(dataset);
		let auxDataset = dataset;
		auxDataset.map((instance, ind) => {
			auxDataset[ind].map((attr, ind2) => {
				auxDataset[ind][ind2] = attr > 0 ? 1 : 0;
			});
		});
		var predictions = new Array(auxDataset.rows);
		for (var i = 0; i < auxDataset.rows; ++i) {
			var currentElement = auxDataset.getRowVector(i);

			console.log(this.conditionalProbability);
			let auxProb = [];
			for (let i = 0; i < this.conditionalProbability.length; i++) {
				auxProb.push([]);
				for (
					let j = 0;
					j < this.conditionalProbability[i].length;
					j++
				) {
					auxProb[i].push(1 - this.conditionalProbability[i][j]);
				}
			}
			auxProb = new Matrix(auxProb);

			var inverseCurrent = [];
			for (let index = 0; index < currentElement[0].length; index++) {
				let val = currentElement[0][index];
				inverseCurrent.push(
					Math.abs(parseInt(currentElement[0][index]) - 1),
				);
			}
			inverseCurrent = new Matrix([ inverseCurrent ]);

			predictions[i] = this.conditionalProbability
				.apply(matrixLog)
				.mulRowVector(currentElement)
				.add(auxProb.apply(matrixLog).mulRowVector(inverseCurrent))
				.sum('row')
				.add(this.priorProbability)
				.maxIndex()[0];
		}

		console.log(predictions);
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
			classes: this.classes,
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
				`${model.name} is not a Bernoulli Naive Bayes`,
			);
		}

		return new BernoulliNB(model);
	}
}

function matrixLog(i, j) {
	this[i][j] = Math.log10(this[i][j]);
}
