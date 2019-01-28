import {
	getDatasetNumericalRepresentation,
	getNumerialMatrixFromRepresentation,
} from './models';

/**
 * @public
 * Evaluate Model with cross validation
  * @param {Object} classificationModel - Classification Model
  * @param {Matrix} X - Instances
  * @param {Array} y - Classes of instances
  * @param {Integer} folds - number of folds for cross validation
 * @return {Matrix} - Numerical representation matrix
 */
export function crossValidationModel(classificationModel, X, y, folds = 10) {
	const instancesAmount = y.length;
	const instancesPerFold = Math.floor(instancesAmount / folds);
	const clasificationResults = [];
	// inspect if there is at least an string attribute
	const stringAttr = X[0].find(
		(attr) => typeof attr === 'string' || attr instanceof String,
	);
	let counter = 0;
	while (counter < folds) {
		let auxXtest = [];
		let auxYtest = [];
		var model = new classificationModel();
		var auxX = X.map((item) => item);
		var auxY = y.map((item) => item);
		if (stringAttr) {
			const attributesRepresentations = getDatasetNumericalRepresentation(
				auxX,
			);
			auxX = getNumerialMatrixFromRepresentation(
				auxX,
				attributesRepresentations,
			);
		}
		auxXtest = auxX.splice(counter * instancesPerFold, instancesPerFold);
		auxYtest = auxY.splice(counter * instancesPerFold, instancesPerFold);
		var auxClassList = getClassesList(auxY);
		model.train(auxX, auxY);
		var predictions = model.predict(auxXtest);
		predictions.map((prediction, i) => {
			clasificationResults.push({
				prediction: auxClassList[prediction].toString(),
				expected: auxYtest[i].toString(),
			});
		});
		counter++;
	}

	const metrics = {};
	const classList = getClassesList(y);
	metrics['generalAccuracy'] = getGeneralAccuracy(clasificationResults);
	metrics['accuracyByClasses'] = {};
	classList.map((c) => {
		metrics['accuracyByClasses'][c.toString()] = getClassAccuracy(
			clasificationResults,
			c,
		);
	});
	return metrics;
}

/**
 * @public
 * Function that returns an array of classes.
 * @param {Array} y - predictions
 * @return {Array} - List of different classes
 */
export function getClassesList(y) {
	var classesList = [];
	y.map((c) => {
		if (!classesList.includes(c)) classesList.push(c);
	});
	return classesList;
}

//  ----------------------- Metrics ----------------------------
/**
 * @public
 * Function that returns model Accuracy.
 * @param {Array} predictions - predictions
 * @return {Double} - Model General Accuracy
 */
export function getGeneralAccuracy(predictions) {
	const predictionsLength = predictions.length;
	let correctlyInstantiatedCounter = 0;
	predictions.map((p) => {
		if (p['prediction'] === p['expected']) correctlyInstantiatedCounter++;
	});
	return `${(parseFloat(correctlyInstantiatedCounter / predictionsLength) *
		100).toFixed(4)}%`;
}

/**
 * @public
 * Function that returns model class Accuracy.
 * @param {Array} predictions - predictions
 * @param {String} currentClass - predictions
 * @return {Double} - Model Class Accuracy
 */
export function getClassAccuracy(predictions, currentClass) {
	const auxPredictions = predictions.filter(
		(p) => p.expected === currentClass.toString(),
	);
	const predictionsLength = auxPredictions.length;
	let correctlyInstantiatedCounter = 0;
	auxPredictions.map((p) => {
		if (p['prediction'] === p['expected']) correctlyInstantiatedCounter++;
	});
	return `${(parseFloat(correctlyInstantiatedCounter / predictionsLength) *
		100).toFixed(4)}%`;
}
