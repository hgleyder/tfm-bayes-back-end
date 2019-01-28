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
	metrics['precisionByClasses'] = {};
	metrics['recallByClasses'] = {};
	metrics['fMeasureByClasses'] = {};
	classList.map((c) => {
		metrics['precisionByClasses'][c.toString()] = getClassPrecision(
			clasificationResults,
			c,
		);
		metrics['recallByClasses'][c.toString()] = getClassRecall(
			clasificationResults,
			c,
		);
		metrics['fMeasureByClasses'][c.toString()] = calculateFMeasure(
			getClassPrecision(clasificationResults, c),
			getClassRecall(clasificationResults, c),
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
 * Function that returns model class Precision.
 * @param {Array} predictions - predictions
 * @param {String} currentClass - predictions
 * @return {Double} - Model Class Precision
 */
export function getClassPrecision(predictions, currentClass) {
	const auxPredictions = predictions.filter(
		(p) => p.expected === currentClass.toString(),
	);
	let tp = 0;
	let fp = 0;
	auxPredictions.map((p) => {
		if (p['prediction'] === p['expected']) tp++;
	});
	predictions.map((p) => {
		if (
			p['prediction'] === currentClass.toString() &&
			p['prediction'] !== p['expected']
		)
			fp++;
	});

	return parseFloat(tp / (tp + fp)).toFixed(4);
}

/**
 * @public
 * Function that returns model class Recall.
 * @param {Array} predictions - predictions
 * @param {String} currentClass - predictions
 * @return {Double} - Model Class Recall
 */
export function getClassRecall(predictions, currentClass) {
	const auxPredictions = predictions.filter(
		(p) => p.expected === currentClass.toString(),
	);
	let tp = 0;
	let fn = 0;
	auxPredictions.map((p) => {
		if (p['prediction'] === p['expected']) tp++;
	});
	predictions.map((p) => {
		if (
			p['prediction'] !== currentClass.toString() &&
			p['expected'] === currentClass.toString()
		)
			fn++;
	});
	return parseFloat(tp / (tp + fn)).toFixed(4);
}

/**
 * @public
 * Function that calculate f-Measure.
 * @param {Double} precision - precision
 * @param {Double} recall - recall
 * @return {Double} - F-Measure
 */
export function calculateFMeasure(precision, recall) {
	const p = parseFloat(precision);
	const r = parseFloat(recall);
	return (2 * (p * r / (p + r))).toFixed(4);
}
