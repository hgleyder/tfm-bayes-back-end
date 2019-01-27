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
	const classList = getClassesList(y);
	const clasificationResults = [];
	let counter = 0;
	while (counter < folds) {
		let auxXtest = [];
		let auxYtest = [];
		var auxX = X.map((item) => item);
		var auxY = y.map((item) => item);
		auxXtest = auxX.splice(counter * instancesPerFold, instancesPerFold);
		auxYtest = auxY.splice(counter * instancesPerFold, instancesPerFold);
		classificationModel.train(auxX, auxY);
		var predictions = classificationModel.predict(auxXtest);
		predictions.map((prediction, i) => {
			clasificationResults.push({
				prediction: classList[prediction].toString(),
				expected: auxYtest[i].toString(),
			});
		});
		counter++;
	}

	const metrics = {};
	metrics['generalAccuracy'] = getGeneralAccuracy(clasificationResults);
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
