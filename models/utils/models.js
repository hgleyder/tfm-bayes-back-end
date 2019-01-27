import Matrix from 'ml-matrix';

/**
 * @private
 * Function that retuns an array of matrices of the cases that belong to each class.
 * @param {Matrix} X - dataset
 * @param {Array} y - predictions
 * @return {Array}
 */
export function separateClasses(X, y) {
	var features = X.columns;
	var totalPerClasses = {};
	for (var i = 0; i < y.length; i++) {
		if (totalPerClasses[y[i]] === undefined) {
			totalPerClasses[y[i]] = 0;
		}
		totalPerClasses[y[i]]++;
	}
	const classes = Object.keys(totalPerClasses).length;
	var separatedClasses = new Array(classes);
	var currentIndex = new Array(classes);

	for (i = 0; i < classes; i++) {
		const totalsKeys = Object.keys(totalPerClasses);
		separatedClasses[i] = new Matrix(
			totalPerClasses[totalsKeys[i]],
			features,
		);
		currentIndex[i] = 0;
	}

	const classesList = Object.keys(totalPerClasses);

	for (i = 0; i < X.rows; i++) {
		const auxIndex = classesList.indexOf(y[i].toString());
		// separatedClasses[y[i].toString()][(currentIndex[y[i]]).toString()] = X[i.toString()];
		// separatedClasses[auxIndex].setRow(currentIndex[auxIndex], X[i.toString()]);
		separatedClasses[auxIndex][currentIndex[auxIndex]] = X[i.toString()];
		currentIndex[auxIndex]++;
	}

	return separatedClasses;
}

/**
 * @public
 * Function that returns an object of attributes numerical representation.
  * @param {Matrix} X - dataset
 * @return {Object} - Numerical representation object
 */
export function getDatasetNumericalRepresentation(X) {
	const attributesLength = X[0].length;
	const auxCounter = new Array(attributesLength).fill(0);
	const datasetRepresentation = {};

	for (let index = 0; index < attributesLength; index++) {
		const aux = [];
		datasetRepresentation[index.toString()] = {};
		X.map((instance) => {
			if (!aux.includes(instance[index])) {
				aux.push(instance[index]);
				datasetRepresentation[index.toString()][
					instance[index].toString()
				] =
					auxCounter[index];
				auxCounter[index] = auxCounter[index] + 1;
			}
		});
	}

	return datasetRepresentation;
}

/**
 * @public
 * Function that returns a numerical matrix from string
 * matrix and attributes numerical representation.
  * @param {Matrix} X - dataset
  * @param {Object} numericalAttributes - Numerical Attributes representation
 * @return {Matrix} - Numerical representation matrix
 */
export function getNumerialMatrixFromRepresentation(X, numericalAttributes) {
	const auxMatrix = [];
	X.map((instance) => {
		const aux = [];
		instance.map((attribute, index) => {
			aux.push(numericalAttributes[index.toString()][attribute]);
		});
		auxMatrix.push(aux);
	});

	return auxMatrix;
}
