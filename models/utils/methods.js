import Matrix from 'ml-matrix';

export function getValuesByClasses(instances, labels) {
	var features = instances.columns;
	var totalPerClasses = {};
	for (var i = 0; i < labels.length; i++) {
		if (totalPerClasses[labels[i]] === undefined) {
			totalPerClasses[labels[i]] = 0;
		}
		totalPerClasses[labels[i]]++;
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

	for (i = 0; i < instances.rows; i++) {
		const auxIndex = classesList.indexOf(labels[i].toString());
		separatedClasses[auxIndex][currentIndex[auxIndex]] =
			instances[i.toString()];
		currentIndex[auxIndex]++;
	}

	return separatedClasses;
}
