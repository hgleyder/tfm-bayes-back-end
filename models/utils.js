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
      separatedClasses[i] = new Matrix(totalPerClasses[totalsKeys[i]], features);
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
 * Function that retuns an array of classes.
 * @param {Array} y - predictions
 * @return {Array} - List of different classes
 */
export function getClassesList(y) {
  var totalPerClasses = {};
  for (var i = 0; i < y.length; i++) {
    if (totalPerClasses[y[i]] === undefined) {
      totalPerClasses[y[i]] = 0;
    }
    totalPerClasses[y[i]]++;
  }
  return Object.keys(totalPerClasses)
}