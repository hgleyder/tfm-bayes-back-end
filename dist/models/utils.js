'use strict';

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.separateClasses = separateClasses;

var _mlMatrix = require('ml-matrix');

var _mlMatrix2 = _interopRequireDefault(_mlMatrix);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * @private
 * Function that retuns an array of matrices of the cases that belong to each class.
 * @param {Matrix} X - dataset
 * @param {Array} y - predictions
 * @return {Array}
 */
function separateClasses(X, y) {
  var features = X.columns;

  var classes = 0;
  var totalPerClasses = new Array(10000); // max upperbound of classes
  for (var i = 0; i < y.length; i++) {
    if (totalPerClasses[y[i]] === undefined) {
      totalPerClasses[y[i]] = 0;
      classes++;
    }
    totalPerClasses[y[i]]++;
  }
  var separatedClasses = new Array(classes);
  var currentIndex = new Array(classes);
  for (i = 0; i < classes; ++i) {
    separatedClasses[i] = new _mlMatrix2.default(totalPerClasses[i], features);
    currentIndex[i] = 0;
  }
  for (i = 0; i < X.rows; ++i) {
    separatedClasses[y[i]].setRow(currentIndex[y[i]], X.getRow(i));
    currentIndex[y[i]]++;
  }
  return separatedClasses;
}