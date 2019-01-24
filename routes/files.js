var express = require('express');
var router = express.Router();
import fs from 'fs';

function savePersonToPublicFolder(person, callback) {
  fs.writeFile('./public/person.json', JSON.stringify(person), callback);
}

/* GET users listing. */
router.get('/', function(req, res, next) {
  savePersonToPublicFolder({name: 'Jose'}, () => console.log('done'))
  res.render('files/index.hbs', {title: 'files'});
});

module.exports = router;
