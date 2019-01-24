var express = require('express');
var router = express.Router();
import fs from 'fs';

function savePersonToPublicFolder(person, callback) {
  fs.writeFile('./public/person.json', JSON.stringify(person), callback);
}

/* GET users listing. */
router.get('/', function(req, res, next) {
  // savePersonToPublicFolder({name: 'Jose'}, () => console.log('done'))
  res.render('files/index.hbs', {title: 'files'});
});

router.get('/api/hello', (req, res) => {
  res.send({ express: 'Hello From Express' });
});
router.post('/api/world', (req, res) => {
  console.log(req.body);
  res.send(
    `I received your POST request. This is what you sent me: ${req.body.post}`,
  );
});

module.exports = router;
