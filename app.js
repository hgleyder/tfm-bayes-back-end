var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
const bodyParser = require('body-parser');

var modelRouter = require('./routes/model');
var spamRouter = require('./routes/spam');

import frontEndUrl from './config/frontEndUrl';

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'hbs');

app.use(bodyParser.json({ limit: '100mb' }));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use(function(req, res, next) {
	// // Website you wish to allow to connect
	// res.setHeader('Access-Control-Allow-Origin', frontEndUrl);
	var allowedOrigins = [
		'https://nbmail.me',
		'http://http://68.183.165.156:3000',
		'http://127.0.0.1:3000',
		'http://127.0.0.1:3002',
		'https://www.nbmail.me',

		'https://bayesian.nbmail.me',
		'http://http://68.183.165.156:4000',
		'http://127.0.0.1:4000',
		'http://127.0.0.1:3002',
		'https://www.bayesian.nbmail.me',
	];
	var origin = req.headers.origin;
	if (allowedOrigins.indexOf(origin) > -1) {
	}

	res.setHeader('Access-Control-Allow-Origin', '*');

	res.setHeader('Access-Control-Allow-Credentials', 'true');
	res.setHeader(
		'Access-Control-Allow-Methods',
		'GET,HEAD,OPTIONS,POST,PUT,DELETE',
	);
	res.setHeader(
		'Access-Control-Allow-Headers',
		'Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers, Authorization, access-control-allow-origin',
	);

	// Pass to next layer of middleware
	next();
});

app.use('/model', modelRouter);
app.use('/spam', spamRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
	next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
	// set locals, only providing error in development
	res.locals.message = err.message;
	res.locals.error = req.app.get('env') === 'development' ? err : {};

	// render the error page
	res.status(err.status || 500);
	res.render('error');
});

app.listen(3001, () => console.log('app listening on port 3001!'));

module.exports = app;
