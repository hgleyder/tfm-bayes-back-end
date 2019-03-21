import * as firebase from 'firebase';
var config = {
	apiKey: 'AIzaSyBHZyYmGT8kVN2XnQyHsXI647NGml2Ut1Q',
	authDomain: 'tfm-spam-messenger.firebaseapp.com',
	databaseURL: 'https://tfm-spam-messenger.firebaseio.com',
	projectId: 'tfm-spam-messenger',
	storageBucket: '',
	messagingSenderId: '3442881501',
};

firebase.initializeApp(config);

export const Database = firebase.database().ref();

export const Auth = firebase.auth();
