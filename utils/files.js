import fs from 'fs';
import path from 'path';

export const modelsDirectory = './uploads/temp/models/';

export const downloadAFileResponse = (response, fileName, directory) => {
	var fileLocation = path.join(directory, fileName);
	response.download(fileLocation, fileName);
};

export const deleteAllFilesFromDir = (directory) => {
	fs.readdir(directory, (err, files) => {
		if (err) throw err;
		for (const file of files) {
			fs.unlink(path.join(directory, file), (err) => {
				if (err) throw err;
			});
		}
	});
};

export const createJsonFile = (json, directory, fileName) => {
	let data = JSON.stringify(json);
	fs.writeFileSync(`${directory}/${fileName}.json`, data);
};
