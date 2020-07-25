import sklearn
from sklearn.ensemble import RandomForestRegressor
from connect import *
import numpy as np
import pandas as pd
import threading
import joblib
import os
import sys
import re
from datetime import date
from datetime import datetime

def split_full_path(full_path):
	m = re.search('^(.+[\/]+)(.+)', full_path)
	path, filename = None, None
	if (m):
		path = m.group(1)
		filename = m.group(2)

	return path, filename


def find_model(path, filename):
	print("Searching for pretrained model...")
	if (path is not None) and (filename is not None):
		files = []
		for root, dirs, files_ in os.walk(path):
			if filename in files_:
				files.append(os.path.join(root, filename))

		query = "(.*)([0-9]{2}-[0-9]{2}-[0-9]{4})"
		dates = {}
		for f in files:
			match = re.search(query, f)
			if match:
				name = match.group(1)
				date = match.group(2)
				dates[datetime.strptime(date, "%d-%m-%Y")] = name+date
		model = None

		if (len(dates) > 0):
			latest_date = max(list(dates.keys()))
			model_filename = dates[latest_date]
			model = joblib.load(model_filename)
			print("Model found: " + model_filename)

		else:
			print("Model not found!")
		return model

	print("Model not found!")
	return None


def save_model(model):
	postfix = date.today().strftime("%d-%m-%Y")
	filename = 'model_' + postfix
	joblib.dump(model, filename)


def fit_model(model, time=None):
	data = None

	if (time == None):
		data = load_data('btc_value')
	else:
		data = load_data_from_time(time)

	trainX, trainY = process_btc_values(data)

	model.fit(trainX, trainY)


def predict(model):
	data = load_data("btc_value_pred")
	if (data is not None):
		ids, X = process_btc_values_pred(data)
		if (X.shape[0] > 1):
			try:
				predictions = model.predict(X)
				insert_predictions(ids, predictions)
			except IndexError:
				pass


def process_btc_values(data):
	times = []
	values = []

	for row in data:
		time = row[1]
		value = row[2]

		times.append(time)
		values.append(value)

	last_fitted = times[len(times)-1]

	df = pd.DataFrame({'time': times, 'target': values})

	return df.loc[:, ['time']].values, df.target.values


def process_btc_values_pred(data):
	ids = []
	times = []

	for row in data:
		value = row[2]
		if (value is None):
			id_ = row[0]
			time = row[1]
			times.append(time)
			ids.append(id_)

	df = pd.DataFrame({'time': times})

	return ids, df.loc[:, ['time']].values

def scheduled_save_model():
	print("Saving model...")
	save_model(model)
	print("Saving model: Done!")


def scheduled_fit_model():
	print("Fitting...")
	fit_model(model, last_fitted)
	print("Fitting: Done!")


def scheduled_predict():
	print("Predicting...")
	try:
		predict(model)
	except sklearn.exceptions.NotFittedError:
		print("You need to fit the model first!")
	print("Predicting: Done!")


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t


model_path, model_filename = None, None
if (len(sys.argv) > 1):
	full_path = sys.argv[1]
	if ('/' not in full_path):
		full_path = './' + full_path
	model_path, model_filename = split_full_path(sys.argv[1])
else:
	model_path, model_filename = '.', 'model_*'

model = find_model(model_path, model_filename)
if model is None:
	model = RandomForestRegressor()


last_fitted = None
fit_interval = 30
predict_interval = 30


set_interval(scheduled_fit_model, fit_interval)
set_interval(scheduled_predict, predict_interval)
set_interval(scheduled_save_model, 60)