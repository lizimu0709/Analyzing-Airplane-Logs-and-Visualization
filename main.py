import pandas as pd
import os
import json
import plotly.io as pio
from flask import Flask, render_template, request, redirect, session, g
from flask_caching import Cache
import plotly.graph_objects as go
# from flask_caching import Cache
from collections import Counter
from sklearn import metrics
from sklearn.mixture import GaussianMixture
import numpy as np
from flask_restful import Api, Resource
import pyrebase

app = Flask(__name__, template_folder='src/template', static_folder='src/static')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
app.secret_key = 'capstone_project'

uploaded_file = None
dataload_file = None
firewall_file = None
staging_file = None


@app.route('/')
def index():
	return render_template('index.html')


def classify_features(features):
	if 'E1' in features and 'E2' not in features:
		return 'Success'
	elif 'E2' in features and 'E1' not in features:
		return 'Fail'
	else:
		return 'Other'


def combine_by_cluster(df):
	df.fillna('None', inplace=True)
	merged_df = df.groupby('Y_Gaussian').agg({
		'Airplane Tail': lambda x: '+'.join(x),
		'Features': lambda x: '+'.join(x),
	})
	merged_df = merged_df.assign(Count=df.groupby('Y_Gaussian')['Airplane Tail'].count())
	merged_df = merged_df.reset_index().rename(columns={
		'Airplane Tail': 'Airplane Tails',
	})

	columns = ["Airplane Tails", "Features"]
	for column in columns:
		merged_df[column] = merged_df[column].apply(lambda x: x.split('+')).apply(lambda x: list(set(x)))

	df_linebreaks = merged_df.copy()
	df_linebreaks['Airplane Tails'] = df_linebreaks['Airplane Tails'].apply(lambda x: '<br>'.join(x))
	df_linebreaks['Features'] = df_linebreaks['Features'].apply(
		lambda x: '<br>'.join([str(lst) for lst in x]))

	return df_linebreaks


def results_by_Category(df_log):
	Category_count = {}
	feature_count = {}
	detailed_information = []
	df_log['Category'] = df_log['Features'].apply(classify_features)
	for Category in ['Success', 'Fail', 'Other']:
		df_temp = df_log[df_log['Category'] == Category]
		count = df_temp['Features'].value_counts()
		feature_count[Category] = {'high': [], 'low': []}
		large = count.nlargest(2).to_dict()
		small = count.nsmallest(2).to_dict()
		for k, v in large.items():
			feature_count[Category]['high'].append(str({k: v})[1:-1])
		for k, v in small.items():
			feature_count[Category]['low'].append(str({k: v})[1:-1])
		if len(df_temp) != 0:
			cluster(getFeature(df_temp), df_temp)
			detailed_information.append(combine_by_cluster(df_temp))
			Category_count[Category] = (df_temp["Y_Gaussian"].value_counts().to_dict())
	return Category_count, feature_count, detailed_information


@app.route('/dataload')
def dataload():
	global dataload_file
	user = session.get('user')
	if dataload_file and user:
		calculated = cache.get('dataload_flag_dataload')
		if calculated is None:
			try:
				absolute_path = os.path.dirname(__file__)
				# need fix (save the file to specific route)
				full_path = os.path.join(absolute_path, dataload_file.filename)
				# Format data to a dataframe
				df = log2df(full_path)
				df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
				df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
				logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')

				fig_bar = go.Bar(x=logs_per_day["Episode Start Date"], y=logs_per_day["Count"], xaxis="x1", yaxis="y1")

				Category_count, feature_count, detailed_information = results_by_Category(df)
				df_e1, df_e2, df_other = detailed_information[0], detailed_information[1], detailed_information[2]
				relative_path = os.path.join("src", "static", "dataload.json")

				file_path = os.path.join(absolute_path, relative_path)
				json_data = json.dumps(Category_count)
				with open(file_path, 'w') as f:
					f.write(json_data)

				# Table
				trace_table = go.Table(
					header=dict(values=list(df.columns),
					            align='left'),
					cells=dict(values=[df[k].tolist() for k in df.columns[0:]],
					           align='left'),
					domain=dict(x=[0, 0.45],
					            y=[0, 1])
				)

				layout = dict(xaxis1=dict(dict(domain=[0.5, 1], anchor='y1')),
				              yaxis1=dict(dict(domain=[0, 1], anchor='x1')),
				              title='Dataload Analysis')
				fig = go.Figure(data=[trace_table, fig_bar], layout=layout)

				trace_table2 = go.Table(
					header=dict(values=list(df_e1.columns),
					            align='left'),
					cells=dict(values=[df_e1[k].tolist() for k in df_e1.columns[0:]],
					           align='left'),
					columnwidth=[80, 400, 600, 80])
				layout2 = dict(title='General information of success clusters')
				fig2 = go.Figure(data=[trace_table2], layout=layout2)

				trace_table3 = go.Table(
					header=dict(values=list(df_e2.columns),
					            align='left'),
					cells=dict(values=[df_e2[k].tolist() for k in df_e2.columns[0:]],
					           align='left'),
					columnwidth=[80, 400, 600, 80])
				layout3 = dict(title='General information of fail clusters')
				fig3 = go.Figure(data=[trace_table3], layout=layout3)

				trace_table4 = go.Table(
					header=dict(values=list(df_other.columns),
					            align='left'),
					cells=dict(values=[df_other[k].tolist() for k in df_other.columns[0:]],
					           align='left'),
					columnwidth=[80, 400, 600, 80])
				layout4 = dict(title='General information of other clusters')
				fig4 = go.Figure(data=[trace_table4], layout=layout4)

				fig_html = pio.to_html(fig, full_html=False)
				success_analysis = pio.to_html(fig2, full_html=False)
				fail_analysis = pio.to_html(fig3, full_html=False)
				other_analysis = pio.to_html(fig4, full_html=False)

				cache.set('dataload_flag_dataload', True, timeout=86400)
				cache.set('dataload_plot_dataload', fig_html, timeout=86400)
				cache.set('dataload_log_count', df.shape[0], timeout=86400)
				cache.set('dataload_feature_count', feature_count, timeout=86400)
				cache.set('dataload_success_analysis', success_analysis, timeout=86400)
				cache.set('dataload_fail_analysis', fail_analysis, timeout=86400)
				cache.set('dataload_other_analysis', other_analysis, timeout=86400)

				return render_template(
					"dataload.html",
					name_dataload=dataload_file.filename,
					plot_dataload=fig_html,
					log_count=df.shape[0],
					feature_count=feature_count,
					success_analysis=success_analysis,
					fail_analysis=fail_analysis,
					other_analysis=other_analysis,
				)
			except pd.errors.EmptyDataError:
				return render_template("error.html", message="File is empty")
		else:
			fig_html = cache.get('dataload_plot_dataload')
			log_count = cache.get('dataload_log_count')
			feature_count = cache.get('dataload_feature_count')
			success_analysis = cache.get('dataload_success_analysis')
			fail_analysis = cache.get('dataload_fail_analysis')
			other_analysis = cache.get('dataload_other_analysis')
			return render_template(
				"dataload.html",
				name_dataload=dataload_file.filename,
				plot_dataload=fig_html,
				log_count=log_count,
				feature_count=feature_count,
				success_analysis=success_analysis,
				fail_analysis=fail_analysis,
				other_analysis=other_analysis,
			)
	else:
		return render_template("error.html", message="No dataload file found")


@app.route('/firewall')
def firewall():
	global firewall_file
	user = session.get('user')
	if firewall_file and user:
		calculated = cache.get('firewall_flag_firewall')
		if calculated is None:
			try:
				absolute_path = os.path.dirname(__file__)
				full_path = os.path.join(absolute_path, firewall_file.filename)

				# Format data to a dataframe
				df = log2df(full_path)
				df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
				df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
				logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')

				fig_bar = go.Bar(x=logs_per_day["Episode Start Date"], y=logs_per_day["Count"], xaxis="x1", yaxis="y1")

				Category_count, feature_count, detailed_information = results_by_Category(df)
				df_other = detailed_information[0]
				relative_path = os.path.join("src", "static", "firewall.json")
				file_path = os.path.join(absolute_path, relative_path)
				json_data = json.dumps(Category_count)
				with open(file_path, 'w') as f:
					f.write(json_data)

				# Table
				trace_table = go.Table(
					header=dict(values=list(df.columns),
					            align='left'),
					cells=dict(values=[df[k].tolist() for k in df.columns[0:]],
					           align='left'),
					domain=dict(x=[0, 0.45],
					            y=[0, 1])
				)

				layout = dict(xaxis1=dict(dict(domain=[0.5, 1], anchor='y1')),
				              yaxis1=dict(dict(domain=[0, 1], anchor='x1')),
				              title='firewall Analysis')

				fig = go.Figure(data=[trace_table, fig_bar], layout=layout)

				trace_table4 = go.Table(
					header=dict(values=list(df_other.columns),
					            align='left'),
					cells=dict(values=[df_other[k].tolist() for k in df_other.columns[0:]],
					           align='left'),
					columnwidth=[80, 400, 600, 80])
				layout4 = dict(title='General information of other clusters')
				fig4 = go.Figure(data=[trace_table4], layout=layout4)

				fig_html = pio.to_html(fig, full_html=False)
				other_analysis = pio.to_html(fig4, full_html=False)

				cache.set('firewall_flag_firewall', True, timeout=86400)
				cache.set('firewall_plot_firewall', fig_html, timeout=86400)
				cache.set('firewall_log_count', df.shape[0], timeout=86400)
				cache.set('firewall_feature_count', feature_count, timeout=86400)
				cache.set('firewall_other_analysis', other_analysis, timeout=86400)

				return render_template(
					"firewall.html",
					name_firewall=firewall_file.filename,
					plot_firewall=fig_html,
					log_count=df.shape[0],
					feature_count=feature_count,
					other_analysis=other_analysis,
				)
			except pd.errors.EmptyDataError:
				return render_template("error.html", message="File is empty")
		else:
			fig_html = cache.get('firewall_flag_dataload')
			log_count = cache.get('firewall_log_count')
			feature_count = cache.get('firewall_feature_count')
			other_analysis = cache.get('firewall_other_analysis')
			return render_template(
				"firewall.html",
				name_firewall=firewall_file.filename,
				plot_firewall=fig_html,
				log_count=log_count,
				feature_count=feature_count,
				other_analysis=other_analysis,
			)
	else:
		return render_template("error.html", message="No firewall file found")


@app.route('/staging')
def staging():
	# if not session['user']:
	# 	return
	global staging_file
	user = session.get('user')
	if staging_file and user:
		calculated = cache.get('staging_flag_staging')
		if calculated is None:
			try:
				absolute_path = os.path.dirname(__file__)
				full_path = os.path.join(absolute_path, staging_file.filename)

				# Format data to a dataframe
				df = log2df(full_path)
				df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
				df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
				logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')

				fig_bar = go.Bar(x=logs_per_day["Episode Start Date"], y=logs_per_day["Count"], xaxis="x1", yaxis="y1")

				Category_count, feature_count, detailed_information = results_by_Category(df)
				df_e1, df_e2, df_other = detailed_information[0], detailed_information[1], detailed_information[2]
				relative_path = os.path.join("src", "static", "staging.json")
				file_path = os.path.join(absolute_path, relative_path)
				json_data = json.dumps(Category_count)
				with open(file_path, 'w') as f:
					f.write(json_data)

				# Table
				trace_table = go.Table(
					header=dict(values=list(df.columns),
					            align='left'),
					cells=dict(values=[df[k].tolist() for k in df.columns[0:]],
					           align='left'),
					domain=dict(x=[0, 0.45],
					            y=[0, 1])
				)

				layout = dict(xaxis1=dict(dict(domain=[0.5, 1], anchor='y1')),
				              yaxis1=dict(dict(domain=[0, 1], anchor='x1')),
				              title='staging Analysis')

				fig = go.Figure(data=[trace_table, fig_bar], layout=layout)

				trace_table2 = go.Table(
					header=dict(values=list(df_e1.columns),
					            align='left'),
					cells=dict(values=[df_e1[k].tolist() for k in df_e1.columns[0:]],
					           align='left'),
					columnwidth=[80, 400, 600, 80])
				layout2 = dict(title='General information of success clusters')
				fig2 = go.Figure(data=[trace_table2], layout=layout2)

				trace_table3 = go.Table(
					header=dict(values=list(df_e2.columns),
					            align='left'),
					cells=dict(values=[df_e2[k].tolist() for k in df_e2.columns[0:]],
					           align='left'),
					columnwidth=[80, 400, 600, 80])
				layout3 = dict(title='General information of fail clusters')
				fig3 = go.Figure(data=[trace_table3], layout=layout3)

				trace_table4 = go.Table(
					header=dict(values=list(df_other.columns),
					            align='left'),
					cells=dict(values=[df_other[k].tolist() for k in df_other.columns[0:]],
					           align='left'),
					columnwidth=[80, 400, 600, 80])
				layout4 = dict(title='General information of other clusters')
				fig4 = go.Figure(data=[trace_table4], layout=layout4)

				fig_html = pio.to_html(fig, full_html=False)
				success_analysis = pio.to_html(fig2, full_html=False)
				fail_analysis = pio.to_html(fig3, full_html=False)
				other_analysis = pio.to_html(fig4, full_html=False)
				print(cache.get('staging_feature_count'))
				cache.set('staging_flag_staging', True, timeout=86400)
				cache.set('staging_plot_staging', fig_html, timeout=86400)
				cache.set('staging_log_count', df.shape[0], timeout=86400)
				cache.set('staging_feature_count', feature_count, timeout=86400)
				cache.set('staging_success_analysis', success_analysis, timeout=86400)
				cache.set('staging_fail_analysis', fail_analysis, timeout=86400)
				cache.set('staging_other_analysis', other_analysis, timeout=86400)

				return render_template(
					"staging.html",
					name_staging=staging_file.filename,
					plot_staging=fig_html,
					log_count=df.shape[0],
					feature_count=feature_count,
					success_analysis=success_analysis,
					fail_analysis=fail_analysis,
					other_analysis=other_analysis,
				)

			except pd.errors.EmptyDataError:
				return render_template("error.html", message="File is empty")
		else:
			fig_html = cache.get('staging_flag_staging')
			log_count = cache.get('staging_log_count')
			feature_count = cache.get('staging_feature_count')
			success_analysis = cache.get('staging_success_analysis')
			fail_analysis = cache.get('staging_fail_analysis')
			other_analysis = cache.get('staging_other_analysis')
			return render_template(
				"staging.html",
				name_dataload=staging_file.filename,
				plot_dataload=fig_html,
				log_count=log_count,
				feature_count=feature_count,
				success_analysis=success_analysis,
				fail_analysis=fail_analysis,
				other_analysis=other_analysis,
			)
	else:
		return render_template("error.html", message="No staging file found")


@app.route('/upload')
def upload():
	return render_template('upload.html')


@app.route('/success', methods=['POST'])
def success():
	global uploaded_file
	global dataload_file
	global staging_file
	global firewall_file
	if request.method == 'POST':
		uploaded_file = request.files.get('file')
		if uploaded_file and uploaded_file.filename:
			filename = uploaded_file.filename
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
			uploaded_file.save(file_path)
			if 'dataload' in filename.lower():
				dataload_file = request.files.get('file')
				return redirect('/dataload')
			elif 'staging' in filename.lower():
				staging_file = request.files.get('file')
				return redirect('/staging')
			elif 'firewall' in filename.lower():
				firewall_file = request.files.get('file')
				return redirect('/firewall')
	print('meet error')
	return render_template("error.html", message="No file uploaded or unsupported file name.")


def log2df(file_path):
	def get_columns_from_file(filename):
		Columns = []
		with open(filename, 'r') as f:
			for line in f:
				if "Begin-----" in line:
					Columns = []
				elif "End-----" in line:
					return Columns
				else:
					Episode_type = line.split(': ')[0].strip()
					if Episode_type not in Columns:
						Columns.append(Episode_type)

	columns = get_columns_from_file(file_path)

	class Episode:
		def __init__(self):
			for column in columns:
				self.__dict__[column] = None

	def read_episodes_from_file(filename):
		Episodes = []
		with open(filename, 'r') as f:
			for line in f:
				if "Begin-----" in line:
					obj = Episode()
				elif "End-----" in line:
					Episodes.append(obj)
				else:
					temp = line.split(': ')[-1].strip()
					Episode_type = line.split(': ')[0].strip()
					obj.__dict__[Episode_type] = temp
		return Episodes

	Episodes = read_episodes_from_file(file_path)
	df_log = pd.DataFrame([vars(obj) for obj in Episodes])
	return df_log


def getFeature(df):
	feature_type = []

	def modify_features(line):
		features = []
		line = line[1:-1].replace("'", "").replace(" ", "").split(",")
		for feature in line:
			features.append(feature)
			if feature not in feature_type:
				feature_type.append(feature)
		return features

	All_Feature = [modify_features(line) for line in df["Features"]]

	df_feature = pd.DataFrame(columns=feature_type, index=list(range(len(All_Feature)))).fillna(0)
	for index, row in df_feature.iterrows():
		feature_dic = Counter(All_Feature[index])
		for key in feature_dic.keys():
			df_feature.loc[index, key] = feature_dic[key]

	return df_feature


def cluster(df_feature, df):
	org_Feature = df["Features"]
	X = np.array(df_feature)
	# get number of clusters and final model
	set_feature = len(set(org_Feature))
	overall = len(org_Feature)

	if set_feature / overall < 0.005 and set_feature <= 10:
		best_num = set_feature
		best_model = GaussianMixture(best_num, covariance_type='spherical', random_state=0)
	else:
		score = []

		for i in range(2, set_feature + 1):
			model = GaussianMixture(i, covariance_type='spherical', random_state=0)
			model.fit(X)
			y_kmeans = model.predict(X)
			score.append(metrics.silhouette_score(X, y_kmeans, metric='euclidean'))

		# plt.plot(list(range(2, set_feature + 1)), score)
		dif = np.diff(score)

		ar = []
		for i in range(len(dif) - 1):
			if dif[i] / 2 > dif[i + 1] > dif[i] / 4 or 0 < dif[i] < 0.002:
				ar.append(i)
		if ar:
			mi = min(ar[0], len(dif) - ar[-1])

			ma = 0
			ind = 0
			for index, item in enumerate(ar):
				if score[item + mi] - score[item - mi] > ma:
					ma = score[item + mi] - score[item - mi]
					ind = index
			best_num = ar[ind] + 1
		else:
			best_num = set_feature
	best_model = GaussianMixture(best_num, covariance_type='spherical', random_state=0)

	# clustering model and results
	final_model = best_model.fit(X)
	y_gaus = final_model.predict(X)

	df["Y_Gaussian"] = y_gaus


cred = {
	'apiKey': "AIzaSyAV6OrgXyST0l0om8I2TKcu1waF65Df68g",
	'authDomain': "boeing-a5759.firebaseapp.com",
	'projectId': "boeing-a5759",
	'storageBucket': "boeing-a5759.appspot.com",
	'messagingSenderId': "517543102347",
	'appId': "1:517543102347:web:1fb464c7a84a496e46d18e",
	'measurementId': "G-1F4FPN1FQC",
	'databaseURL': ''
}
# firebase_admin.initialize_app(cred)
firebase = pyrebase.initialize_app(cred)
auth = firebase.auth()


@app.context_processor
def my_context_processor():
	return {"user": session.get('user')}


# hook
@app.before_request
def my_before_request():
	user_id = session.get("user_id")
	if user_id:
		setattr(g, "user", user)
	else:
		setattr(g, "user", None)


# Register function
@app.route('/register', methods=['GET', 'POST'])
def register():
	# if request.method == 'GET':
	#     return render_template("register.html")
	if request.method == 'POST':
		email = request.form['email']
		password = request.form['password']
		try:
			user = auth.create_user_with_email_and_password(email=email, password=password)
			auth.send_email_verification(user['idToken'])
			return {'message': 'User created successfully', 'status': 'success'}, 200
		except Exception as e:
			if 'EMAIL_EXISTS' in str(e):
				return {'message': 'The email address is already in use.', 'status': 'error'}, 400
			else:
				return {'message': 'Error creating user', 'status': 'error'}, 400

	return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
	if request.method == 'POST':
		email = request.form.get('email')
		password = request.form.get('password')
		try:
			user = auth.sign_in_with_email_and_password(email, password)
			emailVerified = auth.get_account_info(user['idToken'])['users'][0]['emailVerified']
			if not emailVerified:
				return {'status': 'error', 'message': 'Please verify your email before logging in.'}, 400
			# return redirect(url_for('login'))
			else:
				session['user'] = user  # store user info in session
				setattr(g, "user", user)
				with app.app_context():
					cache.clear()
				return {'status': 'success', 'message': 'Logged in successfully'}, 200
		except Exception as e:
			print(e)
			if 'INVALID_EMAIL' or 'INVALID_PASSWORD' or 'EMAIL_NOT_FOUND' in str(e):
				return {'message': 'Invalid credentials', 'status': 'error'}, 400
			else:
				return {'status': 'error', 'message': 'An error occurred: {}'.format(e)}, 500
	else:
		return render_template('login.html')


@app.route('/logout')
def logout():
	session.pop('user', None)
	setattr(g, "user", None)
	cache.set('name', 'value')
	cache.clear()
	return redirect('/')


if __name__ == '__main__':
	app.config['UPLOAD_FOLDER'] = 'uploads'
	if not os.path.exists(app.config['UPLOAD_FOLDER']):
		os.makedirs(app.config['UPLOAD_FOLDER'])
	app.run(debug=True, use_reloader=False)
