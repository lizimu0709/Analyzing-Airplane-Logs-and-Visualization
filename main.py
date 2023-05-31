import pandas as pd
import os
import json
import plotly.io as pio
from flask import Flask, render_template, request, redirect, session, g
from flask_caching import Cache
import plotly.graph_objects as go
import pyrebase
from algorithm import log2df, results_by_Category

app = Flask(__name__, template_folder='src/template', static_folder='src/static')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
app.secret_key = 'capstone_project'


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/dataload')
@cache.memoize()
def dataload():
	user = session.get('user')
	filename = session.get('dataload_filename')
	if filename and user:
		absolute_path = os.path.dirname(__file__)
		full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		df = log2df(full_path)
		if df is None:
			return render_template('error.html', message='Log file is empty')
		try:
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
				domain=dict(x=[0, 1],
				            y=[0, 0.45])
			)

			layout = dict(
						xaxis1=dict(domain=[0, 1], anchor='y1'),
						yaxis1=dict(domain=[0.5, 1], anchor='x1'),
						xaxis2=dict(domain=[0, 1], anchor='y2'),
						yaxis2=dict(domain=[0, 0.5], anchor='x2'),
						title='Logs per Day'
						)
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

			fig_html = pio.to_html(fig, full_html=False, default_height='600px')
			success_analysis = pio.to_html(fig2, full_html=False)
			fail_analysis = pio.to_html(fig3, full_html=False)
			other_analysis = pio.to_html(fig4, full_html=False)


			return render_template(
				"dataload.html",
				name_dataload=filename,
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
		return render_template("error.html", message="No dataload file found")


@app.route('/firewall')
@cache.memoize()
def firewall():
	user = session.get('user')
	filename = session.get('firewall_filename')
	if filename and user:
		absolute_path = os.path.dirname(__file__)
		full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		df = log2df(full_path)
		if df is None:
			return render_template('error.html', message='Log file is empty')
		try:
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
			              title='Firewall Analysis')

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


			return render_template(
				"firewall.html",
				name_firewall=filename,
				plot_firewall=fig_html,
				log_count=df.shape[0],
				feature_count=feature_count,
				other_analysis=other_analysis,
			)
		except pd.errors.EmptyDataError:
			return render_template("error.html", message="File is empty")

	else:
		return render_template("error.html", message="No firewall file found")


@app.route('/staging')
@cache.memoize()
def staging():
	user = session.get('user')
	filename = session.get('staging_filename')
	if filename and user:
		absolute_path = os.path.dirname(__file__)
		full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		df = log2df(full_path)
		if df is None:
			return render_template('error.html', message='Log file is empty')
		try:
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
				domain=dict(x=[0, 1],
				            y=[0, 0.45])
			)

			layout = dict(
						xaxis1=dict(domain=[0, 1], anchor='y1'),
						yaxis1=dict(domain=[0.5, 1], anchor='x1'),
						xaxis2=dict(domain=[0, 1], anchor='y2'),
						yaxis2=dict(domain=[0, 0.5], anchor='x2'),
						title='Logs per Day'
						)
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

			fig_html = pio.to_html(fig, full_html=False, default_height='600px')
			success_analysis = pio.to_html(fig2, full_html=False)
			fail_analysis = pio.to_html(fig3, full_html=False)
			other_analysis = pio.to_html(fig4, full_html=False)


			return render_template(
				"staging.html",
				name_staging=filename,
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
		return render_template("error.html", message="No staging file found")


@app.route('/upload')
def upload():
	return render_template('upload.html')


@app.route('/success', methods=['POST'])
def success():
	if request.method == 'POST':
		uploaded_file = request.files.get('file')
		if uploaded_file and uploaded_file.filename:
			filename = uploaded_file.filename
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			uploaded_file.save(file_path)
			if 'dataload' in filename.lower():
				session['dataload_filename'] = filename
				cache.delete_memoized(dataload)
				return redirect('/dataload')
			elif 'staging' in filename.lower():
				session['staging_filename'] = filename
				cache.delete_memoized(staging)
				return redirect('/staging')
			elif 'firewall' in filename.lower():
				session['firewall_filename'] = filename
				cache.delete_memoized(firewall)
				return redirect('/firewall')

	return render_template("error.html", message="No file uploaded or unsupported file name.")



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
	user = session.get("user")
	if user:
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
				# setattr(g, "user", user)
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
	session.clear()
	setattr(g, "user", None)
	cache.clear()

	# delete uploaded files
	files = os.listdir(app.config['UPLOAD_FOLDER'])
	for file in files:
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
		if os.path.isfile(file_path):
			os.remove(file_path)

	return redirect('/')


if __name__ == '__main__':
	app.config['UPLOAD_FOLDER'] = 'uploads'
	if not os.path.exists(app.config['UPLOAD_FOLDER']):
		os.makedirs(app.config['UPLOAD_FOLDER'])
	app.run(debug=True, use_reloader=False)
