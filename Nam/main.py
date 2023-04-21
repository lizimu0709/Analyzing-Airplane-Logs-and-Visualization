
from distutils.log import debug
from fileinput import filename
import pandas as pd
import os
import plotly.express as px
import plotly.io as pio
from flask import Flask, render_template, request, redirect
from flask_navigation import Navigation
import plotly.graph_objects as go
from flask_caching import Cache
from flask_restful import Api, Resource

app = Flask(__name__)
nav = Navigation(app)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

nav.Bar('top', [nav.Item('Home', 'index'),
                nav.Item('Dataload', 'dataload'),
                nav.Item('Firewall', 'firewall'),
                nav.Item('Staging', 'staging'),
                nav.Item('Upload', 'upload'),
                nav.Item('Login', 'login')
                ])

uploaded_file = None
dataload_file = None
firewall_file = None
staging_file = None
  
@app.route('/')  
def index():  
    return render_template('index.html')  
  
@app.route('/dataload')
def dataload():
    global dataload_file
    if dataload_file:
        try:
            absolute_path = os.path.dirname(__file__)
            full_path = absolute_path + "\\" + dataload_file.filename
            df = pd.read_csv(full_path)
            df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
            df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
            logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')
            fig = px.line(logs_per_day, x="Episode Start Date", y="Count", title="Logs per Day")
            fig_html = pio.to_html(fig, full_html=False)

            table = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns),
                align='left'),
                cells=dict(values=[df[k].tolist() for k in df.columns[0:]],
                align='left')
            )])
            table_html = pio.to_html(table, full_html=False)
            return render_template(
                "dataload.html", 
                name_dataload = dataload_file.filename, 
                data_dataload = table_html,
                plot_dataload = fig_html)
        except pd.errors.EmptyDataError:
            return render_template("error.html", message="File is empty")
    else:
        return render_template("error.html", message="No dataload file found")
    

@app.route('/firewall')
@cache.cached(timeout=86400)
def firewall():
    global firewall_file
    if firewall_file:
        try:
            absolute_path = os.path.dirname(__file__)
            full_path = absolute_path + "\\" + firewall_file.filename
            df = pd.read_csv(full_path)
            df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
            df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
            logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')
            fig = px.line(logs_per_day, x="Episode Start Date", y="Count", title="Logs per Day")
            fig_html = pio.to_html(fig, full_html=False)

            table = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns),
                align='left'),
                cells=dict(values=[df[k].tolist() for k in df.columns[0:]],
                align='left')
            )])
            table_html = pio.to_html(table, full_html=False)
            return render_template(
                "firewall.html", 
                name_firewall = firewall_file.filename, 
                data_firewall = table_html,
                plot_firewall = fig_html)
        except pd.errors.EmptyDataError:
            return render_template("error.html", message="File is empty")
    else:
        return render_template("error.html", message="No firewall file found")

@app.route('/staging')
@cache.cached(timeout=86400)
def staging():
    global staging_file
    if staging_file:
        try:
            absolute_path = os.path.dirname(__file__)
            full_path = absolute_path + "\\" + staging_file.filename
            df = pd.read_csv(full_path)
            df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
            df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
            logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')
            fig = px.line(logs_per_day, x="Episode Start Date", y="Count", title="Logs per Day")
            fig_html = pio.to_html(fig, full_html=False)

            table = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns),
                align='left'),
                cells=dict(values=[df[k].tolist() for k in df.columns[0:]],
                align='left')
            )])
            table_html = pio.to_html(table, full_html=False)
            return render_template(
                "staging.html", 
                name_staging = staging_file.filename, 
                data_staging = table_html,
                plot_staging = fig_html)
        except pd.errors.EmptyDataError:
            return render_template("error.html", message="File is empty")
    else:
        return render_template("error.html", message="No staging file found")

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/success', methods = ['POST'])  
def success():
    global uploaded_file
    global dataload_file
    global staging_file
    global firewall_file
    if request.method == 'POST':
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename:
            filename = uploaded_file.filename
            if filename.lower() == 'dataload.csv':
                dataload_file = request.files.get('file')
                return redirect('/dataload')
            elif filename.lower() == 'staging.csv':
                staging_file = request.files.get('file')
                return redirect('/staging')
            elif filename.lower() == 'firewall.csv':
                firewall_file = request.files.get('file')
                return redirect('/firewall')
    return render_template("error.html", message="No file uploaded or unsupported file name.")

if __name__ == '__main__':  
    app.run(debug=True)