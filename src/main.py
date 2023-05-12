
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
from collections import Counter
from sklearn import manifold, metrics
from numpy import unique, where
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
from plotly.tools import mpl_to_plotly
from flask_restful import Api, Resource
from plotly.subplots import make_subplots
from ipywidgets import widgets
import plotly.figure_factory as ff

app = Flask(__name__, template_folder='react-app/public', static_folder='react-app/public/static')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

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
        calculated = cache.get('flag_dataload')
        if calculated is None:
            try:
                file_name, fig_html, clusters_html = calculate(dataload_file)
                calculated = True
                cache.set('flag_dataload', calculated, timeout=86400)
                cache.set('fig_html_dataload', fig_html, timeout=86400)
                cache.set('clusters_html_dataload', clusters_html, timeout=86400)
                return render_template(
                    "dataload.html", 
                    name_dataload = file_name, 
                    plot_dataload = fig_html,
                    cluster_dataload = clusters_html)
            except pd.errors.EmptyDataError:
                return render_template("error.html", message="File is empty")
        else:
            fig_html = cache.get('fig_html_dataload')
            clusters_html = cache.get('clusters_html_dataload')
            return render_template(
                    "dataload.html", 
                    name_dataload = dataload_file.filename, 
                    plot_dataload = fig_html,
                    cluster_dataload = clusters_html)
    else:
        return render_template("error.html", message="No dataload file found")

@app.route('/firewall')
def firewall():
    global firewall_file
    if firewall_file:
        calculated = cache.get('flag_firewall')
        if calculated is None:
            try:
                file_name, fig_html, clusters_html = calculate(firewall_file)
                calculated = True
                cache.set('flag_firewall', calculated, timeout=86400)
                cache.set('fig_html_firewall', fig_html, timeout=86400)
                cache.set('clusters_html_firewall', clusters_html, timeout=86400)
                return render_template(
                    "firewall.html", 
                    name_firewall = file_name, 
                    plot_firewall = fig_html,
                    cluster_firewall = clusters_html)
            except pd.errors.EmptyDataError:
                return render_template("error.html", message="File is empty")
        else:
            fig_html = cache.get('fig_html_firewall')
            clusters_html = cache.get('clusters_html_firewall')
            return render_template(
                    "firewall.html", 
                    name_firewall = firewall_file.filename, 
                    plot_firewall = fig_html,
                    cluster_firewall = clusters_html)
    else:
        return render_template("error.html", message="No firewall file found")

@app.route('/staging')
def staging():
    global staging_file
    if staging_file:
        calculated = cache.get('flag_staging')
        if calculated is None:
            try:
                file_name, fig_html, clusters_html = calculate(staging_file)
                calculated = True
                cache.set('flag_staging', calculated, timeout=86400)
                cache.set('fig_html_staging', fig_html, timeout=86400)
                cache.set('clusters_html_staging', clusters_html, timeout=86400)
                return render_template(
                    "staging.html", 
                    name_staging = file_name, 
                    plot_staging= fig_html,
                    cluster_staging = clusters_html)
            except pd.errors.EmptyDataError:
                return render_template("error.html", message="File is empty")
        else:
            fig_html = cache.get('fig_html_staging')
            clusters_html = cache.get('clusters_html_staging')
            return render_template(
                    "staging.html", 
                    name_staging = staging_file.filename, 
                    plot_staging = fig_html,
                    cluster_staging = clusters_html)
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
            if 'dataload' in filename.lower():
                dataload_file = request.files.get('file')
                return redirect('/dataload')
            elif 'staging' in filename.lower():
                staging_file = request.files.get('file')
                return redirect('/staging')
            elif 'firewall' in filename.lower():
                firewall_file = request.files.get('file')
                return redirect('/firewall')
    return render_template("error.html", message="No file uploaded or unsupported file name.")

def calculate(curr_file):
    absolute_path = os.path.dirname(__file__)
    full_path = absolute_path + "\\" + curr_file.filename

    # Format data to a dataframe
    df = log2df(full_path)
    df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
    df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
    logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')

    fig_bar = go.Bar(x=logs_per_day["Episode Start Date"], y=logs_per_day["Count"], xaxis="x1", yaxis="y1")

    clusters = cluster(df)

    # Table
    trace_table = go.Table(
        header=dict(values=list(df.columns),
        align='left'),
        cells=dict(values=[df[k].tolist() for k in df.columns[0:]],
        align='left'),
        domain=dict(x=[0, 0.45],
        y=[0, 1])
    )

    layout = dict(xaxis1=dict( dict(domain=[0.5, 1], anchor='y1')),
                yaxis1=dict( dict(domain=[0, 1], anchor='x1')),
                title=f"{curr_file.filename} Analysis")

    fig = go.Figure(data = [trace_table, fig_bar], layout = layout)

    fig_html = pio.to_html(fig, full_html=False)
    clusters_html = pio.to_html(clusters, full_html=False)

    return(curr_file.filename, fig_html, clusters_html)

def log2df(file_path):
    global df_log

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

def cluster(df_log):
    feature_type = []

    def modify_features(line):
        features = []
        line = line[1:-1].replace("'", "").replace(" ", "").split(",")
        for feature in line:
            features.append(feature)
            if feature not in feature_type:
                feature_type.append(feature)
        return features

    All_Feature = []
    org_Feature = df_log["Features"]
    for line in org_Feature:
        All_Feature.append(modify_features(line))

    df_feature = pd.DataFrame(columns=feature_type, index=list(range(len(All_Feature)))).fillna(0)
    for index, row in df_feature.iterrows():
        feature_dic = Counter(All_Feature[index])
        for key in feature_dic.keys():
            df_feature.loc[index, key] = feature_dic[key]

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

        dif = np.diff(score)

        ar = []
        for i in range(len(dif) - 1):
            if dif[i] / 2 > dif[i + 1] > dif[i] / 4:
                ar.append(i)
        mi = min(ar[0], len(dif) - ar[-1])

        ma = 0
        ind = 0
        for index, item in enumerate(ar):
            if score[item + mi] - score[item - mi] > ma:
                ma = score[item + mi] - score[item - mi]
                ind = index
        best_num = ar[ind] + 1
        best_model = GaussianMixture(best_num, covariance_type='spherical', random_state=0)

    # clustering model and results
    final_model = best_model.fit(X)
    y_gaus = final_model.predict(X)

    # t-SNE
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # normilization

    # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    fig = make_subplots(rows=1, cols=2)

    clusters = unique(y_gaus)
    for cluster in clusters:
        row_ix = where(y_gaus == cluster)
        row_ix = list(row_ix[0])
        df = pd.DataFrame(X_norm, columns = ['x', 'y'])
        name = 'Cluster ' + str(cluster)
        print(name)
        fig.append_trace(go.Scatter(x=df[df.index.isin(row_ix)]['x'],
                                         y=df[df.index.isin(row_ix)]['y'],
                                         mode='markers',
                                         name=name
                                         ), row=1, col=2)

    df_log["Y_Gaussian"] = y_gaus
    data = df_log["Y_Gaussian"].value_counts()

    fig_bar = go.Bar(x=data.index, y=data.values, name='Cluster')

    fig.append_trace(fig_bar, row=1, col=1)
    
    return fig

if __name__ == '__main__':  
    app.run(debug=True)