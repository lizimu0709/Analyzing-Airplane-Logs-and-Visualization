import pandas as pd
import os
import json
import plotly.io as pio
from flask import Flask, render_template, request, redirect
from flask_navigation import Navigation
import plotly.graph_objects as go
from flask_caching import Cache
from collections import Counter
from sklearn import manifold, metrics
from numpy import unique, where
from sklearn.mixture import GaussianMixture
import numpy as np
from plotly.subplots import make_subplots
from flask_restful import Api, Resource

app = Flask(__name__, template_folder='react-app/public', static_folder='react-app/public/static')
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


def classify_features(features):
    if 'E1' in features and 'E2' not in features:
        return 'Success'
    elif 'E2' in features and 'E1' not in features:
        return 'Fail'
    else:
        return 'Other'


def results_by_Category(df_log):
    Category_count = {}
    df_log['Category'] = df_log['Features'].apply(classify_features)
    for Category in ['Success', 'Fail', 'Other']:
        df_temp = df_log[df_log['Category'] == Category]
        if len(df_temp) != 0:
            cluster(getFeature(df_temp), df_temp)
            Category_count[Category] = (df_temp["Y_Gaussian"].value_counts().to_dict())
    return Category_count

@app.route('/dataload')
@cache.cached(timeout=86400)
def dataload():
    global dataload_file
    print(dataload_file)
    if dataload_file:
        try:
            absolute_path = os.path.dirname(__file__)
            # need fix (save the file to specific route)
            full_path = absolute_path + "\\" + dataload_file.filename

            # Format data to a dataframe
            df = log2df(full_path)
            df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
            df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
            logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')

            fig_bar = go.Bar(x=logs_per_day["Episode Start Date"], y=logs_per_day["Count"], xaxis="x1", yaxis="y1")

            Category_count = results_by_Category(df)
            relative_path = "react-app\public\static\dataload.json"
            file_path = absolute_path + '\\' + relative_path
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

            fig_html = pio.to_html(fig, full_html=False)

            return render_template(
                "dataload.html",
                name_dataload=dataload_file.filename,
                plot_dataload=fig_html)

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

            # Format data to a dataframe
            df = log2df(full_path)
            df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
            df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
            logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')

            fig_bar = go.Bar(x=logs_per_day["Episode Start Date"], y=logs_per_day["Count"], xaxis="x1", yaxis="y1")

            Category_count = results_by_Category(df)
            relative_path = r"react-app\public\static\firewall.json"
            file_path = absolute_path + '\\' + relative_path
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

            fig_html = pio.to_html(fig, full_html=False)

            return render_template(
                "firewall.html",
                name_firewall=firewall_file.filename,
                plot_firewall=fig_html)

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

            # Format data to a dataframe
            df = log2df(full_path)
            df['Episode Start Date'] = pd.to_datetime(df['Episode Start Date'])
            df['Episode Start Date'] = df['Episode Start Date'].dt.strftime('%Y-%m-%d')
            logs_per_day = df.groupby(df['Episode Start Date']).size().reset_index(name='Count')

            fig_bar = go.Bar(x=logs_per_day["Episode Start Date"], y=logs_per_day["Count"], xaxis="x1", yaxis="y1")

            Category_count = results_by_Category(df)
            relative_path = "react-app\public\static\staging.json"
            file_path = absolute_path + '\\' + relative_path
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

            fig_html = pio.to_html(fig, full_html=False)

            return render_template(
                "staging.html",
                name_staging=staging_file.filename,
                plot_staging=fig_html)

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



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
