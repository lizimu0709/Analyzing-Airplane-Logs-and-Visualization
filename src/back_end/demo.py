from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import manifold
from numpy import unique
from numpy import where
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['inputFile']
    file_path = file.filename
    print(file_path, file)
    if not file:
        return render_template('index.html', error="No file selected.")

    class Episode:
        def __init__(self):
            for column in get_columns_from_file(file_path):
                self.__dict__[column] = None

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

    def modify_features(line):
        features = []
        line = line[1:-1].replace("'","").replace(" ","").split(",")
        for feature in line:
            features.append(feature)
        return features

    Episodes = read_episodes_from_file(file_path)
    df_log = pd.DataFrame([vars(obj) for obj in Episodes])

    All_Feature = []
    for line in df_log["Features"]:
        All_Feature.append(modify_features(line))

    # can optimize here
    feature_type = []
    for sublst in All_Feature:
        for elem in sublst:
            if elem not in feature_type:
                feature_type.append(elem)

    for col in feature_type:
        df_log[col] = 0

    # can optimize
    for index, row in df_log.iterrows():
        feature_dic = Counter(modify_features(row["Features"]))
        for key in feature_dic.keys():
            df_log.loc[index,key] = feature_dic[key]

    used = df_log[feature_type]
    X = np.array(used)

    #t-SNE
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # normilization
    models = GaussianMixture(24, covariance_type='spherical', random_state=0).fit(X)

    #tmp_model = models[2]
    y_gaus = models.predict(X)
    clusters = unique(y_gaus)

    fig1, ax1 = plt.subplots()
    for cluster in clusters:
        row_ix = where(y_gaus == cluster)
        ax1.scatter(X_norm[row_ix, 0], X_norm[row_ix, 1])
    fig1.savefig('static/result_1.png')

    fig2, ax2 = plt.subplots()
    df_log["Y_Gaussian"] = y_gaus
    data = df_log["Y_Gaussian"].value_counts()
    ax2.bar(data.index, data.values)
    fig2.savefig('static/result_2.png')

    df_log = df_log.drop(columns=feature_type)

    return render_template('result.html', tables=[df_log.to_html(classes='data')],
                           titles=df_log.columns.values, filename=file_path.split(".")[0])


if __name__ == '__main__':
    app.run(debug=True)
