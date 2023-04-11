from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import manifold, metrics
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
    
    # get number of clusters and final model
    org_feature = df_log["Features"]
    set_feature = len(set(org_feature))
    overall = len(org_feature)

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
            if dif[i] / 2 > dif[i + 1] and dif[i] / 4 < dif[i + 1]:
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
    
    #t-SNE
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # normilization
    fig1, ax1 = plt.subplots()
    clusters = unique(y_gaus)
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
