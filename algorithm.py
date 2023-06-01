import numpy as np
import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.mixture import GaussianMixture


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
        try:
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
        except FileNotFoundError:
            return None
        return Episodes

    Episodes = read_episodes_from_file(file_path)
    if Episodes is None:
        return None
    df_log = pd.DataFrame([vars(obj) for obj in Episodes])
    return df_log


def cluster(df_feature, df):
    org_Feature = df["Features"]
    X = np.array(df_feature)
    # get number of clusters and final model
    set_feature = len(set(org_Feature))
    overall = len(org_Feature)

    if set_feature / overall < 0.005 and set_feature <= 10:
        best_num = set_feature
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
