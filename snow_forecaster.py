import csv
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
from sklearn.cross_validation import cross_val_score

class SnowForecast:

    CLF_NAMES = ["LinearSVC","AdaBoostClassifier","ExtraTreesClassifier" ,
                 "GradientBoostingClassifier","RandomForestClassifier"]

    def __init__(self):
        u"""各インスタンス変数を初期化"""
        self.clf = None
        self.data = {"target" : [], "data" : []}
        self.weather_data = None
        self.days_data = {}
        self.days_snow = {}

    def load_csv(self):
        u"""学習用CSVファイルを読み込む"""
        with open("sample_data/data.csv", "r") as f:
            reader = csv.reader(f)
            accumulation_yesterday = 0
            temp_yeaterday = 0
            date_yesterday = ""
            
            for row in reader:
                if row[4] == "":
                    continue

                daytime = row[0]
                date = daytime.split(" ")[0]
                temp = int(float(row[1]))
                accumulation = int(row[4])
                wind_speed = float(row[7])
                precipitation = float(row[12])

                if date_yesterday != "":
                    # [温度, 降水量, 昨日の温度, 昨日の積雪量]
                    sample = [temp, precipitation, temp_yeaterday, accumulation_yesterday]
                    exist = self.accumulation_exist(accumulation)
                    self.data["data"].append(sample)
                    self.data["target"].append(exist)
                    self.days_data[daytime] = sample
                    self.days_snow[daytime] = exist

                if date_yesterday != date:
                    accumulation_yesterday = accumulation
                    temp_yeaterday = temp
                    date_yesterday = date

        return self.data

    def is_snow_exist(self, daytime_str):
        u"""雪が積もったなら1、積もらなければ0を返す."""
        return self.days_snow[daytime_str]

    def predict_with_date(self, daytime_str):
        u"""与えられた日付のデータを使って積雪の有無を予想する。"""
        sample = self.days_data[daytime_str]
        temp = sample[0]
        precipitation = sample[1]
        temp_yeaterday = sample[2]
        accumulation_yesterday = sample[3]
        return self.predict(temp, precipitation, temp_yeaterday, accumulation_yesterday)

    def predict(self, temp, precipitation, temp_yeaterday, accumulation_yesterday):
        u"""与えられたパラメータを使って積雪の有無を予想する。"""
        return self.clf.predict([[temp, precipitation, temp_yeaterday, accumulation_yesterday]])[0]

    def train_data(self):
        u"""学習を行うためのデータを返す。すでに読み込み済みならそれを返し、まだならCVSファイルから読み込む"""
        if self.weather_data is None:
            self.weather_data = self.load_csv()

        return self.weather_data

    def accumulation_exist(self, accumulation):
        u"""積雪量(cm)を受け取り、積雪があれば１、なければ0を返す"""
        if accumulation > 0:
            return 1
        else:
            return 0

    def best_score_clf(self):
        u"""各学習モデルのタイプ別にスコアを計算し、もっともスコアの高いタイプのオブジェクトをインスタンス変数にとっておく."""
        features = self._features()
        labels = self._labels()

        # 今回は特徴量の算出に4量しか使わないので特徴量の削減は行わない。よって以下はコメントにしておく。
        # lsa = TruncatedSVD(3)
        # reduced_features = lsa.fit_transform(features)

        best = LinearSVC()
        best_name = self.CLF_NAMES[0]
        best_score = 0

        for clf_name in self.CLF_NAMES:
            clf    = eval("%s()" % clf_name) 
            scores = cross_val_score(clf, features, labels, cv=5) # 特徴量削減した場合は reduced_features を使う
            score  = sum(scores) / len(scores)  #モデルの正解率を計測
            print("%sのスコア:%s" % (clf_name,score))
            if score >= best_score:
                best = clf
                best_name = clf_name
                best_score = score

        print("------\n使用するモデル: %s" % best_name)
        return clf

    def train(self):
        u"""学習を実行する。実際の学習の前にどのモデルを使うかを判定し、自動的に選択させる。"""
        self.clf = self.best_score_clf()
        self.clf.fit(self._features(), self._labels())

    def _features(self):
        u"""学習データを返す。"""
        weather = self.train_data()
        return weather["data"]

    def _labels(self):
        u"""結果のラベルを返す。"""
        weather = self.train_data()
        return weather["target"]

    def judge(self, datetime_str):
        u"""日付文字列を受け取って積雪判定を行う。"""
        print("------")
        result = forecaster.predict_with_date(datetime_str)
        print("%s: 予想:%s 実際:%s" % (datetime_str, result, forecaster.is_snow_exist(datetime_str)))

        if result == 1:
            print("雪が積もります")
        else:
            print("雪は積もらないです")

if __name__ == "__main__":
    forecaster = SnowForecast()
    forecaster.train()
    
    #####################################################
    # 日付を指定して、学習に使用したパラメータを与えて判定する。
    #####################################################
    forecaster.judge("2006/2/19 00:00:00")
    forecaster.judge("2012/2/2 00:00:00")
    forecaster.judge("2014/2/2 13:00:00")
    forecaster.judge("2015/2/28 00:00:00")
    
    #######################################
    # パラメータを直接与えて予測させてみる。
    #######################################
    print("------")
    temp = 0.0
    precipitation = 0
    temp_yeaterday = 3.0
    accumulation_yesterday = 2 
    result = forecaster.predict(temp, precipitation, temp_yeaterday, accumulation_yesterday)
    print("[温度:%s] [降水量:%s] [昨日の温度:%s] [昨日の積雪量:%s]" %
          (temp, precipitation, temp_yeaterday, accumulation_yesterday))
    
    print("判定結果: %s" % result)
    
    if result == 1:
        print("雪が積もります")
    else:
        print("雪は積もらないです")
        
    #########################################################
    # パラメータを直接与えて予測させてみる(昨日の温度を-3.0℃に変更)。
    #########################################################
    print("------")
    temp_yeaterday = -3.0
    result = forecaster.predict(temp, precipitation, temp_yeaterday, accumulation_yesterday)
    print("[温度:%s] [降水量:%s] [昨日の温度:%s] [昨日の積雪量:%s]" %
          (temp, precipitation, temp_yeaterday, accumulation_yesterday))
    
    print("判定結果: %s" % result)
    
    if result == 1:
        print("雪が積もります")
    else:
        print("雪は積もらないです")
        
    print("------")
