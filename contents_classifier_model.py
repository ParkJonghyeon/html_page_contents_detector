import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

import contents_classifier_common as common


class ContentsClassifierModel:
    def __init__(self, model_name='logistic', training_document=''):
        self.model_name = model_name
        self.training_document = training_document
        self.vect = TfidfVectorizer(token_pattern=r'\w+', lowercase=True, stop_words=common.ALL_STOP_WORD)
        self.X = self.vect.fit_transform(self.training_document)
        self.Y = common.CONTENTS
        # X.todense()
        # sentences에는 각 서비스 분류의 문서들이 포함
        # 하나의 카테고리하에 3-4종류의 html 문서가 하나로 통합 된 것 한개씩?
        self.model = self.make_model()


    # model 생성 함수. 필요에 따라 import 후 해당 모델 생성 분기 추가
    def make_model(self):
        if self.model_name == 'logistic':
            return_model = LogisticRegression()
            return_model.fit(self.X, self.Y)
        elif self.model_name == 'naive':
            return_model = GaussianNB()
            return_model.fit(self.X.toarray(), self.Y)
        elif self.model_name == 'decision':
            return_model = DecisionTreeClassifier()
            return_model.fit(self.X, self.Y)
        elif self.model_name == 'knn':
            return_model = KNeighborsClassifier(n_neighbors=3)
            return_model.fit(self.X, self.Y)
        elif self.model_name == 'svm':
            return_model = SVC(gamma='auto')
            return_model.fit(self.X, self.Y)
        elif self.model_name == 'random':
            return_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=0)
            return_model.fit(self.X, self.Y)
        return return_model


    def reselect_model(self, model_name):
        self.model_name = model_name
        self.model = self.make_model()


    # 예측 함수. 예측하려는 문서파일의 경로를 입력받아 결과를 출력.
    # shutil로 문서의 복사본을 지정위치로 이동할 수 있음
    def input_pred(self, pred_text, mode):
        # vect에서 토큰화, stop word, lower 등 적용되고 있음
        X_pred = self.vect.transform([pred_text])
        if self.model_name == 'naive':
            y_pred = self.model.predict(X_pred.toarray())
        else:
            y_pred = self.model.predict(X_pred)
        if mode == 'all':
            shutil.copy(inputfile_route, 'data_set/auto_labeling/'+y_pred[0])
        return y_pred[0]
