from itertools import cycle
from scipy import interp

import os
import shutil
import contents_classifier_common as common
import contents_classifier_utils as ccutils
import contents_classifier_model as ccmodel

import numpy as np
from sklearn.metrics import classification_report,confusion_matrix


LABELED_HTML = {}
for cont_name in common.CONTENTS:
    tmp = os.listdir(common.DATA_DIR+cont_name)
    for key in tmp:
        LABELED_HTML[common.DATA_DIR+cont_name+'/'+key]=cont_name


TEST_DATA_LIST = []
for dir_name in common.CONTENTS:
	dir_route = common.DATA_DIR + dir_name + '/'
	file_list = os.listdir(dir_route)
	for training_data in file_list:
		TEST_DATA_LIST.append(dir_route+training_data)


contents_reader = ccutils.ContentsClassifierUtils(filtering_word_num = 4)
documents = contents_reader.read_text_data('read text')
tfidf_gen_model = ccmodel.ContentsClassifierModel(training_document = documents)


test_data_label = []
for testfile in TEST_DATA_LIST:
    test_data_label.append(LABELED_HTML[testfile])


X_test = None
for testfile in TEST_DATA_LIST:
    test_data = tfidf_gen_model.vect.transform([contents_reader.extract_text_from_html( contents_reader.read_html(testfile) )]).toarray()
    if X_test is None:
        X_test = test_data
    else:
        X_test = np.append(X_test, test_data,axis=0)


model_names = ['naive', 'svm', 'logistic']
for idx in range(len(model_names)):
    classification_model = ccmodel.ContentsClassifierModel(model_names[idx], documents)
    predict_data_label = classification_model.model.predict(X_test)
    print("# Model Name : ", model_names[idx], "\n")
    print("Classification Report")
    print(classification_report(test_data_label,predict_data_label))
    print("Confusion Matrix")
    print(confusion_matrix(test_data_label,predict_data_label),"\n\n")
