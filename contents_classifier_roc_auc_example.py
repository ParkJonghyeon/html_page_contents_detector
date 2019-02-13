from itertools import cycle
from scipy import interp

import os
import shutil
import contents_classifier_common as common
import contents_classifier_utils as ccutils
import contents_classifier_model as ccmodel
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


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

binary_test_data_label = label_binarize(test_data_label, classes=common.CONTENTS)
n_classes = len(common.CONTENTS)

X_test = None
for testfile in TEST_DATA_LIST:
    test_data = tfidf_gen_model.vect.transform([contents_reader.extract_text_from_html( contents_reader.read_html(testfile) )]).toarray()
    if X_test is None:
        X_test = test_data
    else:
        X_test = np.append(X_test, test_data,axis=0)


total_fpr = dict()
total_tpr = dict()
total_roc_auc = dict()

model_names = ['naive', 'svm', 'logistic']
for idx in range(len(model_names)):
    classification_model = ccmodel.ContentsClassifierModel(model_names[idx], documents)
    if model_names[idx] in ['naive','knn','decision','random']:
        y_score = model.model.predict_proba(X_test)
    else:
        y_score = model.model.decision_function(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(binary_test_data_label[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    #average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    total_fpr[model_names[idx]] = all_fpr
    total_tpr[model_names[idx]] = mean_tpr
    total_roc_auc[model_names[idx]] = auc(total_fpr[model_names[idx]], total_tpr[model_names[idx]])


model_names_fix = ['Naive Bayesian', 'SVM', 'Logistic Regression']

# Plot all ROC curves
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'orange'])
for i, color in zip(range(len(model_names)), colors):
    plt.plot(total_fpr[model_names[i]], total_tpr[model_names[i]], color=color, lw=2,
             label='ROC curve of '+model_names_fix[i]+' (area = {0:0.2f})'.format(total_roc_auc[model_names[i]]) )

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

