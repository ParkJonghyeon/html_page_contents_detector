'''
LABELED_HTML = {}
for cont_name in common.CONTENTS:
    cont_val = cont_name
    if cont_val in common.BLACK_MARKET_CONTENTS:
        cont_val = 'black_market'
    elif cont_val in common.LEGAL_CONTENTS:
        cont_val = 'legal'
    tmp = os.listdir(common.DATA_DIR+cont_name)
    for key in tmp:
        LABELED_HTML[common.DATA_DIR+cont_name+'/'+key]=cont_val


ORIG_HTML = {}
for cont_name in common.CONTENTS:
    cont_val = cont_name
    tmp = os.listdir(common.DATA_DIR+cont_name)
    for key in tmp:
        ORIG_HTML[common.DATA_DIR+cont_name+'/'+key]=cont_val


TEST_DATA_LIST = []


#all data use test data. if not pass this for
#for dir_name in common.CONTENTS:
#	dir_route = common.DATA_DIR + dir_name + '/'
#	file_list = os.listdir(dir_route)
#	for training_data in file_list:
#		TEST_DATA_LIST.append(dir_route+training_data)


#contents_reader = ccutils.ContentsClassifierUtils(filtering_word_num = 4)
#documents = contents_reader.read_text_data('read text')


contents_reader = ccutils.ContentsClassifierUtils(filtering_word_num = 4)
import codecs
import random

def read_text_data_for_training(read_file_num):
    return_documents = []
    tmp_doc_list = []
    for dir_name in common.CONTENTS:
        dir_route = common.DATA_DIR + dir_name
        tmp_data = read_training_html(dir_route+'/', read_file_num)
        tmp_doc_list.append(tmp_data)
        return_documents.append( contents_reader.text_clensing(tmp_data) )
    # clensing이 되지 않은 원본 텍스트를 저장. method2에서 읽고 필요에 맞추어 clensing하여 사용
    contents_reader.extracted_text_out(tmp_doc_list)
    return return_documents


def read_training_html(dir_route, read_file_num):
    file_list = os.listdir(dir_route)
    random.shuffle(file_list)
    read_file_num = int(len(file_list)*0.85)
    total_text = ''
    for training_data in file_list[read_file_num:]:
        TEST_DATA_LIST.append(dir_route+training_data)
    for file_name in file_list[:read_file_num]:
        with codecs.open(dir_route+file_name,'r', encoding='utf-8') as html_text:
            total_text = total_text + contents_reader.text_only_from_html(html_text.read())
    return total_text


def cal_precision():
    aver_pre=0
    for con in ['adult', 'bitcoin', 'black_market', 'gamble', 'hacking_cyber_attack', 'weapon_hitman', 'legal']:
        print('category : '+con)
        print((TP[con])/(TP[con]+FP[con]))
        aver_pre += (TP[con])/(TP[con]+FP[con])
    return aver_pre/7


def cal_recall():
    aver_recall=0
    for con in ['adult', 'bitcoin', 'black_market', 'gamble', 'hacking_cyber_attack', 'weapon_hitman', 'legal']:
        print('category : '+con)
        print((TP[con])/(TP[con]+FN[con]))
        aver_recall += (TP[con])/(TP[con]+FN[con])
    return aver_recall/7


TEST_DATA_LIST = []
documents = read_text_data_for_training(15)
#model_names = ['logistic', 'naive', 'svm', 'knn', 'decision', 'random']
model_names = ['logistic', 'svm', 'naive']
TP = {'adult':0, 'bitcoin':0, 'black_market':0, 'gamble':0, 'hacking_cyber_attack':0, 'weapon_hitman':0, 'legal':0}
FP = {'adult':0, 'bitcoin':0, 'black_market':0, 'gamble':0, 'hacking_cyber_attack':0, 'weapon_hitman':0, 'legal':0}
#TN = {'adult':0, 'bitcoin':0, 'black_market':0, 'gamble':0, 'hacking_cyber_attack':0, 'weapon_hitman':0, 'legal':0}
FN = {'adult':0, 'bitcoin':0, 'black_market':0, 'gamble':0, 'hacking_cyber_attack':0, 'weapon_hitman':0, 'legal':0}

documents[0] = documents[0].replace('post','').replace('topic','').replace('magnet','').replace('online','').replace('chat','').replace('site','').replace('update','').replace('download','').replace('forum','').replace('community','').replace('support','')
documents[5] = documents[5].replace('account','').replace('mastercard','').replace('register','').replace('paypal','').replace('online','').replace('program','').replace('proxy','').replace('bitcoin','').replace('download','').replace('system','')
documents[6] = documents[6].replace('facebook','').replace('post','').replace('social','').replace('media','').replace('website','').replace('topic','').replace('people','').replace('person','').replace('search','').replace('online','').replace('database','').replace('service','')
documents[9] = documents[9].replace('video','').replace('media','').replace('porn','').replace('girl','').replace('pédo','').replace('pedo','').replace('search','').replace('shop','')
documents[10] = documents[10].replace('video','').replace('media','').replace('porn','').replace('girl','').replace('pédo','').replace('pedo','').replace('service','').replace('information','').replace('admin','').replace('search','').replace('shop','').replace('child','')

for index in range(len(documents)):
    if index == 1:
        documents[index] = contents_reader.text_clensing(documents[index].replace('bitcoins','').replace('bitcoin','').replace('btc',''))
    else:
        documents[index] = contents_reader.text_clensing(documents[index])

#hacking 문서의 특징이 없는지 legal을 해킹으로 분류하고 있음. legal이 스스로를 제대로 분류할 수 있으면 해결 될 듯

for i in range(len(model_names)):
    TP = {'adult':0, 'bitcoin':0, 'black_market':0, 'gamble':0, 'hacking_cyber_attack':0, 'weapon_hitman':0, 'legal':0}
    FP = {'adult':0, 'bitcoin':0, 'black_market':0, 'gamble':0, 'hacking_cyber_attack':0, 'weapon_hitman':0, 'legal':0}
    #TN = {'adult':0, 'bitcoin':0, 'black_market':0, 'gamble':0, 'hacking_cyber_attack':0, 'weapon_hitman':0, 'legal':0}
    FN = {'adult':0, 'bitcoin':0, 'black_market':0, 'gamble':0, 'hacking_cyber_attack':0, 'weapon_hitman':0, 'legal':0}
    model = ccmodel.ContentsClassifierModel(model_names[i], documents)
    for testfile in TEST_DATA_LIST:
        pred_result = model.input_pred(contents_reader.extract_text_from_html( contents_reader.read_html(testfile) ), "single")
        tmp_orig_result = pred_result
        if pred_result in common.BLACK_MARKET_CONTENTS:
            pred_result = 'black_market'
        elif pred_result in common.LEGAL_CONTENTS:
            pred_result = 'legal'
        if pred_result == LABELED_HTML[testfile]:
            TP[LABELED_HTML[testfile]] += 1
        else:
            print(ORIG_HTML[testfile]+' is predicted '+tmp_orig_result)
            FN[LABELED_HTML[testfile]] += 1
            FP[pred_result] += 1
    print(model_names[i])		
    print('Precision')
    print(cal_precision())
    print('Recall')
    print(cal_recall())
    print('\n')


###################################################3

onion_cont_dict = {}
contents_count = {}

for c in common.CONTENTS:
    contents_count[c] = 0

contents_count['unknown'] = 0




for html_f in input_files:
    text_data = contents_reader.read_html_to_text(html_f)
    onion_add = html_f.split('/')[-1].replace('.html','.onion')
    if len(text_data.split(' ')) < 3:
        contents_count['unknown'] += 1
        onion_cont_dict[onion_add] = 'unknown'
    else:
        contents = model.input_pred(text_data,'single')
        contents_count[contents] += 1
        onion_cont_dict[onion_add] = contents




###############################################################

LABELED_HTML = {}
for cont_name in common.CONTENTS:
    cont_val = cont_name
    if cont_val in common.BLACK_MARKET_CONTENTS:
        cont_val = 'black_market'
    elif cont_val in common.LEGAL_CONTENTS:
        cont_val = 'legal'
    tmp = os.listdir(common.DATA_DIR+cont_name)
    for key in tmp:
        LABELED_HTML[common.DATA_DIR+cont_name+'/'+key]=cont_val


ORIG_HTML = {}
for cont_name in common.CONTENTS:
    cont_val = cont_name
    tmp = os.listdir(common.DATA_DIR+cont_name)
    for key in tmp:
        ORIG_HTML[common.DATA_DIR+cont_name+'/'+key]=cont_val


TEST_DATA_LIST = []


#all data use test data. if not pass this for
for dir_name in common.CONTENTS:
	dir_route = common.DATA_DIR + dir_name + '/'
	file_list = os.listdir(dir_route)
	for training_data in file_list:
		TEST_DATA_LIST.append(dir_route+training_data)


contents_reader = ccutils.ContentsClassifierUtils(filtering_word_num = 4)
documents = contents_reader.read_text_data('read text')


contents_reader = ccutils.ContentsClassifierUtils(filtering_word_num = 4)
import codecs
import random

def read_text_data_for_training(read_file_num):
    return_documents = []
    tmp_doc_list = []
    for dir_name in common.CONTENTS:
        dir_route = common.DATA_DIR + dir_name
        tmp_data = read_training_html(dir_route+'/', read_file_num)
        tmp_doc_list.append(tmp_data)
        return_documents.append( contents_reader.text_clensing(tmp_data) )
    # clensing이 되지 않은 원본 텍스트를 저장. method2에서 읽고 필요에 맞추어 clensing하여 사용
    contents_reader.extracted_text_out(tmp_doc_list)
    return return_documents


def read_training_html(dir_route, read_file_num):
    file_list = os.listdir(dir_route)
    random.shuffle(file_list)
    read_file_num = int(len(file_list)*0.80)
    total_text = ''
    for training_data in file_list[read_file_num:]:
        TEST_DATA_LIST.append(dir_route+training_data)
    for file_name in file_list[:read_file_num]:
        with codecs.open(dir_route+file_name,'r', encoding='utf-8') as html_text:
            total_text = total_text + contents_reader.text_only_from_html(html_text.read())
    return total_text


TEST_DATA_LIST = []
documents = read_text_data_for_training(15)
model_names = ['naive']
#model_names = ['logistic', 'naive', 'svm', 'knn', 'decision', 'random']
#model_names = ['decision', 'random']
#model_names = ['logistic', 'svm', 'naive']
#model_names = ['svm']


documents[0] = documents[0].replace('post','').replace('topic','').replace('magnet','').replace('online','').replace('chat','').replace('site','').replace('update','').replace('download','').replace('forum','').replace('community','').replace('support','')
#documents[5] = documents[5].replace('account','').replace('mastercard','').replace('register','').replace('paypal','').replace('online','').replace('program','').replace('proxy','').replace('bitcoin','').replace('download','').replace('system','')
#documents[6] = documents[6].replace('facebook','').replace('post','').replace('social','').replace('media','').replace('website','').replace('topic','').replace('people','').replace('person','').replace('search','').replace('online','').replace('database','').replace('service','')
documents[4] = documents[4].replace('video','').replace('media','').replace('porn','').replace('girl','').replace('pédo','').replace('pedo','').replace('search','').replace('shop','').replace('information','').replace('admin','')

for index in range(len(documents)):
    if index == 1:
        documents[index] = contents_reader.text_clensing(documents[index].replace('bitcoins','').replace('bitcoin','').replace('btc',''))
    else:
        documents[index] = contents_reader.text_clensing(documents[index])


model = ccmodel.ContentsClassifierModel(model_names[0], documents, 1)
test_data_label = []
for testfile in TEST_DATA_LIST:
    test_data_label.append(LABELED_HTML[testfile])


test_data_label_bin = label_binarize(test_data_label, classes=common.CONTENTS)
y_bin = label_binarize(common.CONTENTS, classes=common.CONTENTS)
n_classes = y_bin.shape[1]

X_test = None
for testfile in TEST_DATA_LIST:
    test_data = model.vect.transform([contents_reader.extract_text_from_html( contents_reader.read_html(testfile) ) ]).toarray()
    if X_test is None:
        X_test = test_data
    else:
        X_test = np.append(X_test, test_data,axis=0)


total_fpr = dict()
total_tpr = dict()
total_roc_auc = dict()

for idx in range(len(model_names)):
    model = ccmodel.ContentsClassifierModel(model_names[idx], documents, 1)
    predict_data_label = model.model.predict(X_test)
    print(model_names[idx])
    print(classification_report(test_data_label,predict_data_label))
    print(confusion_matrix(test_data_label,predict_data_label))
    if model_names[idx] in ['naive','knn','decision','random']:
        y_score = model.model.predict_proba(X_test)
    else:
        y_score = model.model.decision_function(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_data_label_bin[:, i], y_score[:, i])
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

tf_recur = [total_fpr]
tt_recur = [total_tpr]
tr_recur = [total_roc_auc]

tf_recur.append(total_fpr)
tt_recur.append(total_tpr)
tr_recur.append(total_roc_auc)


for idx in range(len(model_names)):
    total_fpr[model_names[idx]] = (tf_recur[0][model_names[idx]] + tf_recur[1][model_names[idx]] + tf_recur[2][model_names[idx]] + tf_recur[3][model_names[idx]] + tf_recur[4][model_names[idx]])/5
    total_tpr[model_names[idx]] = (tt_recur[0][model_names[idx]] + tt_recur[1][model_names[idx]] + tt_recur[2][model_names[idx]] + tt_recur[3][model_names[idx]] + tt_recur[4][model_names[idx]])/5
    total_roc_auc[model_names[idx]] = (tr_recur[0][model_names[idx]] + tr_recur[1][model_names[idx]] + tr_recur[2][model_names[idx]] + tr_recur[3][model_names[idx]] + tr_recur[4][model_names[idx]])/5

#model_names_fix = ['Logistic Regression', 'Naive Bayesian', 'SVM', 'KNN', 'Decision tree', 'Random forest']
model_names_fix = ['Naive Bayesian', 'SVM', 'Logistic Regression']

# Plot all ROC curves
plt.figure()
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'orange'])
for i, color in zip(range(len(model_names)), colors):
    plt.plot(total_fpr[model_names[i]], total_tpr[model_names[i]], color=color, lw=2,
             label='ROC curve of '+model_names_fix[i]+' (area = {0:0.2f})'.format(total_roc_auc[model_names[i]]) )

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()




    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()




from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [
    {'clf__C': param_range, 'clf__kernel': ['linear']},
    {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
gs = gs.fit(X_test, test_data_label)

print(gs.best_score_)
print(gs.best_params_)


pipe_logi = Pipeline([('scl', StandardScaler()), ('clf', LogisticRegression())])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_iter_range = [1, 10, 50]
param_grid = [
    {'clf__C': param_range, 'clf__solver': ['liblinear'], 'clf__multi_class': ['ovr']},
    {'clf__C': param_range, 'clf__solver': ['saga', 'lbfgs'], 'clf__max_iter': param_iter_range, 'clf__multi_class': ['multinomial']}]

gs = GridSearchCV(estimator=pipe_logi, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
gs = gs.fit(X_test, test_data_label)

print(gs.best_score_)
print(gs.best_params_)


pipe_decision = Pipeline([('scl', StandardScaler()), ('clf', DecisionTreeClassifier())])

param_range = [None, 5, 10, 20, 30]
split = [0.0001, 0.001, 0.01, 0.1, 1]
leaf = [0.0001, 0.001, 0.01, 0.5]
param_splitter = ['best', 'random']
param_grid = [
    {'clf__splitter': param_splitter, 'clf__max_depth': param_range, 'clf__max_features':param_range , 'clf__max_leaf_nodes':param_range }]

gs = GridSearchCV(estimator=pipe_decision, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
gs = gs.fit(X_test, test_data_label)

print(gs.best_score_)
print(gs.best_params_)


pipe_decision = Pipeline([('scl', StandardScaler()), ('clf', RandomForestClassifier(random_state=1))])

n_estimators = [1, 5, 10, 20, 30]
max_depth = [None, 5, 10, 20, 30]
param_range = [None, 5, 10, 20, 30]
param_grid = [
    {'clf__n_estimators': n_estimators, 'clf__max_depth': max_depth, 'clf__max_features':param_range , 'clf__max_leaf_nodes':param_range }]

gs = GridSearchCV(estimator=pipe_decision, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
gs = gs.fit(X_test, test_data_label)

print(gs.best_score_)
print(gs.best_params_)


logi = LogisticRegression(panalty='l2', solver = 'lbfgs', multi_class = 'multinomial', C = 0.01)
logi.fit(model.X, model.Y)
predict_data_label = logi.predict(X_test)
print(classification_report(test_data_label,predict_data_label))

'''
