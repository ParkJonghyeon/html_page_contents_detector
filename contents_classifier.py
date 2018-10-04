import os, string, numpy
import contents_classifier_common as common
import contents_classifier_utils as ccutils
import contents_classifier_model as ccmodel


# 테스트용 메소드
def testing_method(contents_name, mode):
    pred_accurate = 0
    if mode == 'all':
        route = '/media/lark/extra_storage/onion_link_set/html_171001_to_180327/unlabeled/'
        inputfile_route_list=os.listdir(route)
        for idx in range(len(inputfile_route_list)):
            inputfile_route_list[idx] = route + inputfile_route_list[idx]
        for inputfile_route in inputfile_route_list:
            pred_result = model.input_pred(contents_reader.read_html_to_pred_text(inputfile_route), mode)
    else :
        route = common.DATA_DIR+contents_name+'/'
        inputfile_route_list=os.listdir(route)
        for idx in range(len(inputfile_route_list)):
            inputfile_route_list[idx] = route + inputfile_route_list[idx]
        for inputfile_route in inputfile_route_list:
            pred_result = model.input_pred(contents_reader.read_html_to_pred_text(inputfile_route), mode)
            if pred_result == contents_name:
                pred_accurate += 1
        print(contents_name)
        print(pred_accurate)
        print(pred_accurate/len(inputfile_route_list))


# 예측 모델 생성
contents_reader = ccutils.ContentsClassifierUtils(filtering_word_num = 5)

#documents = contents_reader.read_text_data(1)
documents = contents_reader.read_text_data(2)


USE_MODEL = 'logistic'
#USE_MODEL = 'naive'
#USE_MODEL = 'decision'
#USE_MODEL = 'svm'
#USE_MODEL = 'random'
KNN_NEIGHBOR = 3
model = ccmodel.ContentsClassifierModel(USE_MODEL, documents)


#test all labeled html
for cont_name in common.CONTENTS:
    testing_method(cont_name, 'single')


#test all unlabeled html
for cont_name in common.CONTENTS:
    os.mkdir(common.AUTO_LABELING_DIR+cont_name)

testing_method(common.CONTENTS, 'all')


LABELED_HTML = {}

for cont_name in common.CONTENTS:
    tmp = os.listdir(cont_name)
    for key in tmp:
        LABELED_HTML[common.DATA_DIR+cont_name+'/'+key]=cont_name

TRAINING_DATA_LIST = []


'''def read_text_data_for_training(read_file_num):
    return_documents = []
    tmp_doc_list = []
    for dir_name in common.CONTENTS:
        dir_route = common.DATA_DIR + dir_name
        tmp_data = read_training_html(dir_route+'/', read_file_num)
        tmp_doc_list.append(tmp_data)
        return_documents.append( text_clensing(tmp_data, FILTERING_WORD_NUM) )
    # clensing이 되지 않은 원본 텍스트를 저장. method2에서 읽고 필요에 맞추어 clensing하여 사용
    contents_reader.extracted_text_out(tmp_doc_list)
    return return_documents


def read_training_html(dir_route, read_file_num):
    file_list = os.listdir(dir_route)
    total_text = ''
    for training_data in file_list[read_file_num:]:
        TRAINING_DATA_LIST.append(training_data)
    for file_name in file_list[:read_file_num]:
        with codecs.open(dir_route+file_name,'r', encoding='utf-8') as html_text:
            total_text = total_text + text_only_from_html(html_text.read())
    return total_text


documents = read_text_data_for_training(15)

vect = TfidfVectorizer(token_pattern=r'\w+', lowercase=True, stop_words=common.ALL_STOP_WORD)
X = vect.fit_transform(documents)
# X.todense()
# sentences에는 각 서비스 분류의 문서들이 포함
# 하나의 카테고리하에 3-4종류의 html 문서가 하나로 통합 된 것 한개씩?
Y = common.CONTENTS

USE_MODEL = 'logistic'
#USE_MODEL = 'naive'
#USE_MODEL = 'decision'
#USE_MODEL = 'svm'
#USE_MODEL = 'random'
KNN_NEIGHBOR = 3
model = make_model(X,Y)
'''
