import os, re, codecs, shutil, nltk, string, numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from bs4 import BeautifulSoup
from polyglot.detect import Detector
from nltk.tokenize import RegexpTokenizer


FILTERING_WORD_NUM = 5
tokenizer = RegexpTokenizer(r'\w+')

DATA_DIR = '/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/'
DIR_LIST = ['adult', 'bitcoin', 'black_market', 'counterfeit', 'drug', 
'gamble', 'hacking_cyber_attack', 'legal', 'weapon_hitman']


# html에서 추출한 텍스트의 언어 식별 후, 적당한 stop word 적용 후 토큰화하여 반환
def tokenize_and_stopword(text):
    try:
        text = text.lower()
        detector = Detector(text, quiet=True)
        stop_list = get_stop_words(detector.language.code)
        tokens = tokenizer.tokenize(text)
        text_tokens = [i for i in tokens if not i in stop_list]
        text = " ".join(t.strip() for t in text_tokens)
    except:
        stop_list = get_stop_words('en')
        tokens = tokenizer.tokenize(text)
        text_tokens = [i for i in tokens if not i in stop_list]
    return text


# html에서 텍스트 추출하고 토큰화 함수 결과 값 반환
def text_only_from_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    # html 내부의 visible 텍스트를 모두 추출
    for script in soup(["script", "style"]):
        script.extract()
    texts = soup.get_text()
    # meta 데이터에서 콘텐츠 판별에 사용 가능한 데이터를 추출
    metas = soup.findAll("meta")
    for meta in metas:
        if meta.get("name") in ["description", "keywords"]:
            try:
                meta_data = meta.get("content")
                texts += ' ' + meta_data
            except TypeError:
                meta_data = meta.get("value")
                texts += ' ' + meta_data
    # img 태그의 alt 텍스트를 통해 이미지 콘텐츠 유추
    imgs = soup.findAll("img")
    for img in imgs:
        if 'alt' in img.attrs.keys():
            if len(img['alt']) > 1:
                texts += ' ' + img['alt']
    return tokenize_and_stopword(texts)


# 각 디렉토리의 html 파일을 읽어서 토큰들의 문자열로 컨텐츠의 모든 html 텍스트를 통합
def read_all_html(dir_route):
    file_list = os.listdir(dir_route)
    total_text = ''
    for file_name in file_list:
        with codecs.open(dir_route+file_name,'r', encoding='utf-8') as html_text:
            total_text = total_text + text_only_from_html(html_text.read())
    return total_text


# text clensing 과정에서 특수문자 제거
def replacing(target, replace_word):
    return target.replace(replace_word, ' ')


# text clensing하여 통합 된 모든 html 텍스트를 정리
def text_clensing(text, filtering_word_num):
    # word에서 숫자값 제거
    pattern_only_num = re.compile('[0-9]+')
    pattern_complex_num_1 = re.compile('[0-9]+[A-Za-z]+')
    pattern_complex_num_2 = re.compile('[A-Za-z]+[0-9]+')
    # word에서 bitcoin 주소 패턴 제거
    pattern_bitcoin = re.compile("[A-Za-z0-9]{30,80}")
    replace_word_list = ['(',')','[',']','{','}','/','|','<','>',':',',','=','_','-','+','*','!','?','\"','\'','\n','\t']
    for replace_word in replace_word_list:
        text = replacing(text, replace_word)
    orig_word = text.split(' ')
    return_word = []
    for w in orig_word:
        #number only 단어면 pass
        if pattern_only_num.match(w) is not None:
            continue
        #bitcoin 단어면 pass
        if pattern_bitcoin.match(w) is not None:
            continue
        #word+숫자, 숫자+word 패턴이면 숫자 부분을 삭제 후 체크
        if pattern_complex_num_1.match(w) is not None or pattern_complex_num_2.match(w) is not None:
            w = re.sub('[0-9]+','',w)
        if 'login' in w or 'logins' in w:
            continue
        #single character거나 space면 pass
        if len(w) < filtering_word_num:
            continue
        #나머지 단어들만을 return_word에 append
        return_word.append(w)
    return_word = ' '.join(return_word)
    return return_word


# clensing까지 완료 된 텍스트들의 output을 생성
def extracted_text_out(extract_target_documnets):
    for doc_index in range(len(extract_target_documnets)):
        with codecs.open(DATA_DIR+'0_Training/'+DIR_LIST[doc_index]+'_extract_data.txt','w', encoding='utf-8') as extract_data:
            extract_data.write(extract_target_documnets[doc_index])


# 모델에 사용할 documents 리스트 작성
def read_text_data(method):
    return_documents = []
    # method1 dir의 모든 html을 읽어들여서 documents 리스트 생성하고 모델링 제작. 학습용 데이터의 초기 생성시 수행
    if method == 1:
        tmp_doc_list = []
        for dir_name in DIR_LIST:
            dir_route = DATA_DIR + dir_name
            tmp_data = read_all_html(dir_route+'/')
            tmp_doc_list.append(tmp_data)
            return_documents.append( text_clensing(tmp_data, FILTERING_WORD_NUM) )
        # clensing이 되지 않은 원본 텍스트를 저장. method2에서 읽고 필요에 맞추어 clensing하여 사용
        extracted_text_out(tmp_doc_list)
        return return_documents
    # method2 기존에 읽었던 html의 텍스트들을 파일로 만들어 읽어들인 후 바로 모델링으로 제작
    elif method == 2:
        extracted_text_dir = '/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/0_Training/'
        text_list = os.listdir(extracted_text_dir)
        text_list.sort()
        for text_name in text_list:
            text_route = extracted_text_dir+text_name
            with codecs.open(text_route,'r',encoding='utf-8') as data:
                return_documents.append(text_clensing(data.read(), FILTERING_WORD_NUM))
        return return_documents


# model list
# LogisticRegression / Naive_bayes / 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# model 생성 함수. 필요에 따라 import 후 해당 모델 생성 분기 추가
def make_model(x_input, y_input):
    if USE_MODEL == 'logistic':
        return_model = LogisticRegression()
        return_model.fit(x_input, y_input)
    elif USE_MODEL == 'naive':
        return_model = GaussianNB()
        return_model.fit(x_input.toarray(), y_input)
    elif USE_MODEL == 'decision':
        return_model = DecisionTreeClassifier()
        return_model.fit(x_input, y_input)
    elif USE_MODEL == 'knn':
        return_model = KNeighborsClassifier(n_neighbors=KNN_NEIGHBOR)
        return_model.fit(x_input, y_input)
    elif USE_MODEL == 'svm':
        return_model = SVC(gamma='auto')
        return_model.fit(x_input, y_input)
    return return_model


# 예측 함수. 예측하려는 문서파일의 경로를 입력받아 결과를 출력.
# shutil로 문서의 복사본을 지정위치로 이동할 수 있음
def input_pred(inputfile_route, mode):
    with codecs.open(inputfile_route, 'r', encoding='utf-8') as input_file:
        pred_text = text_only_from_html(input_file.read())
        pred_text = text_clensing(pred_text, FILTERING_WORD_NUM)
    # vect에서 토큰화, stop word, lower 등 적용되고 있음
    X_pred = vect.transform([pred_text])
    if USE_MODEL == 'naive':
        y_pred = model.predict(X_pred.toarray())
    else:
        y_pred = model.predict(X_pred)
    if mode == 'all':
        shutil.copy(inputfile_route, '/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/0_Test/auto_labeling/'+y_pred[0])
    return y_pred[0]


# 테스트용 메소드
def testing_method(contents_name, mode):
    pred_accurate = 0
    if mode == 'all':
        route='/media/lark/extra_storage/onion_link_set/html_171001_to_180327/unlabeled/'
        inputfile_route_list=os.listdir(route)
        for idx in range(len(inputfile_route_list)):
            inputfile_route_list[idx] = route + inputfile_route_list[idx]
        for inputfile_route in inputfile_route_list:
            pred_result = input_pred(inputfile_route, mode)
    else :
        route='/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/'+contents_name+'/'
        inputfile_route_list=os.listdir(route)
        for idx in range(len(inputfile_route_list)):
            inputfile_route_list[idx] = route + inputfile_route_list[idx]
        for inputfile_route in inputfile_route_list:
            pred_result = input_pred(inputfile_route, mode)
            if pred_result == contents_name:
                pred_accurate += 1
        print(contents_name)
        print(pred_accurate)
        print(pred_accurate/len(inputfile_route_list))


#documents = read_text_data(1)
documents = read_text_data(2)

# 예측 모델 생성
all_stop = ['co', 'com', 'org', 'www', 'net', 'onion', 'php', 'html', 'txt', 'png', 'jpg', 'gif', 'onionadd', 'btcadd', 'ipadd']
all_stop += get_stop_words('en') + get_stop_words('french') + get_stop_words('german') + get_stop_words('italian') + get_stop_words('spanish') + get_stop_words('russian') + get_stop_words('arabic')

vect = TfidfVectorizer(token_pattern=r'\w+', lowercase=True, stop_words=all_stop)
X = vect.fit_transform(documents)
# X.todense()
# sentences에는 각 서비스 분류의 문서들이 포함
# 하나의 카테고리하에 3-4종류의 html 문서가 하나로 통합 된 것 한개씩?
Y = DIR_LIST

USE_MODEL = 'logistic'
#USE_MODEL = 'naive'
#USE_MODEL = 'decision'
#USE_MODEL = 'svm'
KNN_NEIGHBOR = 3
model = make_model(X,Y)


#test all labeled html
for dir_name in DIR_LIST:
    testing_method(dir_name, 'single')


#test all unlabeled html
for dir_name in DIR_LIST:
    os.mkdir('/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/0_Test/auto_labeling/'+dir_name)

testing_method(dir_name, 'all')
