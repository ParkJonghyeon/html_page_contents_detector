from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
import re
from bs4 import BeautifulSoup
import nltk, string, numpy
import os
from polyglot.detect import Detector
from nltk.tokenize import RegexpTokenizer
import codecs
import shutil


FILTERING_WORD_NUM = 5
tokenizer = RegexpTokenizer(r'\w+')

# html에서 title과 text의 언어 식별 후, 언어에 해당하는 stop word 적용
# 텍스트와 타이틀을 통합한 하나의 문자열을 반환
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


# html에서 텍스트 추출을 위한 함수
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
    # img 태그의 alt 텍스트를 통해 이미지 콘텐츠 유추 시도
    imgs = soup.findAll("img")
    for img in imgs:
        if 'alt' in img.attrs.keys():
            if len(img['alt']) > 1:
                texts += ' ' + img['alt']
    return tokenize_and_stopword(texts)


# text clensing
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


# text clensing_특수문자 제거
def replacing(target, replace_word):
    return target.replace(replace_word, ' ')


# 각 디렉토리의 html을 읽어서 하나의 문자열로 모든 텍스트를 통합
def read_all_html(dir_route):
    file_list = os.listdir(dir_route)
    total_text = ''
    for file_name in file_list:
#    for file_idx in range(0,7):
        with codecs.open(dir_route+file_name,'r', encoding='utf-8') as html_text:
#        with codecs.open(dir_route+file_list[file_idx],'r', encoding='utf-8') as html_text:
            total_text = total_text + text_only_from_html(html_text.read())
    return total_text


# 각 디렉토리의 html을 읽어서 하나의 문자열로 모든 텍스트를 통합
def extracted_text_out(documnets):
    for doc_index in range(len(dir_list)):
        with codecs.open(data_dir+'0_Training/'+dir_list[doc_index]+'_extract_data.txt','w', encoding='utf-8') as extract_data:
            extract_data.write(documents[doc_index])


# 모델링에 사용할 documents 리스트 작성
def read_text_data(dir_list, method):
    #return list
    return_documents = []
    # method1 dir의 모든 html을 읽어들여서 documents 리스트 생성하고 모델링 제작
    if method == 1:
        for dir_name in dir_list:
            dir_route = data_dir + dir_name
            return_documents.append( text_clensing(read_all_html(dir_route+'/'), FILTERING_WORD_NUM) )
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


# 예측 함수. 예측하려는 문서파일의 경로를 입력받아 결과를 출력.
# shutil로 문서의 복사본을 지정위치로 이동할 수 있음
def input_pred(inputfile_route):
    with codecs.open(inputfile_route, 'r', encoding='utf-8') as input_file:
        pred_text = text_only_from_html(input_file.read())
        pred_text = text_clensing(pred_text, FILTERING_WORD_NUM)
    # vect에서 토큰화, stop word, lower 등 적용되고 있음
    X_pred = vect.transform([pred_text])
    y_pred = model.predict(X_pred)
    #shutil.copy(inputfile_route, '/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/0_Test/auto_labeling/'+y_pred[0])
    return y_pred[0]


# 테스트용 메소드
def testing_method(contents_name):
    pred_accurate = 0
    route='/media/lark/extra_storage/onion_link_set/html_171001_to_180327/unlabeled/'
    route='/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/'+contents_name+'/'
#    route='/media/lark/extra_storage/onion_link_set/html_171001_to_180327/adult_tmp/'
    inputfile_route_list=os.listdir(route)
    for idx in range(len(inputfile_route_list)):
        inputfile_route_list[idx] = route + inputfile_route_list[idx]
    for inputfile_route in inputfile_route_list:
        pred_result = input_pred(inputfile_route)
        if pred_result == contents_name:
            pred_accurate += 1
    print(contents_name)
    print(pred_accurate)
    print(pred_accurate/len(inputfile_route_list))


data_dir = '/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/'
dir_list = os.listdir(data_dir)
dir_list.sort()
dir_list.remove('0_Test')
dir_list.remove('0_Training')
dir_list.remove('0_detail')
dir_list.remove('unknown')
dir_list.remove('tfidf_lr.py')
dir_list.remove('tf-idf_wc.py')
dir_list.remove('labeling_rule.txt')
dir_list.remove('memo.txt')
dir_list.remove('auto_labeling.txt')
dir_list.remove('auto_labeling_list.txt')
dir_list.remove('hs_freqency.py')
dir_list.remove('README.md')
dir_list.remove('.git')
#dir_list.remove('legal')


#documents = read_text_data(dir_list, 1)
#extracted_text_out(documents)
documents = read_text_data(dir_list, 2)
#documents.remove(documents[7])


# 예측 모델 생성
#en_stop = get_stop_words('en')
all_stop = ['co', 'com', 'org', 'www', 'net', 'onion', 'php', 'html', 'txt', 'png', 'jpg', 'gif', 'onionadd', 'btcadd', 'ipadd']
all_stop += get_stop_words('en') + get_stop_words('french') + get_stop_words('german') + get_stop_words('italian') + get_stop_words('spanish') + get_stop_words('russian') + get_stop_words('arabic')

vect = TfidfVectorizer(token_pattern=r'\w+', lowercase=True, stop_words=all_stop)
X = vect.fit_transform(documents)
# X.todense()
# sentences에는 각 서비스 분류의 문서들이 포함
# 하나의 카테고리하에 3-4종류의 html 문서가 하나로 통합 된 것 한개씩?
Y = dir_list


# LR과 tfidf 예측 모델
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, Y)


for dir_name in dir_list:
    os.mkdir('/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/0_Test/auto_labeling/'+dir_name)


#test all
for dir_name in dir_list:
    testing_method(dir_name)


#test one
testing_method(dir_list[0])


'''
# NB와 tfidf 예측 모델
from sklearn.naive_bayes import GaussianNB
modelNB = GaussianNB()
modelNB.fit(X.toarray(),Y)


def input_pred_NB(inputfile_route):
    with codecs.open(inputfile_route, 'r', encoding='utf-8') as input_file:
        pred_text = text_only_from_html(input_file.read())
    # vect에서 토큰화, stop word, lower 등 적용되고 있음
    X_pred = vect.transform([pred_text]).toarray()
    y_pred = model.predict(X_pred)
    print(y_pred)


input_pred('/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/black_market/norabizt5qqxefxy.html')
input_pred_NB('/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/black_market/norabizt5qqxefxy.html')


# hashing과 LR 모델
# Hashing == Count vectorizer
from sklearn.feature_extraction.text import HashingVectorizer
vect = HashingVectorizer(n_features=200)
X = vectorizer.transform(documents)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, dir_list)

input_pred('/media/lark/extra_storage/onion_link_set/html_171001_to_180327/training_html/black_market/norabizt5qqxefxy.html')
'''
