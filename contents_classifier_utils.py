import codecs, os, nltk, re
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
import contents_classifier_common as common


class ContentsClassifierUtils:
    def __init__(self, filtering_word_num=common.FILTERING_WORD_NUM):
        self.filtering_word_num = filtering_word_num
        self.tokenizer = RegexpTokenizer(r'\w+')


    # html에서 추출한 텍스트의 언어 식별 후, 적당한 stop word 적용 후 토큰화하여 반환
    def tokenize_and_stopword(self, text):
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)
        text_tokens = [i for i in tokens if not i in common.ALL_STOP_WORD]
        text = " ".join(t.strip() for t in text_tokens)
        return text


    # html에서 텍스트 추출하고 토큰화 함수 결과 값 반환
    def text_only_from_html(self, html_text):
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
        return self.tokenize_and_stopword(texts)


# 주어진 파일 경로의 html을 읽어 토큰들의 문자열로 반환
    def read_html_to_text(self, target_file_route):
        with codecs.open(target_file_route, 'r', encoding='utf-8') as target_file:
            return_text = self.text_clensing( self.text_only_from_html(target_file.read()) )
        return return_text


    # 주어진 파일 경로들의 html을 읽어 토큰들의 문자열로 반환
    def read_all_html(self, target_files_route_list):
        total_text = ''
        for target_file in target_files_route_list:
            total_text = total_text + self.read_html_to_text(target_file)
        return total_text


    # text clensing 과정에서 특수문자 제거
    def replacing(self, target, replace_word):
        return target.replace(replace_word, ' ')


    # text clensing하여 통합 된 모든 html 텍스트를 정리
    def text_clensing(self, text):
        # word에서 숫자값 제거
        pattern_only_num = re.compile('[0-9]+')
        pattern_complex_num_1 = re.compile('[0-9]+[A-Za-z]+')
        pattern_complex_num_2 = re.compile('[A-Za-z]+[0-9]+')
        # word에서 bitcoin 주소 패턴 제거
        pattern_bitcoin = re.compile("[A-Za-z0-9]{30,80}")
        for replace_word in common.REPLACE_WORD:
            text = self.replacing(text, replace_word)
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
            if len(w) < self.filtering_word_num:
                continue
            #나머지 단어들만을 return_word에 append
            return_word.append(w)
        return_word = ' '.join(return_word)
        return return_word


    # clensing까지 완료 된 텍스트들의 output을 생성
    def extracted_text_out(self, extract_target_documnets):
        for doc_index in range(len(extract_target_documnets)):
            with codecs.open(common.TEXT_DATA_DIR + common.CONTENTS[doc_index] + common.OUTPUT_TEXT_NAME,'w', encoding='utf-8') as extract_data:
                extract_data.write(extract_target_documnets[doc_index])


    # 모델에 사용할 documents 리스트 작성
    def read_text_data(self, method):
        return_documents = []
        # method1 dir의 모든 html을 읽어들여서 documents 리스트 생성하고 모델링 제작. 학습용 데이터의 초기 생성시 수행
        if method == 'make text':
            tmp_doc_list = []
            for dir_name in common.CONTENTS:
                dir_route = os.listdir(common.DATA_DIR + dir_name + '/')
                tmp_data = self.read_all_html(target_files_route_list)
                tmp_doc_list.append(tmp_data)
                return_documents.append( self.text_clensing(tmp_data) )
            # clensing이 되지 않은 원본 텍스트를 저장. method2에서 읽고 필요에 맞추어 clensing하여 사용
            self.extracted_text_out(tmp_doc_list)
            return return_documents
        # method2 기존에 읽었던 html의 텍스트들을 파일로 만들어 읽어들인 후 바로 모델링으로 제작
        elif method == 'read text':
            for content_name in common.CONTENTS:
                text_route = common.TEXT_DATA_DIR+content_name+'_extract_data.txt'
                with codecs.open(text_route,'r',encoding='utf-8') as data:
                    return_documents.append( self.text_clensing(data.read()) )
            return return_documents


    
