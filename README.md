# html_page_contents_detector

### 개요
  Tor 크롤러로 수집한 페이지들을 classification하는 코드  
  clustering이 아닌 classification이므로 사전에 라벨링 된 학습용 html을 제공해야함  
  머신러닝 모델은 LogisticRegression, MultinomialNB, SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier 을 사용  
  학습 데이터인 html들을 라벨 별로 tf-idf 기반 word vector 세트로 만들어 학습하고 classification 하려는 데이터를 word vector로 만들어 가장 유사한 카테고리로 분류  
  머신러닝을 통한 html 콘텐츠 분류의 자세한 내용은 석사 졸업 논문 참조  
  https://docs.google.com/presentation/d/1yYs08VKlBGpsPPb4OvV25-RPeHdm38ZsRewhaB_hU8M/edit?usp=sharing

### ROC-AUC, F1 스코어 정리 자료  
  해당 PPT에 scikit-learn을 사용한 ROC-AUC 및 F1 스코어 측정 관련 자료 정리되어있으므로 참조  
  https://docs.google.com/presentation/d/1V1yvlNMObT7MsF0BNGt54-alxSMkcz8-DQkjZBTBIOM/edit?usp=sharing

### Category 종류
* adult - 불법 포르노 동영상/이미지
* bitcoin - bitcoin 거래소, 환전소, 세탁
* black_market - 마약 거래, 해킹 및 해킹 툴, 복제 상품 및 카드, 위조 서류, 살인청부 등 불법적인 거래 전반
    * hacking_cyber_attack - 해킹, 계정 훔치기, 사이버 공격(DDOS, 랜섬웨어 등)
    * weapon_hitman - 무기 판매, 살인청부, 납치의뢰, 테러모의 등
    * counterfeit - 복제 상품 및 카드, 위조품, 위조 서류/공문서
    * drug - 약물만 판매/약 제조 레시피 등 
* gamble - 사설 도박, 온라인 도박
* regal - 위에 해당하지 않는 서비스(개인 블로그, 디렉토리, 일반 서비스, 인권 신장등을 위한 내부 고발 사이트)
    * commercial - 결제서비스
    * default_apache/nginx - 아파치/nginx 서버 기본 페이지
    * white_market - 선불카드, 일반물품 판매
    * book - 도서,ebook
    * software_&_file_share - 소프트웨어, 파일공유, 저장소
    * cloud_server_&_hosting - 클라우드 서버, ISP, 호스팅, 그룹웨어
    * blog_&_personal_page - 블로그 및 개인 페이지
    * hs_directory - HS 디렉토리, 위키
    * forum_&_chat_&_mail - 포럼, 채팅서버, 메일 서비스, 뉴스
* unkown - 특징을 추출할 텍스트가 부족하거나 이미지만 존재하는 페이지
* hs_directory - 다른 hidden service의 onion 주소를 모아놓은 directory 페이지. 한페이지에서 등장하는 서로다른 onion 주소 수를 카운트해서 별도 분류함  

regal과 black_market의 경우 세부분류가 필요할 수 있지만, 현재 성능 상 7개 카테고리로 분류하는 것이 가장 의미있는 성능을 보임
