import os
import shutil
import contents_classifier_common as common
import contents_classifier_utils as ccutils
import contents_classifier_model as ccmodel


# 테스트용 메소드
# 정확도 측정을 위한 카운트는 이 함수에서 카운트해야함
# 모델의 결과값은 예측한 컨텐츠결과만을 반환하므로 정답과 대조하여 어떤 유형의 정답/오답인지 카운트해야함
def testing_method(contents_name, mode):
    pred_accurate = 0
    if mode == 'all':
        inputfile_route_list=os.listdir(common.UNLABELED_DIR)
        for idx in range(len(inputfile_route_list)):
            inputfile_route_list[idx] = common.UNLABELED_DIR + inputfile_route_list[idx]
        for inputfile_route in inputfile_route_list:
            #exception page check(default page, directory page)
            #read_html_to_text -> find unknown page
            hs_text_data = contents_reader.extract_text_from_html( contents_reader.read_html(inputfile_route) )
            if len(hs_text_data) > 10:
                pred_result = model.input_pred(hs_text_data, mode)
            else:
                pred_result = 'unknown'
            shutil.copy(inputfile_route, common.AUTO_LABELING_DIR + pred_result)
    else :
        route = common.DATA_DIR+contents_name+'/'
        inputfile_route_list=os.listdir(route)
        for idx in range(len(inputfile_route_list)):
            inputfile_route_list[idx] = route + inputfile_route_list[idx]
        for inputfile_route in inputfile_route_list:
            hs_text_data = contents_reader.extract_text_from_html( contents_reader.read_html(inputfile_route) )
            pred_result = model.input_pred(hs_text_data, mode)
            if pred_result == contents_name:
                pred_accurate += 1
            elif pred_result in common.LEGAL_CONTENTS and contents_name in common.LEGAL_CONTENTS:
                pred_accurate += 1
            elif pred_result in common.BLACK_MARKET_CONTENTS and contents_name in common.BLACK_MARKET_CONTENTS:
                pred_accurate += 1
        print(contents_name)
        print(pred_accurate)
        print(len(inputfile_route_list))
        print(pred_accurate/len(inputfile_route_list))


#test all labeled html
def test_one_category():
    for cont_name in common.CONTENTS:
        testing_method(cont_name, 'single')


#test all unlabeled html
def test_all_category():
    for cont_name in common.CONTENTS:
        os.mkdir(common.AUTO_LABELING_DIR+cont_name)
    os.mkdir(common.AUTO_LABELING_DIR+'unknown')
    testing_method(common.CONTENTS, 'all')


# 모델 생성 및 예측을 위한 텍스트 리더
contents_reader = ccutils.ContentsClassifierUtils(filtering_word_num = 4)
#documents = contents_reader.read_text_data('make text')
documents = contents_reader.read_text_data('read text')


# 모델의 생성
model_names = ['logistic', 'naive', 'decision', 'svm', 'random']
model = ccmodel.ContentsClassifierModel(model_names[3], documents)
#model.reselect_model(model_names[1])


#test_one_category()
test_all_category()

