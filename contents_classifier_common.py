from os.path import join
from stop_words import get_stop_words


# directory/file name
DATA_DIR = 'data_set/contents/'
TEXT_DATA_DIR = 'data_set/all_contents_text/'
AUTO_LABELING_DIR = 'data_set/auto_labeling_test/'
OUTPUT_TEXT_NAME = '_extract_data.txt'


# contents name list
CONTENTS = ['adult', 'bitcoin', 'black_market', 'counterfeit', 'drug', 
    'gamble', 'hacking_cyber_attack', 'legal', 'weapon_hitman']


# replace word list
REPLACE_WORD = ['(',')','[',']','{','}','/','|','<','>',':',',','=',
    '_','-','+','*','!','?','\"','\'','\n','\t']


# stop word list
DEFAULT_STOP_WORD = ['co', 'com', 'org', 'www', 'net', 'onion', 
    'php', 'html', 'txt', 'png', 'jpg', 'gif']
LANGUAGES = ['繁體中文', '中文', 'deutsch', 'čeština', 'ελληνικά', 
    'english', 'español', 'français', '日本語', 'italiano', 'magyar', 
    'nederlands', 'norsk', 'فارسی', 'العربية', 'polski', 'português', 
    'română', 'pусский', 'slovenski', 'shqip', 'svenska', 'türkçe']
ALL_STOP_WORD = DEFAULT_STOP_WORD + LANGUAGES + get_stop_words('en') 
    + get_stop_words('french') + get_stop_words('german') 
    + get_stop_words('italian') + get_stop_words('spanish') 
    + get_stop_words('russian') + get_stop_words('arabic')


# filtering word number
FILTERING_WORD_NUM = 5
