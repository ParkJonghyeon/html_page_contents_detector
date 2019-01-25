from os.path import join
from stop_words import get_stop_words


# directory/file name
DATA_DIR = 'data_set/contents/'
TEXT_DATA_DIR = 'data_set/all_contents_text/'
AUTO_LABELING_DIR = 'data_set/auto_labeling_test/'
UNLABELED_DIR = 'data_set/unlabeled/'
OUTPUT_TEXT_NAME = '_extract_data.txt'


# contents name list
#CONTENTS = ['adult', 'bitcoin', 'black_market', 'counterfeit', 'drug', 
#'gamble', 'hacking_cyber_attack', 'weapon_hitman', 'software_&_file_share', 'blog_&_personal_page', 'forum_&_chat_&_mail', 'cloud_server_&_hosting', 'white_market']
CONTENTS = ['adult', 'bitcoin', 'black_market', 'gamble', 'legal']
LEGAL_CONTENTS = ['software_&_file_share', 'blog_&_personal_page', 'forum_&_chat_&_mail', 'cloud_server_&_hosting', 'white_market']
ILLEGAL_CONTENTS = list(set(CONTENTS) - set(LEGAL_CONTENTS))
BLACK_MARKET_CONTENTS = ['black_market', 'counterfeit', 'drug', 'hacking_cyber_attack', 'weapon_hitman']


# replace word list
REPLACE_WORD = ['(',')','[',']','{','}','/','|','<','>',':',',','=',
    '_','-','+','*','!','?','\"','\'','\n','\t']


# stop word list
DEFAULT_STOP_WORD = ['co', 'com', 'org', 'www', 'net', 'onion', 
    'php', 'html', 'txt', 'png', 'jpg', 'gif', 'pdf', 'username', 'password', 'id', 'pw', 'login', 'nickname']
LANGUAGES = ['繁體中文', '中文', 'deutsch', 'čeština', 'ελληνικά', 
    'english', 'español', 'français', '日本語', 'italiano', 'magyar', 
    'nederlands', 'norsk', 'فارسی', 'العربية', 'polski', 'português', 
    'română', 'pусский', 'slovenski', 'shqip', 'svenska', 'türkçe']
ALL_STOP_WORD = DEFAULT_STOP_WORD + LANGUAGES + get_stop_words('en') + get_stop_words('french') + get_stop_words('german') + get_stop_words('italian') + get_stop_words('spanish') + get_stop_words('russian') + get_stop_words('arabic')


# filtering word number
FILTERING_WORD_NUM = 5
KNN_NEIGHBOR = 3
