'''
A pytorch chatbot
'''

import os
from preprocessing import *


#Load the two dataset
SOS_token = 2
EOS_token = 3
UNK_token = 1
MAX_LENGTH = 50 #We will abandon any sentence that is longer than this length

#Load the Dataset
data_path = '/home/zmykevin/machine_translation_vision/dataset/NMT_WMT_2017_V2'

#Load the train, val and test dataset
train_source = load_data(os.path.join(data_path,'train_WMT_2017.en'))
val_source = load_data(os.path.join(data_path,'val_WMT_2017.en'))
test_source = load_data(os.path.join(data_path,'test_WMT_2017.en'))

print("train_data_set_size: {}".format(len(train_source)))
#Create a fake dataset from this three soruces
