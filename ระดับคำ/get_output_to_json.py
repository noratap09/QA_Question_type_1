from keras.models import Model, Sequential,load_model
from keras.layers import  Input, Dense,Conv1D, MaxPooling1D ,GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math

from pythainlp.tag.named_entity import ThaiNameTagger
from pythainlp.tag import pos_tag, pos_tag_sents
ner = ThaiNameTagger()

import json

from elasticsearch import Elasticsearch
es = Elasticsearch()

question_len = 100
input_text_len = 128
Word2Vec_len = 300
pos_len = 46
slide_size = 64 

result_dict = dict()

my_file = open("data_set_fix.json",'r',encoding = 'utf-8-sig')
txt = my_file.read()

json_obj = json.loads(txt)

model = load_model("train_model_v3\model_Unet_200_0_00563.h5")
for num_train_file in range(14001,15001):
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")
    x_train = x_train[:,:,:]
    print(y_train.shape)

    question = json_obj['data'][num_train_file-1]['question'].lower()
    article_id = json_obj['data'][num_train_file-1]['article_id']
    answer = json_obj['data'][num_train_file-1]['answer']
    GT_being = int(json_obj['data'][num_train_file-1]['answer_begin_position'])
    GT_end = int(json_obj['data'][num_train_file-1]['answer_end_position'])

    txt = es.get(index="test_search_engine_v1",doc_type='_doc',id=str(article_id))
    txt = txt['_source']['text']
    txt = [x.lower() for x in txt]
    pred_weight = model.predict(np.asarray(x_train))
    begin_count = 1

    all_pred_ans = list()
    for i in range(0,math.ceil(len(txt)/slide_size)):
        input_text = txt[i*slide_size:i*slide_size+input_text_len-1]
        name_entity = ner.get_ner(input_text)
        pos = pos_tag(input_text)

        overlap = 0
        pred_ans = list()
        for j in range(0,len(input_text)):
            if(j>=slide_size):
                overlap = overlap+len(input_text[j])
            pred_ans.append((begin_count,input_text[j],str(pred_weight[i][j][0]),str(y_train[i][j]),(name_entity[j][1],name_entity[j][2]),pos[j][1]))
            begin_count = begin_count + len(input_text[j])
        begin_count = begin_count-overlap
        all_pred_ans.append(pred_ans)
        #print(pred_ans)
    result_dict[num_train_file] = {'question':question,'article_id':article_id,'answer':answer,'GT_start':GT_being,'GT_end':GT_end,'frame_data':all_pred_ans}

#print(result_dict)
json_obj = json.dumps(result_dict,ensure_ascii=False)
target_file = open("result_to_post process.json","w",encoding = 'utf-8')
target_file.write(json_obj)
target_file.close()
