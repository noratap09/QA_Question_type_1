from keras.models import Model, Sequential,load_model
from keras.layers import  Input, Dense,Conv1D, MaxPooling1D ,GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math

import json

from elasticsearch import Elasticsearch
es = Elasticsearch()

question_len = 100
input_text_len = 100
slide_size = 75 #overlap 100

my_file = open("data_set_fix.json",'r',encoding = 'utf-8-sig')
txt = my_file.read()

json_obj = json.loads(txt)

def show_result(model,num_train_file):    
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")
    x_train = x_train[:,:,:]

    question = json_obj['data'][num_train_file-1]['question'].lower()
    article_id = json_obj['data'][num_train_file-1]['article_id']
    answer = json_obj['data'][num_train_file-1]['answer']

    txt = es.get(index="test_search_engine_v1",doc_type='_doc',id=str(article_id))
    txt = txt['_source']['text']
    txt = [x.lower() for x in txt]
    pred = model.predict(np.asarray(x_train))
    pre_ans = list()
    for i in range(0,math.ceil(len(txt)/slide_size)):
        input_text = txt[i*slide_size:i*slide_size+input_text_len-1]
        #print(pred[i][1][0] > 0.5)
        #print(y_train[i])
        #print(pred[i])
        #input()
        for j in range(0,len(input_text)):
            if(pred[i][j][0]>0.05):
                #print("OK")
                #print(input_text[j])
                pre_ans.append(input_text[j])

    print("QUESTION_ID : ",num_train_file)
    print("Q : ",question)
    print("GT : ",answer)
    print("Pred : ",pre_ans)
    pre_ans.clear()

model = load_model("train_model_v3\model_v3.h5")

print("--------------SHOW RESULT---------------")
t_score = 0
f_score = 0
for num_train_file in range(1001,1101):
    show_result(model,num_train_file)