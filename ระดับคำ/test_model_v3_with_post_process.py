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

import deepcut

question_len = 100
input_text_len = 128
Word2Vec_len = 300
pos_len = 46
slide_size = 64 

my_file = open("data_set_fix.json",'r',encoding = 'utf-8-sig')
txt = my_file.read()

json_obj = json.loads(txt)

#Load Num Unit
num_unit_file = open("all_num_unit.txt",'r',encoding = 'utf-8-sig')
num_unit_txt = num_unit_file.read()
all_num_unit = num_unit_txt.split("|")
num_unit_file.close()

iou_T = 0
iou_M = 0 
iou_F = 0

def overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2) + 1)

def union(min1, max1, min2, max2):
    return (max1-min1 + 1)+(max2-min2 + 1 )-overlap(min1,max1,min2,max2)

def iou(x1,y1,x2,y2):
    i = overlap(x1,y1,x2,y2)
    u = union(x1,y1,x2,y2)
    return(i/u)

def predict_have_num(question):
    question_troken = deepcut.tokenize(question)
    if(question.find("เท่าไร")!=-1 or question.find("เท่าไหร่")!=-1 or question.find("เท่าใด")!=-1  or question.find("พ.ศ. ใด")!=-1 or question.find("ค.ศ. ใด")!=-1 or question.find("ปีใด")!=-1 or question.find("ปีอะไร")!=-1):
        return True
    elif("กี่" in question_troken):
        return True
    else:
        return False

def show_result(model,num_train_file):    
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")
    x_train = x_train[:,:,:]

    question = json_obj['data'][num_train_file-1]['question'].lower()
    article_id = json_obj['data'][num_train_file-1]['article_id']
    answer = json_obj['data'][num_train_file-1]['answer']
    GT_being = int(json_obj['data'][num_train_file-1]['answer_begin_position'])
    GT_end = int(json_obj['data'][num_train_file-1]['answer_end_position'])

    txt = es.get(index="test_search_engine_v1",doc_type='_doc',id=str(article_id))
    txt = txt['_source']['text']
    txt = [x.lower() for x in txt]
    pred = model.predict(np.asarray(x_train))
    pre_ans = list()

    begin_count = 1

    for i in range(0,math.ceil(len(txt)/slide_size)):
        input_text = txt[i*slide_size:i*slide_size+input_text_len-1]
        #print(pred[i][1][0] > 0.5)
        #print(y_train[i])
        #print(pred[i])
        #input()
        
        overlap_position = 0
        for j in range(0,len(input_text)):
            if(j>=slide_size):
                overlap_position = overlap_position+len(input_text[j])
            if(pred[i][j][0]>0.12):
                #print("OK")
                #print(input_text[j])
                pre_ans.append((begin_count,input_text[j]))
            begin_count = begin_count + len(input_text[j])
        begin_count = begin_count-overlap_position

    pre_ans = sorted(list(set(pre_ans)),key = lambda i:i[0])
    print("QUESTION_ID : ",num_train_file)
    print("Q : ",question)
    print("GT : ",answer)
    #ทดสอบแยก name entity
    pre_name_entity = list()
    pre_pos_tag = list()
    if(pre_ans != []):
        pre_name_entity = ner.get_ner([item[1] for item in pre_ans])

    #pre_ans_respose = filter(lambda x:x[2] != 'O',pre_ans_respose)
    #pre_ans_respose = "".join(item[0] for item in pre_ans_respose)
    ####
    #print("NE Ans : ",pre_name_entity)
    #print("POS Ans : ",pos_tag([item[1] for item in pre_ans]))
    #print(pre_ans)

    result_ans = list()
    #แยกว่าเป็นคำถามที่คำตอบเป็นตัวเลขป่าว
    if(predict_have_num(question)):
        print("Ok")
        for num_ans_index,num_ans in enumerate(pre_name_entity,start=0):
            if(num_ans[1] == "NUM" or pre_ans[num_ans_index][1] in all_num_unit or pre_ans[num_ans_index][1].find("พ.ศ")!=-1 or pre_ans[num_ans_index][1].find("ค.ศ")!=-1):
                result_ans.append(pre_ans[num_ans_index])
    else:
        result_ans = pre_ans

    #pre_ans_respose = "".join([item[1] for item in result_ans])

    pre_being = 0
    pre_end = 0
    if(result_ans != []):
        pre_being = result_ans[0][0]
        pre_end = result_ans[len(result_ans)-1][0]+len(result_ans[len(result_ans)-1][1])
    print("Pred Start Ans : ",pre_being)
    print("Pred End Ans : ",pre_end)
    #("Pre_Ans_with_Postion : ",pre_ans)
    iou_score = iou(GT_being,GT_end,pre_being,pre_end)
    print("IOU : ",iou_score)
    
    global iou_T,iou_F,iou_M
    if(iou_score==1.0): 
        iou_T = iou_T + 1
    elif(iou_score==0.0): 
        iou_F = iou_F + 1
    else: 
        iou_M = iou_M + 1

    begin_count = 1
    pre_ans_respose = ""
    pre_ans = list()
    for i in range(0,math.ceil(len(txt)/slide_size)):
        input_text = txt[i*slide_size:i*slide_size+input_text_len-1]
        overlap_position = 0
        for j in range(0,len(input_text)):
            if(j>=slide_size):
                overlap_position = overlap_position+len(input_text[j])
            if(begin_count >= pre_being and (begin_count + len(input_text[j])) <= pre_end):
                pre_ans.append((begin_count,input_text[j]))
            begin_count = begin_count + len(input_text[j])
        begin_count = begin_count-overlap_position

    pre_ans = sorted(list(set(pre_ans)),key = lambda i:i[0])
    pre_ans_respose = "".join([item[1] for item in pre_ans])
    print("Pred Ans : ",pre_ans_respose)
    
    if(pre_ans_respose.find(answer)>=0):
        return True,len(answer)/len(pre_ans_respose),iou_score
    pre_ans.clear()
    return False,0,iou_score

model = load_model("train_model_v3\model_Unet_230_0_00559.h5")

print("--------------SHOW RESULT---------------")
t_score = 0
iou_subword = list()
iou_position = list()
f_score = 0
for num_train_file in range(14901,14911):
    ck1,iou1,iou2 = show_result(model,num_train_file)
    if(ck1):
        t_score = t_score + 1
    else:
        f_score = f_score + 1
    iou_subword.append(iou1)
    iou_position.append(iou2)
print("True :",t_score,"/",(t_score+f_score))
print("False :",f_score,"/",(t_score+f_score))
print("IOU_Word :",np.mean(iou_subword))
print("IOU_Position :",np.mean(iou_position))
print("IOU_T :",iou_T)
print("IOU_M :",iou_M)
print("IOU_F :",iou_F)
