from gensim.models import Word2Vec
from scipy.spatial import distance

# load word2vec
model = Word2Vec.load('Word2Vec_1.model') #load model

#print(model.wv.vocab.keys())
#print("Consin:",model.wv.similarity("พรีเดเตอร์","มอนสเตอร์"))
#print("Eucli:",distance.euclidean(model.wv.get_vector("พรีเดเตอร์"), model.wv.get_vector("มอนสเตอร์")))

import deepcut
import json

my_file = open("data_set_fix.json",'r',encoding = 'utf-8-sig')
txt = my_file.read()

json_obj = json.loads(txt)

#หา max_question_len
"""
max_question_len = 0
for data in json_obj['data']:
    if(data['question_type']==1):
        question = data['question']
        answer_begin_position = data['answer_begin_position']
        answer_end_position = data['answer_end_position']

        max_question_len = max(max_question_len,len(deepcut.tokenize(question)))

        #print(question)

print("max_question_len:",max_question_len)
max_question_len = 54
"""

import numpy as np
import math
from elasticsearch import Elasticsearch
es = Elasticsearch()

question_len = 100
input_text_len = 100
slide_size = 75 #overlap 100

all_input = list()
all_output = list()

ck_point_time = 1000
last_data_count = 0

#res2 = es.get(index="test_search_engine_v1",doc_type='_doc',id=str(69492))
#res2 = res2['_source']['text'][0:10]
#res2 = [x.lower() for x in res2]
#print(res2)

#save file
def ck_point(all_input,all_output,k):
        print("SAVE ",k)
        all_input = np.asarray(all_input)
        all_output = np.asarray(all_output)

        np.save("train_data\input\input_A_"+str(k),all_input)
        np.save("train_data\output\output_A_"+str(k),all_output)

for count_data , data in enumerate(json_obj['data'][0:1001],start=1):
    if(data['question_type']==1):
        begin_count = 0
        end_count = 0

        question_id = data['question_id']
        print("QUESTION_ID: ",question_id)

        question = data['question'].lower()
        question = deepcut.tokenize(question)
        answer_begin_position = data['answer_begin_position']
        answer_end_position = data['answer_end_position']
        article_id = data['article_id']
        txt = es.get(index="test_search_engine_v1",doc_type='_doc',id=str(article_id))
        txt = txt['_source']['text']
        txt = [x.lower() for x in txt]
        for i in range(0,math.floor(len(txt)/slide_size)):
                pre_data = np.zeros((question_len,input_text_len,2),dtype=np.float32)
                pre_ans = np.zeros((3),dtype=np.float32)
                input_text = txt[i*slide_size:i*slide_size+input_text_len-1]
                #print(len(txt)," > ",i,":",i*slide_size,"-",i*slide_size+input_text_len)
                #get input feature
                for n_j,j in enumerate(input_text,start=0):
                        for n_k,k in enumerate(question,start=0):
                                if(j in model.wv.vocab.keys() and k in model.wv.vocab.keys()):
                                        pre_data[n_k,n_j,0] = model.wv.similarity(j,k)
                                        pre_data[n_k,n_j,1] = distance.euclidean(model.wv.get_vector(j), model.wv.get_vector(k))
                #draw_heat_map
                #import heat_map
                #heat_map.make_heatmap("heatmap/"+str(question_id)+"_"+str(i)+".png",input_text,question,pre_data[:,:,0])

                temp = 0
                for n in input_text:
                        temp=temp+len(n)

                end_count = begin_count+temp

                #print("range : ",begin_count," - ",end_count)
                #print("ans_rang : ",answer_begin_position,"-",answer_end_position)

                #get output
                if(not((answer_end_position < begin_count) or (answer_begin_position > end_count))):
                    pre_ans[0] = 1.0
                    if(answer_begin_position > begin_count):
                        pre_ans[1] = answer_begin_position/(end_count)

                    if(answer_end_position < end_count):
                        pre_ans[2] = answer_end_position/(end_count)
                    elif(answer_end_position >= end_count):
                        pre_ans[2] = 1.0

                #print("Ans : ",pre_ans)

                #add input and output to list
                all_input.append(pre_data)
                all_output.append(pre_ans)

                #update count word
                #25 is overlapping
                overlap = 0
                for n in input_text[(i*slide_size+input_text_len-1)-(25):i*slide_size+input_text_len-1]:
                        overlap=overlap+len(n)
                begin_count = end_count-overlap

        #check_point
        if(count_data%ck_point_time==0):
                ck_point(all_input,all_output,math.floor(count_data/ck_point_time))
                all_input.clear()
                all_output.clear()
                last_data_count = count_data
                #print(pre_data)

#save final
#ck_point(all_input,all_output,math.floor((last_data_count+1)/ck_point_time))
