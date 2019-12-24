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
Word2Vec_len = 300
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

        np.save("train_data_2\input\input_A_"+str(k),all_input)
        np.save("train_data_2\output\output_A_"+str(k),all_output)

for count_data , data in enumerate(json_obj['data'][0:1100],start=1):
    if(data['question_type']==1):
        begin_count = 1
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

        print("TXT : ",txt)

        ck_answer = list()
        answer = data['answer']

        print("t ->",math.ceil(len(txt)/slide_size))
        for i in range(0,math.ceil(len(txt)/slide_size)):
                pre_data = np.zeros((input_text_len,question_len+Word2Vec_len),dtype=np.float32)
                pre_ans = np.zeros((input_text_len),dtype=np.float32)
                input_text = txt[i*slide_size:i*slide_size+input_text_len-1]

                print("INPUT TXT : ",input_text)
                #print(len(txt)," > ",i,":",i*slide_size,"-",i*slide_size+input_text_len)
                #get input feature
                for n_j,j in enumerate(input_text,start=0):
                        for n_k,k in enumerate(question,start=0):
                                if(j in model.wv.vocab.keys() and k in model.wv.vocab.keys()):
                                        pre_data[n_j,n_k] = model.wv.similarity(j,k)
                        if(j in model.wv.vocab.keys()):
                                pre_data[n_j,question_len:question_len+Word2Vec_len] = (model.wv.get_vector(j)+2.936043)/(2.8039522+2.936043)
                #draw_heat_map
                #import heat_map
                #heat_map.make_heatmap("heatmap/"+str(question_id)+"_"+str(i)+".png",question,input_text,pre_data)

                temp = 0
                for n in input_text:
                        temp=temp+len(n)
                        print(n,"|",temp)

                end_count = begin_count+temp

                print("rang begin end: ",begin_count," - ",end_count)
                print("ans_rang : ",answer_begin_position,"-",answer_end_position)

                #get output
                curent = begin_count
                overlap = 0
                for n_target,target_data in enumerate(input_text,start=0):
                        if(not((answer_end_position <= curent) or (answer_begin_position >= (curent+len(target_data))))):
                                pre_ans[n_target] = 1.0
                                ck_answer.append(target_data)
                                print("rang : ",curent,"-", (curent+len(target_data)))
                                print(target_data)
                        if(n_target>=slide_size):
                                print("OVER LAP : ",target_data)
                                overlap=overlap+len(target_data)
                        curent = curent+len(target_data)
                begin_count = end_count-overlap
                    

                #add input and output to list
                all_input.append(pre_data)
                all_output.append(pre_ans)

        print("ANS : ",ck_answer)
        print("CK_ANS : ",answer)
        #check_point
        ck_point(all_input,all_output,int(question_id))
        all_input.clear()
        all_output.clear()

        ck_answer.clear()
        #print(pre_data)

#save final
#ck_point(all_input,all_output,math.ceil((last_data_count+1)/ck_point_time))
