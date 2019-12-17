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

        max_question_len = max(max_question_len,len(question))

        print(question)

print("max_question_len:",max_question_len)
"""

import numpy as np
import math
import all_char

question_len = 300
input_text_len = 50
slide_size = 25 #overlap 100
size_char = len(all_char.all)

all_input = list()
all_output = list()

ck_point_time = 50
last_data_count = 0

#save file
def ck_point(all_input,all_output,k):
        print("SAVE ",k)
        all_input = np.asarray(all_input)
        all_output = np.asarray(all_output)

        np.save("train_data\input\input_A_"+str(k),all_input)
        np.save("train_data\output\output_A_"+str(k),all_output)

for count_data , data in enumerate(json_obj['data'],start=1):
    if(data['question_type']==1):
        question_id = data['question_id']
        print("QUESTION_ID: ",question_id)

        question = data['question'].lower()
        answer_begin_position = data['answer_begin_position']
        answer_end_position = data['answer_end_position']
        article_id = data['article_id']
        txt = open("../ex_data/documents-nsc/documents-nsc/"+str(article_id)+".txt",'r',encoding = 'utf-8-sig')
        txt = txt.read()
        for i in range(0,math.floor(len(txt)/slide_size)):
                pre_data = np.zeros((question_len,input_text_len,size_char),dtype=np.float32)
                pre_ans = np.zeros((3),dtype=np.float32)
                input_text = txt[i*slide_size:i*slide_size+input_text_len-1].lower()
                print(len(txt)," > ",i,":",i*slide_size,"-",i*slide_size+input_text_len)
                #get input feature
                for n_j,j in enumerate(input_text,start=0):
                        for n_k,k in enumerate(question,start=0):
                                if(j == k and j in all_char.all):
                                        if(n_j >= 300 or n_k >= 300):
                                                print(n_j,n_k)
                                                print(j)
                                                print(input_text)
                                                print(len(input_text))
                                        pre_data[n_k,n_j,all_char.all.index(j)] = 1.0
                #get output
                if(not((answer_end_position < i*slide_size) or (answer_begin_position > i*slide_size+input_text_len-1))):
                    pre_ans[0] = 1.0
                    if(answer_begin_position > i*slide_size):
                        pre_ans[1] = answer_begin_position/(i*slide_size+input_text_len-1)

                    if(answer_end_position < i*slide_size+input_text_len-1):
                        pre_ans[2] = answer_end_position/(i*slide_size+input_text_len-1)
                    elif(answer_end_position >= i*slide_size+input_text_len-1):
                        pre_ans[2] = 1.0

                #add input and output to list
                all_input.append(pre_data)
                all_output.append(pre_ans)

        del(pre_data)
        del(pre_ans)
        del(input_text)
        del(txt)
        #check_point
        if(count_data%ck_point_time==0):
                ck_point(all_input,all_output,math.floor(count_data/ck_point_time))
                all_input.clear
                all_output.clear
                last_data_count = count_data
                #print(pre_data)

#save final
ck_point(all_input,all_output,math.floor((last_data_count+1)/ck_point_time))