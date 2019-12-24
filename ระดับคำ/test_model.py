"""
from keras.models import Model, Sequential
from keras.layers import  Input, Dense, Conv2D, MaxPool2D, Flatten ,Conv1D, Reshape
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math

#Create Model
question_len = 100
input_text_len = 100

input = Input(shape = (question_len,input_text_len,2))
conv1 = Conv2D(100,(5,5),activation='relu',padding="same")(input)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(100,(5,5),activation='relu',padding="same")(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(1,(3,3),strides=(1,1),activation='relu',padding="same")(pool2)

re_size =  Reshape(target_shape=(25,25))(conv3)
conv4 = Conv1D(10, 3, activation='relu',padding="same")(re_size)
flat = Flatten()(conv4)
h1 = Dense(100, activation='relu')(flat)
output = Dense(3, activation='sigmoid')(h1)
model = Model(inputs=input, outputs=output)

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.load_weights('train_model_v1/model_v1.h5')

def show_result(model,input_data,gt):
        for i,data_gt in enumerate(gt,start=0):
            if(data_gt[0]==1.0):
                pred = model.predict(np.asarray([input_data[i]]))
                print("GT : ",data_gt)
                print("Pred : ",pred)


x_train = np.load("train_data\input\input_A_1.npy")
y_train = np.load("train_data\output\output_A_1.npy")

print(x_train.shape)

#show_result(model,x_train,y_train)
#print(model.predict(x_train))
"""
import json

my_file = open("data_set_fix.json",'r',encoding = 'utf-8-sig')
txt = my_file.read()

json_obj = json.loads(txt)

article_id = json_obj['data'][0]['article_id']
print(article_id)

import numpy as np

x_train = np.load("train_data_2\input\input_A_"+str(1)+".npy")
y_train = np.load("train_data_2\output\output_A_"+str(1)+".npy")

from elasticsearch import Elasticsearch
es = Elasticsearch()
txt = es.get(index="test_search_engine_v1",doc_type='_doc',id=str(115035))

print(txt['_source']['text'][76-25+3])
print(y_train[1].argmax())