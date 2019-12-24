from keras.models import Model, Sequential
from keras.layers import  Input, Dense,Conv1D, MaxPooling1D ,UpSampling1D, GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math

#Create Model
question_len = 100
Word2Vec_len = 300
input_text_len = 100

Input_layer = Input(shape=(input_text_len,(question_len+Word2Vec_len),))
Con1 = Conv1D(16, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Input_layer)
Con2 = Conv1D(16, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Con1)
Down1 = MaxPooling1D(2)(Con2)
Con3 = Conv1D(32, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Down1)
Con4 = Conv1D(32, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Con3)
Down2 = MaxPooling1D(2)(Con4)
Con5 = Conv1D(64, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Down2)
Con6 = Conv1D(64, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Con5)
Up1 = UpSampling1D(2)(Con6)
Con7 = Conv1D(32, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Up1)
Con8 = Conv1D(32, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Con7)
Up2 = UpSampling1D(2)(Con8)
Con9 = Conv1D(16, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Up2)
Con10 = Conv1D(16, 3, activation='relu',padding="same", kernel_initializer='he_normal')(Con9)
Output_layer = Conv1D(1, 3, activation='sigmoid',padding="same", kernel_initializer='he_normal')(Con10)


model = Model(inputs=Input_layer, outputs=Output_layer)

model.summary()
"""
#Train model
def iou(y_true, y_pred):
    ck1 = np.asarray([1,0,0])
    ck2 = np.asarray([0,1,0])
    ck3 = np.asarray([0,0,1])


    k_ck1 = K.variable(ck1,dtype="float32")
    k_ck2 = K.variable(ck1,dtype="float32")
    k_ck3 = K.variable(ck1,dtype="float32")

    val_ck1_pred = K.sum(y_pred*k_ck1)
    val_ck1_true = K.sum(y_true*k_ck1)
    
    val_ck1_pred = K.greater(val_ck1_pred, 0.5)
    val_ck1_true = K.greater(val_ck1_true, 0.5)

    val_ck2_pred = K.sum(y_pred*k_ck2)
    val_ck2_true = K.sum(y_true*k_ck2)    

    val_ck3_pred = K.sum(y_pred*k_ck3)
    val_ck3_true = K.sum(y_true*k_ck3)

    I = K.maximum(1.0,K.minimum(val_ck3_pred,val_ck3_true))-K.minimum(K.maximum(val_ck2_pred,val_ck2_true),0.0)
    U = K.maximum(K.maximum(val_ck3_pred,val_ck3_true),1.0) - K.minimum(0.0,K.minimum(val_ck2_pred,val_ck2_true))
    IOU = (I+K.epsilon())/(U+K.epsilon())
    #return IOU
    return IOU*K.cast(val_ck1_true,dtype="float32")
"""
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
        #input()
        for j in range(0,len(pred[i])):
            if(pred[i][j][0]>0.5):
                #print("OK")
                #print(input_text[j])
                pre_ans.append(input_text[j])

    print("QUESTION_ID : ",num_train_file)
    print("Q : ",question)
    print("GT : ",answer)
    print("Pred : ",pre_ans)
    pre_ans.clear()
            



model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#binary_crossentropy

all_x_train = list()
all_y_train = list()

all_x_val = list()
all_y_val = list()
#Load Data
for num_train_file in range(1,1001):
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")

    for x in x_train: 
        all_x_train.append(x[:,:])
    for y in y_train:
        all_y_train.append(np.reshape(y,(y.shape[0],1)))
    #print(y_train)

for num_train_file in range(1001,1101):
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")

    for x in x_train: 
        all_x_val.append(x[:,:])
    for y in y_train:
        all_y_val.append(np.reshape(y,(y.shape[0],1)))
    #print(y_train)

all_x_train = np.asarray(all_x_train)
all_y_train = np.asarray(all_y_train)

all_x_val = np.asarray(all_x_val)
all_y_val = np.asarray(all_y_val)

#Train Model
BATCH_SIZE = 10
EPOCHS = 500

checkpoint = ModelCheckpoint('train_model_v3\model_v3.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')

#print(len(all_y_train))

history = model.fit(all_x_train,
                    all_y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=[checkpoint],
                    validation_data=(all_x_val,all_y_val))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('train_model_v3\his\model_v3_epoch_final.png')
plt.clf()

model.save('train_model_v3\model_v3_final.h5')

#show result after train
print("--------------SHOW RESULT---------------")
for num_train_file in range(1,1001):
    show_result(model,num_train_file)



