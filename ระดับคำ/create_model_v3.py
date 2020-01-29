from keras.models import Model, Sequential,load_model
from keras.layers import  Input, Dense,Conv1D, MaxPooling1D ,UpSampling1D, GlobalAveragePooling1D,BatchNormalization , Activation,Dropout,Concatenate
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

#Create Model
question_len = 100
input_text_len = 128
Word2Vec_len = 300
pos_len = 46
slide_size = 64 #overlap 100

Input_layer = Input(shape=(input_text_len,(question_len+Word2Vec_len+pos_len),))
#Contracting Path
Con1 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Input_layer)
Bat1 = BatchNormalization()(Con1)
Act1 = Activation("relu")(Bat1)

Con2 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Act1)
Bat2 = BatchNormalization()(Con2)
Act2 = Activation("relu")(Bat2)

Down1 = MaxPooling1D(2)(Act2)
Drop1 = Dropout(0.1)(Down1)

Con3 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Drop1)
Bat3 = BatchNormalization()(Con3)
Act3 = Activation("relu")(Bat3)

Con4 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Act3)
Bat4 = BatchNormalization()(Con4)
Act4 = Activation("relu")(Bat4)

Down2 = MaxPooling1D(2)(Act4)
Drop2 = Dropout(0.1)(Down2)

Con5 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Drop2)
Bat5 = BatchNormalization()(Con5)
Act5 = Activation("relu")(Bat5)

Con6 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Act5)
Bat6 = BatchNormalization()(Con6)
Act6 = Activation("relu")(Bat6)

Down3 = MaxPooling1D(2)(Act6)
Drop3 = Dropout(0.1)(Down3)

Con7 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Drop3)
Bat7 = BatchNormalization()(Con7)
Act7 = Activation("relu")(Bat7)

Con8 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Act7)
Bat8 = BatchNormalization()(Con8)
Act8 = Activation("relu")(Bat8)

Down4 = MaxPooling1D(2)(Act8)
Drop4 = Dropout(0.1)(Down4)

#Middle Path
Con9 = Conv1D(256, 3,padding="same", kernel_initializer='he_normal')(Drop4)
Bat9 = BatchNormalization()(Con9)
Act9 = Activation("relu")(Bat9)

Con10 = Conv1D(256, 3,padding="same", kernel_initializer='he_normal')(Act9)
Bat10 = BatchNormalization()(Con10)
Act10 = Activation("relu")(Bat10)

#expansive path
Up1 = UpSampling1D(2)(Act10)
Up1 = Concatenate()([Up1,Act8])
Drop5 = Dropout(0.1)(Up1)

Con11 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Drop5)
Bat11 = BatchNormalization()(Con11)
Act11 = Activation("relu")(Bat11)

Con12 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Act11)
Bat12 = BatchNormalization()(Con12)
Act12 = Activation("relu")(Bat12)

Up2 = UpSampling1D(2)(Act12)
Up2 = Concatenate()([Up2,Act6])
Drop6 = Dropout(0.1)(Up2)

Con13 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Drop6)
Bat13 = BatchNormalization()(Con13)
Act13 = Activation("relu")(Bat13)

Con14 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Act13)
Bat14 = BatchNormalization()(Con14)
Act14 = Activation("relu")(Bat14)

Up3 = UpSampling1D(2)(Act14)
Up3 = Concatenate()([Up3,Act4])
Drop7 = Dropout(0.1)(Up3)

Con15 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Drop7)
Bat15 = BatchNormalization()(Con15)
Act15 = Activation("relu")(Bat15)

Con16 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Act15)
Bat16 = BatchNormalization()(Con16)
Act16 = Activation("relu")(Bat16)

Up4 = UpSampling1D(2)(Act16)
Up4 = Concatenate()([Up4,Act2])
Drop8 = Dropout(0.1)(Up4)

Con17 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Drop8)
Bat17 = BatchNormalization()(Con17)
Act17 = Activation("relu")(Bat17)

Con18 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Act17)
Bat18 = BatchNormalization()(Con18)
Act18 = Activation("relu")(Bat18)

Output_layer = Conv1D(1, 3, activation='sigmoid',padding="same", kernel_initializer='he_normal')(Act18)


model = Model(inputs=Input_layer, outputs=Output_layer)

model.summary()

#Save Model Diagram
from keras.utils import plot_model
plot_model(model, to_file='model_Unet_Con1d.png')
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

import json

from elasticsearch import Elasticsearch
es = Elasticsearch()

my_file = open("data_set_fix.json",'r',encoding = 'utf-8-sig')
txt = my_file.read()

json_obj = json.loads(txt)

def show_result(model,num_train_file):    
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data_3\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_3\output\output_A_"+str(num_train_file)+".npy")
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
"""       
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#binary_crossentropy

all_x_train = list()
all_y_train = list()

all_x_val = list()
all_y_val = list()
"""
#Load Data Tarin
for num_train_file in range(1,14001):
    print("Load Data : ",num_train_file)
    #x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")

    #for x in x_train: 
        #all_x_train.append(x[:,:])
    for y in y_train:
        all_y_train.append(np.reshape(y,(y.shape[0],1)))
    #print(y_train)
"""
#Load Data Val
for num_train_file in range(14001,15001):
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")

    for x in x_train: 
        all_x_val.append(x[:,:])
    for y in y_train:
        all_y_val.append(np.reshape(y,(y.shape[0],1)))

#all_x_train = np.asarray(all_x_train)
#all_y_train = np.asarray(all_y_train)

all_x_val = np.asarray(all_x_val)
all_y_val = np.asarray(all_y_val)

#print(all_y_train.shape,all_y_val.shape)

"""
#Load Train Data
def generator_train_data():
    while(True):
        for num_train_file in range(1,14001):
            all_x_train = list()
            all_y_train = list()
            print("Load Data_Train: ",num_train_file)
            x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
            y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")
            for x in x_train: 
                all_x_train.append(x[:,:])
            for y in y_train:
                all_y_train.append(np.reshape(y,(y.shape[0],1)))
            all_x_train = np.asarray(all_x_train)
            all_y_train = np.asarray(all_y_train)
            yield (all_x_train,all_y_train)
            del(all_x_train)
            del(all_y_train)
        #print(all_y_train.shape)
        #yield all_x_train,all_y_train
        #print(y_train)
#Load Val Data
def generator_val_data():
    while(True):
        for num_train_file in range(14001,15001):
            all_x_val = list()
            all_y_val = list()
            print("Load Data_Val: ",num_train_file)
            x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
            y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")
            for x in x_train: 
                all_x_val.append(x[:,:])
            for y in y_train:
                all_y_val.append(np.reshape(y,(y.shape[0],1)))
            #print(y_train)
            all_x_val = np.asarray(all_x_val)
            all_y_val = np.asarray(all_y_val)
            yield (all_x_val,all_y_val)
            del(all_x_val)
            del(all_y_val)

"""
#Train Model
BATCH_SIZE = 50
EPOCHS = 30

checkpoint = ModelCheckpoint('train_model_v3\model_Unet.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')

model = load_model('train_model_v3\model_Unet_final.h5')

#print(len(all_y_train))

for i in range(0,EPOCHS):
    for num_train_file in range(1,14001):
        print("EPOCH ",i," Load Data : ",num_train_file)
        x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
        y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")

        for x in x_train: 
            all_x_train.append(x[:,:])
        for y in y_train:
            all_y_train.append(np.reshape(y,(y.shape[0],1)))
        print("Memory used :",sys.getsizeof(all_x_train))
        
        if(sys.getsizeof(all_x_train)>=80000):
            all_x_train = np.asarray(all_x_train)
            all_y_train = np.asarray(all_y_train)

            history = model.fit(all_x_train,
                            all_y_train,
                            batch_size=BATCH_SIZE,
                            epochs=1,
                            callbacks=[checkpoint],
                            validation_data=(all_x_val,all_y_val))

            model.save('train_model_v3\model_Unet_final.h5')

            all_x_train = list()
            all_y_train = list()
"""
history = model.fit_generator(generator=generator_train_data(),
                              validation_data=generator_val_data(),
                              epochs=EPOCHS,
                              callbacks=[checkpoint],
                              steps_per_epoch=(215025//BATCH_SIZE),
                              validation_steps=(18458//BATCH_SIZE))
"""
#215025,18458
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('train_model_v3\his\model_v3_epoch_final.png')
plt.clf()
"""
#show result after train
print("--------------SHOW RESULT---------------")
for num_train_file in range(1,1001):
    show_result(model,num_train_file)
"""


