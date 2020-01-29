from keras.models import Model, Sequential,load_model
from keras.layers import  Input, Dense,Conv1D, MaxPooling1D ,UpSampling1D, GlobalAveragePooling1D,BatchNormalization , Activation,Dropout,Concatenate
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

#Create Model
question_len = 128
input_text_len = 128
Word2Vec_len = 300
pos_len = 46
slide_size = 64 #overlap 64

#Input Path
Input_layer_pagger = Input(shape=(input_text_len,(100+Word2Vec_len+pos_len),))
Input_layer_question = Input(shape=(question_len,(128+Word2Vec_len+pos_len),))

#Contracting Path 1
Con1 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Input_layer_pagger)
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

#Contracting Path 2
Con1_2 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Input_layer_question)
Bat1_2 = BatchNormalization()(Con1_2)
Act1_2 = Activation("relu")(Bat1_2)

Con2_2 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Act1_2)
Bat2_2 = BatchNormalization()(Con2_2)
Act2_2 = Activation("relu")(Bat2_2)

Down1_2 = MaxPooling1D(2)(Act2_2)
Drop1_2 = Dropout(0.1)(Down1_2)

Con3_2 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Drop1_2)
Bat3_2 = BatchNormalization()(Con3_2)
Act3_2 = Activation("relu")(Bat3_2)

Con4_2 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Act3_2)
Bat4_2 = BatchNormalization()(Con4_2)
Act4_2 = Activation("relu")(Bat4_2)

Down2_2 = MaxPooling1D(2)(Act4_2)
Drop2_2 = Dropout(0.1)(Down2_2)

Con5_2 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Drop2_2)
Bat5_2 = BatchNormalization()(Con5_2)
Act5_2 = Activation("relu")(Bat5_2)

Con6_2 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Act5_2)
Bat6_2 = BatchNormalization()(Con6_2)
Act6_2 = Activation("relu")(Bat6_2)

Down3_2 = MaxPooling1D(2)(Act6_2)
Drop3_2 = Dropout(0.1)(Down3_2)

Con7_2 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Drop3_2)
Bat7_2 = BatchNormalization()(Con7_2)
Act7_2 = Activation("relu")(Bat7_2)

Con8_2 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Act7_2)
Bat8_2 = BatchNormalization()(Con8_2)
Act8_2 = Activation("relu")(Bat8_2)

Down4_2 = MaxPooling1D(2)(Act8_2)
Drop4_2 = Dropout(0.1)(Down4_2)

#Middle Path
Con9 = Concatenate()([Drop4,Drop4_2])
Con9 = Conv1D(256, 3,padding="same", kernel_initializer='he_normal')(Con9)
Bat9 = BatchNormalization()(Con9)
Act9 = Activation("relu")(Bat9)

Con10 = Conv1D(256, 3,padding="same", kernel_initializer='he_normal')(Act9)
Bat10 = BatchNormalization()(Con10)
Act10 = Activation("relu")(Bat10)

#expansive path
Up1 = UpSampling1D(2)(Act10)
Up1 = Concatenate()([Up1,Act8,Act8_2])
Drop5 = Dropout(0.1)(Up1)

Con11 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Drop5)
Bat11 = BatchNormalization()(Con11)
Act11 = Activation("relu")(Bat11)

Con12 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Act11)
Bat12 = BatchNormalization()(Con12)
Act12 = Activation("relu")(Bat12)

Up2 = UpSampling1D(2)(Act12)
Up2 = Concatenate()([Up2,Act6,Act6_2])
Drop6 = Dropout(0.1)(Up2)

Con13 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Drop6)
Bat13 = BatchNormalization()(Con13)
Act13 = Activation("relu")(Bat13)

Con14 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Act13)
Bat14 = BatchNormalization()(Con14)
Act14 = Activation("relu")(Bat14)

Up3 = UpSampling1D(2)(Act14)
Up3 = Concatenate()([Up3,Act4,Act4_2])
Drop7 = Dropout(0.1)(Up3)

Con15 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Drop7)
Bat15 = BatchNormalization()(Con15)
Act15 = Activation("relu")(Bat15)

Con16 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Act15)
Bat16 = BatchNormalization()(Con16)
Act16 = Activation("relu")(Bat16)

Up4 = UpSampling1D(2)(Act16)
Up4 = Concatenate()([Up4,Act2,Act2_2])
Drop8 = Dropout(0.1)(Up4)

Con17 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Drop8)
Bat17 = BatchNormalization()(Con17)
Act17 = Activation("relu")(Bat17)

Con18 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Act17)
Bat18 = BatchNormalization()(Con18)
Act18 = Activation("relu")(Bat18)

Output_layer = Conv1D(1, 3, activation='sigmoid',padding="same", kernel_initializer='he_normal')(Act18)


model = Model(inputs=[Input_layer_pagger,Input_layer_question], outputs=Output_layer)

model.summary()

#Save Model Diagram
from keras.utils import plot_model
plot_model(model, to_file='model_Unet_Con1d_Bi_Input.png')

all_x_train = list()
all_x_train_question = list()
all_y_train = list()

all_x_val = list()
all_x_val_question = list()
all_y_val = list()


#Load Data Val
for num_train_file in range(14001,14101):
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data_2_2\input\input_A_"+str(num_train_file)+".npy")
    x_train_question = np.load("train_data_2_2\input_question\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_2_2\output\output_A_"+str(num_train_file)+".npy")

    for x in x_train: 
        #print(x.shape)
        all_x_val.append(x)
    for x_question in x_train_question:
        #print(x_question.shape)
        all_x_val_question.append(x_question)
    for y in y_train:
        all_y_val.append(np.reshape(y,(y.shape[0],1)))


all_x_val = np.asarray(all_x_val)
all_x_val_question = np.asarray(all_x_val_question)
all_y_val = np.asarray(all_y_val)

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Train Model
BATCH_SIZE = 100
EPOCHS = 30

checkpoint = ModelCheckpoint('train_model_v4_2\model_Unet_bi.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')

model = load_model('train_model_v4_2\model_Unet_bi_final.h5')

for i in range(0,EPOCHS):
    for num_train_file in range(1,14001):
        print("EPOCH ",i," Load Data : ",num_train_file)
        x_train = np.load("train_data_2_2\input\input_A_"+str(num_train_file)+".npy")
        x_train_question = np.load("train_data_2_2\input_question\input_A_"+str(num_train_file)+".npy")
        y_train = np.load("train_data_2_2\output\output_A_"+str(num_train_file)+".npy")

        for x in x_train: 
            all_x_train.append(x)
        for x_question in x_train_question:
            all_x_train_question.append(x_question)
        for y in y_train:
            all_y_train.append(np.reshape(y,(y.shape[0],1)))
        print("Memory used :",(sys.getsizeof(all_x_train)+sys.getsizeof(all_x_train_question)))
        
        if((sys.getsizeof(all_x_train)+sys.getsizeof(all_x_train_question))>=150000):
            all_x_train = np.asarray(all_x_train)
            all_x_train_question = np.asarray(all_x_train_question)
            all_y_train = np.asarray(all_y_train)
            
            """
            history = model.fit([all_x_train,all_x_train_question],
                            all_y_train,
                            batch_size=BATCH_SIZE,
                            epochs=1)
            """
            history = model.fit([all_x_train,all_x_train_question],
                            all_y_train,
                            batch_size=BATCH_SIZE,
                            epochs=1,
                            callbacks=[checkpoint],
                            validation_data=([all_x_val,all_x_val_question],all_y_val))
        

            model.save('train_model_v4_2\model_Unet_bi_final.h5')

            all_x_train = list()
            all_x_train_question = list()
            all_y_train = list()

