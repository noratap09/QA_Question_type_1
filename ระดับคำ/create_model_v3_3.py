from keras.models import Model, Sequential,load_model
from keras.layers import  Input, Dense,Conv1D, MaxPooling1D ,UpSampling1D, GlobalAveragePooling1D,BatchNormalization , Activation,Dropout,Concatenate,Bidirectional,LSTM
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

import keras
from keras_self_attention import SeqSelfAttention

#Create Model
question_len = 100
input_text_len = 128
Word2Vec_len = 300
pos_len = 46
slide_size = 64 #overlap 100

Input_layer = Input(shape=(input_text_len,(question_len+Word2Vec_len+pos_len),))
#Contracting Path
Con1 = Conv1D(300, 3,padding="same", kernel_initializer='he_normal')(Input_layer)
Bat1 = BatchNormalization()(Con1)
Act1 = Activation("relu")(Bat1)

lstm1 = Bidirectional(LSTM(100, return_sequences=True))(Act1)
att1 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm1)

Con2 = Conv1D(300, 3,padding="same", kernel_initializer='he_normal')(att1)
Bat2 = BatchNormalization()(Con2)
Act2 = Activation("relu")(Bat2)

Output_layer = Conv1D(1, 3, activation='sigmoid',padding="same", kernel_initializer='he_normal')(Act2)


model = Model(inputs=Input_layer, outputs=Output_layer)

model.summary()

#Save Model Diagram
from keras.utils import plot_model
plot_model(model, to_file='model_Con1d_with bi_LSTM_self_att.png')

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

all_x_val = list()
all_y_val = list()

#Load Data Val
for num_train_file in range(14001,15001):
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data_2\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data_2\output\output_A_"+str(num_train_file)+".npy")

    for x in x_train: 
        all_x_val.append(x[:,:])
    for y in y_train:
        all_y_val.append(np.reshape(y,(y.shape[0],1)))

all_x_val = np.asarray(all_x_val)
all_y_val = np.asarray(all_y_val)

#Train Model
BATCH_SIZE = 100
EPOCHS = 10

checkpoint = ModelCheckpoint('train_model_v3_lstm\model_conv1d_Lstm.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')

model.load_weights('train_model_v3_lstm\model_conv1d_Lstm.h5')

#print(len(all_y_train))

all_x_train = list()
all_y_train = list()

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

            model.save('train_model_v3_lstm\model_conv1d_Lstm_final.h5')

            all_x_train = list()
            all_y_train = list()

#215025,18458
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('train_model_v3_lstm\his\model_conv1d_Lstm.png')
plt.clf()




