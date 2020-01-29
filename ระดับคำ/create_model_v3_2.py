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
Con1 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Input_layer)
Bat1 = BatchNormalization()(Con1)
Act1 = Activation("relu")(Bat1)

lstm1 = Bidirectional(LSTM(20, return_sequences=True))(Act1)
att1 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm1)

Con2 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(att1)
Bat2 = BatchNormalization()(Con2)
Act2 = Activation("relu")(Bat2)

Down1 = MaxPooling1D(2)(Act2)
Drop1 = Dropout(0.1)(Down1)

Con3 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Drop1)
Bat3 = BatchNormalization()(Con3)
Act3 = Activation("relu")(Bat3)

lstm2 = Bidirectional(LSTM(20, return_sequences=True))(Act3)
att2 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm2)

Con4 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(att2)
Bat4 = BatchNormalization()(Con4)
Act4 = Activation("relu")(Bat4)

Down2 = MaxPooling1D(2)(Act4)
Drop2 = Dropout(0.1)(Down2)

Con5 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Drop2)
Bat5 = BatchNormalization()(Con5)
Act5 = Activation("relu")(Bat5)

lstm3 = Bidirectional(LSTM(20, return_sequences=True))(Act5)
att3 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm3)

Con6 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(att3)
Bat6 = BatchNormalization()(Con6)
Act6 = Activation("relu")(Bat6)

Down3 = MaxPooling1D(2)(Act6)
Drop3 = Dropout(0.1)(Down3)

Con7 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Drop3)
Bat7 = BatchNormalization()(Con7)
Act7 = Activation("relu")(Bat7)

lstm4 = Bidirectional(LSTM(20, return_sequences=True))(Act7)
att4 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm4)

Con8 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(att4)
Bat8 = BatchNormalization()(Con8)
Act8 = Activation("relu")(Bat8)

Down4 = MaxPooling1D(2)(Act8)
Drop4 = Dropout(0.1)(Down4)

#Middle Path
Con9 = Conv1D(256, 3,padding="same", kernel_initializer='he_normal')(Drop4)
Bat9 = BatchNormalization()(Con9)
Act9 = Activation("relu")(Bat9)

lstm5 = Bidirectional(LSTM(20, return_sequences=True))(Act9)
att5 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm5)

Con10 = Conv1D(256, 3,padding="same", kernel_initializer='he_normal')(att5)
Bat10 = BatchNormalization()(Con10)
Act10 = Activation("relu")(Bat10)

#expansive path
Up1 = UpSampling1D(2)(Act10)
Up1 = Concatenate()([Up1,Act8])
Drop5 = Dropout(0.1)(Up1)

Con11 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(Drop5)
Bat11 = BatchNormalization()(Con11)
Act11 = Activation("relu")(Bat11)

lstm6 = Bidirectional(LSTM(20, return_sequences=True))(Act11)
att6 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm6)

Con12 = Conv1D(128, 3,padding="same", kernel_initializer='he_normal')(att6)
Bat12 = BatchNormalization()(Con12)
Act12 = Activation("relu")(Bat12)

Up2 = UpSampling1D(2)(Act12)
Up2 = Concatenate()([Up2,Act6])
Drop6 = Dropout(0.1)(Up2)

Con13 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(Drop6)
Bat13 = BatchNormalization()(Con13)
Act13 = Activation("relu")(Bat13)

lstm7 = Bidirectional(LSTM(20, return_sequences=True))(Act13)
att7 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm7)

Con14 = Conv1D(64, 3,padding="same", kernel_initializer='he_normal')(att7)
Bat14 = BatchNormalization()(Con14)
Act14 = Activation("relu")(Bat14)

Up3 = UpSampling1D(2)(Act14)
Up3 = Concatenate()([Up3,Act4])
Drop7 = Dropout(0.1)(Up3)

Con15 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(Drop7)
Bat15 = BatchNormalization()(Con15)
Act15 = Activation("relu")(Bat15)

lstm8 = Bidirectional(LSTM(20, return_sequences=True))(Act15)
att8 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm8)

Con16 = Conv1D(32, 3,padding="same", kernel_initializer='he_normal')(att8)
Bat16 = BatchNormalization()(Con16)
Act16 = Activation("relu")(Bat16)

Up4 = UpSampling1D(2)(Act16)
Up4 = Concatenate()([Up4,Act2])
Drop8 = Dropout(0.1)(Up4)

Con17 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(Drop8)
Bat17 = BatchNormalization()(Con17)
Act17 = Activation("relu")(Bat17)

lstm9 = Bidirectional(LSTM(20, return_sequences=True))(Act17)
att9 = att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4)(lstm9)

Con18 = Conv1D(16, 3,padding="same", kernel_initializer='he_normal')(att9)
Bat18 = BatchNormalization()(Con18)
Act18 = Activation("relu")(Bat18)

Output_layer = Conv1D(1, 3, activation='sigmoid',padding="same", kernel_initializer='he_normal')(Act18)


model = Model(inputs=Input_layer, outputs=Output_layer)

model.summary()

#Save Model Diagram
from keras.utils import plot_model
plot_model(model, to_file='model_Unet_Con1d_with bi_LSTM_self_att.png')

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
BATCH_SIZE = 50
EPOCHS = 30

checkpoint = ModelCheckpoint('train_model_v3_lstm\model_Unet_Lstm.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')

#model = load_model('train_model_v3\model_Unet_Lstm_final.h5')

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

            model.save('train_model_v3_lstm\model_Unet_Lstm_final.h5')

            all_x_train = list()
            all_y_train = list()

#215025,18458
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('train_model_v3_lstm\his\model_v3_epoch_final.png')
plt.clf()




