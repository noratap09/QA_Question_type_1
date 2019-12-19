from keras.models import Model, Sequential
from keras.layers import  Input, Dense,Conv1D, MaxPooling1D ,GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math

#Create Model
question_len = 100
input_text_len = 100

Input_layer = Input(shape=(question_len,input_text_len,))
Con1 = Conv1D(100, 5, activation='relu',padding="same")(Input_layer)
Con2 = Conv1D(100, 3, activation='relu',padding="same")(Con1)
Max1 = MaxPooling1D(3)(Con2)
Con3 = Conv1D(160, 3, activation='relu',padding="same")(Max1)
Con4 = Conv1D(160, 5, activation='relu',padding="same")(Con3)
Glo = GlobalAveragePooling1D()(Con4)
output_layer = Dense(1, activation='sigmoid')(Glo)

model = Model(inputs=Input_layer, outputs=output_layer)

model.summary()

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

def show_result(model,input_data,gt):
        for i,data_gt in enumerate(gt,start=0):
            if(data_gt==1.0):
                pred = model.predict(np.asarray([input_data[i]]))
                print("GT : ",data_gt)
                print("Pred : ",pred)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#Load Data
for count in range(0,10):
    for num_train_file in range(1,2):
        print("Load Data : ",num_train_file)
        x_train = np.load("train_data\input\input_A_"+str(num_train_file)+".npy")
        y_train = np.load("train_data\output\output_A_"+str(num_train_file)+".npy")

        x_train = x_train[:,:,:,0]
        y_train = y_train[:,0]

        BATCH_SIZE = 10
        EPOCHS = 5

        checkpoint = ModelCheckpoint('train_model_v2\model_v2.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='max')

        history = model.fit(x_train,
                            y_train,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            callbacks=[checkpoint],
                            validation_split=0.1)
        
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.savefig('train_model_v2\his\model_v2_epoch_'+str(count)+'_num_'+str(num_train_file)+'.png')
        plt.clf()

        #show result after train
        print("--------------SHOW RESULT---------------")
        show_result(model,x_train,y_train)

model.save('train_model_v2\model_v2_final.h5')
