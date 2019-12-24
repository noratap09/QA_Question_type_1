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
Con1 = Conv1D(100, 3, activation='relu',padding="same")(Input_layer)
Con2 = Conv1D(100, 3, activation='relu',padding="same")(Con1)
Max1 = MaxPooling1D(3)(Con2)
Con3 = Conv1D(160, 3, activation='relu',padding="same")(Max1)
Con4 = Conv1D(160, 3, activation='relu',padding="same")(Con3)
Glo = GlobalAveragePooling1D()(Con4)
h1_1 = Dense(100, activation='tanh')(Glo)
h1_2 = Dense(100, activation='tanh')(h1_1)
h1_3 = Dense(100, activation='tanh')(h1_2)
output_class = Dense(1, activation='sigmoid',name="out_class")(h1_3)

h2_1 = Dense(100, activation='selu')(Glo)
h2_2 = Dense(100, activation='selu')(h2_1)
h2_3 = Dense(100, activation='selu')(h2_2)
output_reges = Dense(2, activation='sigmoid',name="out_regs")(h2_3)

model = Model(inputs=Input_layer, outputs=[output_class,output_reges])

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

def show_result(model,num_train_file):    
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data\output\output_A_"+str(num_train_file)+".npy")

    x_train = x_train[:,:,:,0]
    y_train = y_train[:,0]

    pred = model.predict(np.asarray(x_train))
    ck = True
    for data_n,pred_data in enumerate(pred[0],start=0):
            result = 0.0
            if(pred_data[0]>0.5):
                result = 1.0
            if(result != y_train[data_n]):
                    print("False at",data_n)
                    print("GT : ",y_train[data_n])
                    print("Pred : ",pred_data[0])
                    ck = False
    return ck

#Load Data
all_x_train = list()
all_y_train_class = list()
all_y_train_reges = list()
#Load Data
for num_train_file in range(1,1001):
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data\output\output_A_"+str(num_train_file)+".npy")

    for x in x_train: 
        all_x_train.append(x[:,:,0])
    for y in y_train:
        all_y_train_class.append(y[0])
        all_y_train_reges.append(y[1:3])

all_x_train = np.asarray(all_x_train)
all_y_train_class = np.asarray(all_y_train_class)
all_y_train_reges = np.asarray(all_y_train_reges)

#loss function
losses = {
        "out_class" :  "binary_crossentropy",
        "out_regs" :  "mse",
}

model.compile(optimizer='adam',
              loss=losses,
              metrics=['accuracy'])

BATCH_SIZE = 10
EPOCHS = 1000

checkpoint = ModelCheckpoint('train_model_v2\model_v2.h5', verbose=1, monitor='out_regs_acc',save_best_only=True, mode='max')

history = model.fit(all_x_train,
                        [all_y_train_class,all_y_train_reges],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=[checkpoint],
                        validation_split=0)

plt.plot(history.history['out_class_acc'])
plt.plot(history.history['out_regs_acc'])
plt.savefig('train_model_v2\his\model_v2.png')
plt.clf()

#show result after train
print("--------------SHOW RESULT---------------")
for num_train_file in range(1,1001):
    t_score = 0
    f_score = 0
    if(show_result(model,num_train_file)):
            print("True Q:",num_train_file)
            t_score = t_score + 1
    else:
            print("False Q:",num_train_file) 
            f_score = f_score + 1

print("Score T:",t_score,"/",(t_score+f_score))
print("Score F:",f_score,"/",(t_score+f_score))
model.save('train_model_v2\model_v2_final.h5')
