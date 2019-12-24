from keras.models import Model, Sequential,load_model
from keras.layers import  Input, Dense,Conv1D, MaxPooling1D ,GlobalAveragePooling1D
from keras.callbacks import ModelCheckpoint, Callback
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import math

def show_result(model,num_train_file):    
    print("Load Data : ",num_train_file)
    x_train = np.load("train_data\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data\output\output_A_"+str(num_train_file)+".npy")

    x_train = x_train[:,:,:,0]
    y_train_class = y_train[:,0]
    y_train_reges =  y_train[:,1:3]

    pred , pred_rage = model.predict(np.asarray(x_train))
    ck = True
    for data_n,pred_data in enumerate(pred,start=0):
            result = 0.0
            if(pred_data[0]>0.5):
                result = 1.0
            if(result != y_train_class[data_n]):
                    print("False at",data_n)
                    print("GT : ",y_train_class[data_n])
                    print("Pred : ",pred_data[0])
                    ck = False
            else:
                print("GT Rang : ",y_train_reges[data_n])
                print("pred Rang : ",pred_rage[data_n])
    return ck

model = load_model("train_model_v2\model_v2_final.h5")

print("--------------SHOW RESULT---------------")
t_score = 0
f_score = 0
for num_train_file in range(1,1001):
    if(show_result(model,num_train_file)):
            print("True Q:",num_train_file)
            t_score = t_score + 1
    else:
            print("False Q:",num_train_file) 
            f_score = f_score + 1

print("Score T:",t_score,"/",(t_score+f_score))
print("Score F:",f_score,"/",(t_score+f_score))