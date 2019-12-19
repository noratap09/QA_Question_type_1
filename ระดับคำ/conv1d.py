from keras.models import Model, Sequential
from keras.layers import  Input,Dense,Conv1D, Reshape ,MaxPooling1D,GlobalAveragePooling1D,Dropout
import numpy as np
#Create Model
question_len = 100
input_text_len = 100

Input_layer = Input(shape=(question_len,input_text_len,))
Con1 = Conv1D(100, 2, activation='relu',padding="same")(Input_layer)
Con2 = Conv1D(100, 2, activation='relu',padding="same")(Con1)
Max1 = MaxPooling1D(3)(Con2)
Con3 = Conv1D(160, 2, activation='relu',padding="same")(Max1)
Con4 = Conv1D(160, 2, activation='relu',padding="same")(Con3)
Glo = GlobalAveragePooling1D()(Con4)
output_layer = Dense(2, activation='softmax')(Glo)

model =  Model(inputs=Input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
print(model.summary())
"""
for num_train_file in range(1,14):
    x_train = np.load("train_data\input\input_A_"+str(num_train_file)+".npy")
    y_train = np.load("train_data\output\output_A_"+str(num_train_file)+".npy")

train_input = [
               [[0,0,0],
                [0,0,0],
                [0,0,0],
                [1,1,1],
                [1,1,1]],
               [[1,1,1],
                [1,1,1],
                [0,0,0],
                [0,0,0],
                [0,0,0],]
              ]

train_output = [[1,0],[0,1]]

train_input = np.asarray(train_input)
train_output = np.asarray(train_output)

model.fit(train_input, train_output, epochs=50, batch_size=4)

test_input = [
               [[0,0,0],
                [0,0,0],
                [0,0,0],
                [0,0,0],
                [1,1,1]],
               [[0,0,0],
                [1,1,1],
                [0,0,0],
                [0,0,0],
                [0,0,0],]
              ]

test_output = [[1,0],[0,1]]

test_input = np.asarray(test_input)
test_output = np.asarray(test_output)

print(model.evaluate(test_input,test_output))

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 400
EPOCHS = 50

history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)
"""