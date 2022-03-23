from keras.layers import Input, Dense, LSTM, merge, Conv1D, Dropout, Bidirectional, Multiply
from keras import backend as K
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from model_train import TPR, FPR, AUC
from sklearn.metrics import roc_curve, auc

INPUT_DIMS = 20
lstm_units = 128
filters = 164


def attention_model():
    inputs = Input(shape=(1, INPUT_DIMS))
    # 过滤器数目  卷积核尺寸   激活函数
    x = Conv1D(filters=filters, kernel_size=1, activation='tanh')(inputs)
    x = Dropout(0.3)(x)

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block2(lstm_out)

    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
    attention_mul = Flatten()(attention_mul)
    output = Dense(2, activation='softmax')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model


def attention_3d_block2(inputs, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # 转置
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

x_train = train.iloc[:, :20]
y_train = train.iloc[:, [20]]

x_test = test.iloc[:, :20]
y_test = test.iloc[:, [20]]
x_train = np.array(x_train)
x_train = x_train.reshape(51427, 1, 20)

y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.reshape(12857, 1, 20)

m = attention_model()

m.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
TensorBoard = TensorBoard(log_dir='./log/train_log', histogram_feq=1, write_graph=True, update_freq='epoch')
loss = m.fit(x_train, y_train, epochs=50, validation_split=0.3, callbacks=[TensorBoard])
m.save('./model/new_cnn.h5')
pre = m.predict(x_test)

y_pre = []
for i in range(len(pre)):
    temp = list(pre[i])
    temp = temp.index(max(temp))
    y_pre.append(temp)

precision = precision_score(y_test, y_pre)
recall = recall_score(y_test, y_pre)
f1 = f1_score(y_test, y_pre)
print('精确率为：%0.5f' % precision)
print('召回率：%0.5f' % recall)
print('F1均值为：%0.5f' % f1)


