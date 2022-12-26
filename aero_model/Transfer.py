import os.path
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.callbacks import Callback
from keras.layers import LSTM, Dense
# 确定随机种子，确保每次结果一样
from opt_einsum.backends import torch
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
absolute_path = os.path.split(sys.argv[0])[0]


class TransferModel():

    def load_data(self, path=None):
        path = path or f'{absolute_path}/dat/有粘.csv'
        df_datas = None
        if path.endswith('.csv'):
            df_datas = pd.read_csv(path)
            # self.column_headers = list(self.df_datas.columns.values)
        elif path.endswith('.xlsx' or 'xls'):
            df_datas = pd.read_excel(path, engine='openpyxl')
            # self.column_headers = list(self.df_datas.columns.values)
        else:
            print("文件类型错误，请插入csv或xlsx类型的文件")
        return df_datas

    def data_init(self, df=None, x_name=None, y_name=None, randomCut=False, ration=None):
        self.x_name = x_name or ['钝度', '来流马赫数', '第1壁面切向角', '第2壁面切向角', '第3壁面切向角', '第4壁面切向角', '喉道高度']
        self.y_name = y_name or ['喉道马赫数', '总压恢复系数']

        self.x = df[self.x_name].to_numpy()
        self.y = df[self.y_name].to_numpy()

        # 不切分：
        self.x_train_ori = self.x
        self.y_train_ori = self.y

        # # 均匀切分
        # if not randomCut:
        #     ration = ration or 0.5
        #     self.x_train_ori = self.x[0:self.x.shape[0]:2]
        #     self.y_train_ori = self.y[0:self.y.shape[0]:2]
        #
        #     self.x_test_ori = self.x[1:self.x.shape[0]:2]
        #     self.y_test_ori = self.y[1:self.y.shape[0]:2]
        #
        # # 随机切分
        # else:
        #     ration = ration or 0.5
        #     self.x_train_ori, self.x_test_ori, self.y_train_ori, self.y_test_ori = train_test_split(self.x, self.y,
        #                                                                                             train_size=ration,
        #                                                                                             random_state=42)
        # 验证集
        self.x_valid = self.x_train_ori[3:self.x_train_ori.shape[0]:5]
        x_train1 = self.x_train_ori[0:self.x_train_ori.shape[0]:5]
        x_train2 = self.x_train_ori[1:self.x_train_ori.shape[0]:5]
        x_train3 = self.x_train_ori[2:self.x_train_ori.shape[0]:5]
        x_train4 = self.x_train_ori[4:self.x_train_ori.shape[0]:5]
        self.x_train_ori = np.concatenate((x_train1, x_train2, x_train3, x_train4))

        self.y_valid = self.y_train_ori[3:self.y_train_ori.shape[0]:5]
        y_train1 = self.y_train_ori[0:self.y_train_ori.shape[0]:5]
        y_train2 = self.y_train_ori[1:self.y_train_ori.shape[0]:5]
        y_train3 = self.y_train_ori[2:self.y_train_ori.shape[0]:5]
        y_train4 = self.y_train_ori[4:self.y_train_ori.shape[0]:5]
        self.y_train_ori = np.concatenate((y_train1, y_train2, y_train3, y_train4))

        # #  训练数据排序
        # all_data = np.hstack((self.x_train_ori, self.y_train_ori))  # 合并数据集
        # sorted_indices = np.argsort(all_data[:, 0])  # 排序规则
        # all_data = all_data[sorted_indices]  # 排序
        # self.x_train_ori, self.y_train_ori = np.array_split(all_data, 2, axis=1)  # 水平分割
        #
        # self.x_train_ori = all_data[:, :self.x.shape[1]]
        # self.y_train_ori = all_data[:, self.x.shape[1]:]
        #
        # #  测试数据排序
        # all_data = np.hstack((self.x_train_ori, self.y_train_ori))  # 合并数据集
        # sorted_indices = np.argsort(all_data[:, 0])  # 排序规则
        # all_data = all_data[sorted_indices]  # 排序
        # self.x_train_ori, self.y_train_ori = np.array_split(all_data, 2, axis=1)  # 水平分割
        #
        # self.x_train_ori = all_data[:, :self.x.shape[1]]
        # self.y_train_ori = all_data[:, self.x.shape[1]:]

        if self.trained_model_name == None:
            self.x_scaler = StandardScaler().fit(self.x_train_ori)
            self.y_scaler = StandardScaler().fit(self.y_train_ori)
        else:
            self.x_scaler = self.trained_model.x_scaler
            self.y_scaler = self.trained_model.y_scaler

        self.x_train = self.x_scaler.transform(self.x_train_ori)
        self.y_train = self.y_scaler.transform(self.y_train_ori)

        self.x_valid = self.x_scaler.transform(self.x_valid)
        self.y_valid = self.y_scaler.transform(self.y_valid)

        # self.x_test = self.x_scaler.transform(self.x_test_ori)
        # self.y_test = self.y_scaler.transform(self.y_test_ori)

        self.x_shape = self.x_train.shape
        self.y_shape = self.y_train.shape

        self.x_train_st = self.reshape_input(self.x_train)
        # self.x_test_st = self.reshape_input(self.x_test)
        self.x_valid_st = self.reshape_input(self.x_valid)

    def reshape_input(self, x) -> None:
        x_st = x.reshape((x.shape[0], 1, x.shape[1]))
        return x_st

    def model_bulid(self):
        self.model = Sequential()
        self.model.add(
            LSTM(64, input_shape=(self.x_train_st.shape[1], self.x_train_st.shape[2]), activation='relu', name="lstm"))
        self.model.add(Dense(13, activation='relu', name="layer2"))
        self.model.add(Dense(13, activation='relu', name="layer3"))
        # self.model.add(Dense(32, activation='relu', name="layer4"))
        self.model.add(Dense(self.y_shape[1], input_dim=13, name="out"))

    def model_train(self, epochs=None, epoch_callback=None):

        # 模型加载
        if self.trained_model_name != None:
            path = f'{absolute_path}/dat/model/{self.trained_model_name}/{self.trained_model_name}.h5'
            self.model.load_weights(path, by_name=True)
        epochs = epochs or 1000
        for layer in self.model.layers[2:3]:
            layer.trainable = False
        print(self.model.summary())
        self.model.compile(optimizer=tf.optimizers.Nadam(), loss=mmd_loss)
        history = LossHistory(epoch_callback)
        self.model.fit(self.x_train_st, self.y_train, epochs=epochs,
                       validation_data=(self.x_valid_st, self.y_valid),
                       callbacks=[history],
                       verbose=0,
                       batch_size=8)  # fit开始训练

    def fine_tune(self, epochs=None, epoch_callback=None):
        epochs = epochs or 1000
        # 解冻
        for layer in self.model.layers:
            layer.trainable = True
        self.model.compile(optimizer=tf.optimizers.Nadam(1e-3), loss="MSE")
        history = LossHistory(epoch_callback, flag=False, epochs=epochs)
        self.model.fit(self.x_train_st, self.y_train, epochs=epochs,
                       validation_data=(self.x_valid_st, self.y_valid),
                       callbacks=[history],
                       verbose=0,
                       batch_size=8)  # fit开始训练

    def model_load_job(self, Trained_model_name=None) -> None:
        # self.model.load_weights("PreTrain_model.h5", by_name=True)
        self.trained_model_name = Trained_model_name
        if self.trained_model_name == None:
            pass
        else:
            self.trained_model = self.loadModel(self.trained_model_name)

    def model_save(self, model_name=None) -> None:
        self.model_name = model_name or "预训练模型"

        if not os.path.exists(f"{absolute_path}/dat/model/{model_name}/"):
            os.makedirs(f"{absolute_path}/dat/model/{model_name}/")
        # 模型参数保存
        model = self.model
        self.path = f"{absolute_path}/dat/model/{self.model_name}/{self.model_name}.h5"
        model.save(f"{absolute_path}/dat/model/{self.model_name}/{self.model_name}.h5")
        # 其他参数保存

        pkl_path = f"{absolute_path}/dat/model/{self.model_name}/{self.model_name}.pkl"
        f = open(pkl_path, 'wb')
        self.model = None
        self.trained_model = None
        pickle.dump(self, f)
        f.close()
        self.model = model

    def predict(self, x):
        # 归一化
        x = self.x_scaler.transform(x)
        # reshape
        x_st = self.reshape_input(x)
        y_pre = self.model.predict(x_st)
        y = self.y_scaler.inverse_transform(y_pre)
        return y

    @staticmethod
    def loadModel(model_name):
        model_name = model_name
        path = f"{absolute_path}/dat/model/{model_name}/{model_name}.pkl"
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        from keras.models import load_model
        obj.model = load_model(f'{absolute_path}/dat/model/{model_name}/{model_name}.h5')
        return obj


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_s = tf.shape(source)[0]
    n_s = 64 if n_s is None else n_s
    n_t = tf.shape(target)[0]
    n_t = 64 if n_t is None else n_t
    n_samples = n_s + n_t
    total = tf.concat([source, target], axis=0)  # [None,n]
    total0 = tf.expand_dims(total, axis=0)  # [1,b,n]
    total1 = tf.expand_dims(total, axis=1)  # [b,1,n]
    L2_distance = tf.reduce_sum(((total0 - total1) ** 2),
                                axis=2)  # [b,b,n]=>[b,b]                                 #   [None,None,n]=>[128,128,1]
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / float(n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)   #[b,b]


def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    n_s = tf.shape(source)[0]
    n_s = 64 if n_s is None else n_s
    n_t = tf.shape(target)[0]
    n_t = 64 if n_t is None else n_t
    XX = tf.reduce_sum(kernels[:n_s, :n_s]) / float(n_s ** 2)
    YY = tf.reduce_sum(kernels[-n_t:, -n_t:]) / float(n_t ** 2)
    XY = tf.reduce_sum(kernels[:n_s, -n_t:]) / float(n_s * n_t)
    YX = tf.reduce_sum(kernels[-n_t:, :n_s]) / float(n_s * n_t)
    loss = XX + YY - XY - YX
    return loss


def mmd_loss(y_true, y_pred):
    mmd = MMD(y_pred, y_true)
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return K.mean(math_ops.squared_difference(y_pred, y_true) + mmd, axis=-1)


class LossHistory(Callback):
    def __init__(self, status_callback, flag=True, epochs=0):
        super().__init__()
        self.flag = flag
        self.epochs = epochs
        self.status_callback = status_callback

    def on_train_begin(self, logs={}):
        pass
        # self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        print("epoc: {} , {}".format(epoch, logs))
        if self.flag:
            if self.status_callback:
                self.status_callback(epoch, logs['loss'], logs['val_loss'])
        else:
            if self.status_callback:
                self.status_callback(epoch + self.epochs, logs['loss'], logs['val_loss'])


if __name__ == '__main__':
    model = TransferModel()
    df = model.load_data()
    model.model_load_job('222')
    model.data_init(df=df)


    def status_callback(inx, loss, val_loss):
        print("epoc{}: {} , {}".format(inx, loss, val_loss))


    model.model_bulid()
    model.model_train(10, status_callback)
    model.model_save('333')
