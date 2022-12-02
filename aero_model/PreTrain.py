import os.path
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.python.keras.optimizers import
from keras import Sequential
from keras.callbacks import Callback
from keras.layers import LSTM, Dense
from keras.optimizers import Nadam
# 确定随机种子，确保每次结果一样
from sklearn.preprocessing import StandardScaler

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

absolute_path = os.path.split(sys.argv[0])[0]


class PreTrainModel():

    def load_data(self, path=None):
        path = path or f'{absolute_path}/dat/无粘.csv'
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

    def data_init(self, df=None, x_name=None, y_name=None, model_name=None, randomCut=False, ration=None):
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

        # 归一化
        if model_name == None:
            self.x_scaler = StandardScaler().fit(self.x_train_ori)
            self.y_scaler = StandardScaler().fit(self.y_train_ori)
        else:
            x_scaler_file = f"{absolute_path}/dat/model/{model_name}/x_scaler_file.sav"
            y_scaler_file = f"{absolute_path}/dat/model/{model_name}/y_scaler_file.sav"
            self.x_scaler = pickle.load(open(x_scaler_file, 'rb'))
            self.y_scaler = pickle.load(open(y_scaler_file, 'rb'))

        self.x = self.x_scaler.transform(self.x)

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
        self.model.add(Dense(units=self.y_shape[1], input_dim=13, name="output"))

    def model_train(self, epochs=None, epoch_callback=None):
        epochs = epochs or 1000
        print(self.model.summary())
        self.model.compile(optimizer=Nadam(), loss="MSE")
        history = LossHistory(epoch_callback)
        self.model.fit(self.x_train_st, self.y_train, epochs=epochs,
                       validation_data=(self.x_valid_st, self.y_valid),
                       callbacks=[history],
                       verbose=0,
                       batch_size=8)  # fit开始训练

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


class LossHistory(Callback):
    def __init__(self, status_callback):
        super().__init__()
        self.status_callback = status_callback

    def on_train_begin(self, logs={}):
        pass
        # self.losses = [] 

    def on_epoch_end(self, epoch, logs={}):
        print("epoc: {} , {}".format(epoch, logs))
        if self.status_callback:
            self.status_callback(epoch, logs['loss'], logs['val_loss'])


if __name__ == '__main__':
    preTrainModel = PreTrainModel()
    df = preTrainModel.load_data()
    preTrainModel.data_init(df=df)


    def status_callback(inx, loss, val_loss):
        print("epoc{}: {} , {}".format(inx, loss, val_loss))


    preTrainModel.model_bulid()
    preTrainModel.model_train(10, status_callback)
    preTrainModel.model_save('222')
