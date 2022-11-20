import numpy as np
import pandas as pd
import sys,os
import keras.backend as K 
from sklearn import preprocessing
from keras.layers import Dense, LSTM, Bidirectional
from keras.callbacks import LearningRateScheduler
from tensorflow import keras
from keras import Sequential
from keras.callbacks import Callback
absolute_path = os.path.split(sys.argv[0])[0] 

class SaleModel():
    __default_path = f'{absolute_path}/dat/lstm_datas.xlsx'
    meta = {
           "total": {"x_names": ['卷烟总量'], "y_names": ['卷烟总量']},
           "price_13_18": {"x_names": ['13-18价类'], "y_names": ['13-18价类'] },
           "price_18_23": {"x_names": ['18-23价类'], "y_names": ['18-23价类']},
           "price_23_30": {"x_names": ['23-30价类'], "y_names": ['23-30价类']},
           "size_middle": {"x_names": ['中支烟',], "y_names": ['中支烟']},
            "size_thin": {"x_names": ['细支烟'], "y_names": ['细支烟']},
            "band_zhenglong": {"x_names": ['真龙品牌'], "y_names": ['真龙品牌']},
            "band_jiaozi": {"x_names": ['娇子品牌'], "y_names": ['娇子品牌']},
        }
    
    def __init__(self, p_type='total'):        
        
        self.p_type = p_type
        self.data = None
        self.dataX = None
        self.dataY = None
        self.predictY_on_trainX=None
        self.predictY_on_testX =None
        self.predict_12 =None 
        self.scaler = None
        self.bilstm = None
        self.windowsize = 6
        self.epoch_max = 150  
        self.loss = None
        self.rmse = None

    def load_data(self, path =None):  
        path = path or SaleModel.__default_path
        datasource = pd.read_excel(path)
        var_name = SaleModel.meta[self.p_type]['y_names']
        oridata = datasource[var_name].values
        self.oridata = oridata.astype('float32')
        print(f'p_type:{self.p_type},y_colums:{var_name},y0:{self.oridata[0]}')
        # 标准化数据
        self.scaler = preprocessing.StandardScaler().fit(self.oridata)
        self.data = self.scaler.transform(self.oridata)

        def time_convert(para):  
            delta = pd.Timedelta(str(para)+'days') 
            time = pd.to_datetime('1899-12-30')+delta 
            return time.strftime('%Y-%m') 
        self.time_line = datasource['时间'].apply(time_convert).values #2011-1 ~ 2021-12
        self.time_line_validate = np.array([f"2021-{inx:0>2}" for inx in range(1,13)],dtype=np.dtype('O')) # 2021-1 ~ 2021-12
        self.time_forword_12 =np.array( [f"2022-{inx:0>2}" for inx in range(1,13)],dtype=np.dtype('O')) # 2022-1 ~ 2022-12
        self.time_line_predict = np.concatenate((self.time_line_validate, self.time_forword_12), axis=0) 
        

    # 处理数据为lstm所需格式 (n_sampele, windowsize)
    def data_precesing(self):
        data_len = len(self.data)
        listX, listY = [], []
         
        for i in range( data_len - self.windowsize):
            a = self.data[i:(i + self.windowsize), 0]
            listX.append(a)
            listY.append(self.data[i + self.windowsize, 0])
         
        self.dataX = np.array(listX)
        self.dataY = np.array(listY)

    # 划分训练集和测试集
    def data_partition(self, train_ratio=0.91):
        train_size = int(len(self.dataY) * train_ratio)
        self.train_size = train_size
        self.test_size = len(self.dataY) - train_size
        trainX, testX = self.dataX[0:train_size,:], self.dataX[train_size:len(self.dataY), :]
        self.trainY, self.testY = self.dataY[0:train_size], self.dataY[train_size:len(self.dataY)]

        # 将特征维度补充到数据格式中
        self.trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        self.testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    def scheduler(self, epoch,lr):
        # 每隔125个epoch，学习率减小为原来的2/10
        if epoch % 125 == 0 and epoch != 0:
            lr = lr*0.2 
            print("lr from{} change to{}".format(K.get_value(self.bilstm.optimizer.lr), format(lr))) 
        return lr

    def model_build(self):
        input_dim = 5
        inputshape = (input_dim, self.windowsize, 1)
        # 隐藏层block的个数
        self.bilstm = Sequential()
        self.bilstm.add(Bidirectional(LSTM(
                    units=200,  
                    batch_input_shape=inputshape,  # 输入维度
                    stateful=False,  # 保持状态
                ), merge_mode='concat'))
        self.bilstm.add(Dense(1))

        adam = keras.optimizers.Adam(learning_rate=0.005, clipnorm=1.0)  # 梯度缩放阈值clipnorm=1.0
        self.bilstm.compile(optimizer=adam, loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])

    def train(self, status_callback):
        self.data_precesing()
        self.data_partition()
        self.model_build()
        reduce_lr = LearningRateScheduler(self.scheduler)
        history = LossHistory(status_callback)
        history_back=self.bilstm.fit(self.trainX, self.trainY, validation_data=(self.testX, self.testY), 
                        batch_size=3, epochs=self.epoch_max, 
                        callbacks=[reduce_lr, history],
                        verbose=0, validation_freq=10, shuffle=False)
        self.loss = history_back.history['loss']
        self.rmse = history_back.history['root_mean_squared_error']

    def validate(self):# 2021年1月-2021年12月
        predictY_on_testX = self.bilstm.predict(self.testX)
        predictY_on_testX = self.scaler.inverse_transform(predictY_on_testX)
        ytrue = self.scaler.inverse_transform(self.testY[-self.test_size:].reshape(-1, 1) )
        ytrue = ytrue.reshape(ytrue.shape[0])
        predictY_on_testX = predictY_on_testX.reshape(predictY_on_testX.shape[0])
        self.error = ytrue - predictY_on_testX
        self.ytrue_on_testX = ytrue
        self.predictY_on_testX = predictY_on_testX

    def predict_onTrain(self):
        predictY_on_trainX = self.bilstm.predict(self.trainX)
        self.predictY_on_trainX = self.scaler.inverse_transform(np.array(predictY_on_trainX).reshape(-1,1)).reshape(-1)
        pass

    def predict(self):# 前1~6个月->预测第7个月 向前预测12个月的数据
        x_pre=self.data[-6:,0]
        predictY_12 = []
        for i in range(12):
            x_input = np.reshape(x_pre, (1, 6, 1))
            predictY_i = self.bilstm.predict(x_input)
            predictY_12.append(predictY_i[0,0])
            x_pre = np.append(x_pre[1:],predictY_i)
        self.predictY_12 = self.scaler.inverse_transform(np.array(predictY_12).reshape(-1,1)).reshape(-1)

    def saveModel(self):
        import pickle
        # file_id=str(uuid.uuid1()) 
        file_id=f'lstm_{self.p_type}'

        bilstm = self.bilstm
        bilstm.save(f'{absolute_path}/dat/files_lstm/{file_id}.h5') 

        if not os.path.exists(f"{absolute_path}/dat/files_lstm/"):
            os.makedirs(f"{absolute_path}/dat/files_lstm/")
        path = f"{absolute_path}/dat/files_lstm/{file_id}.pkl" 
        f = open(path, 'wb')
        self.bilstm =None
        pickle.dump(self, f)
        f.close()
        self.bilstm = bilstm
        return file_id 
    
    @staticmethod
    def loadModel(file_id) -> 'SaleModel':
        import pickle
        path = f"{absolute_path}/dat/files_lstm/{file_id}.pkl"
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        from keras.models import load_model
        obj.bilstm = load_model(f'{absolute_path}/dat/files_lstm/{file_id}.h5')
        return obj
    # def saveModel(self):
    #     import uuid
    #     import json
    #     # oridata, windowsize, y_pre, y_true, y_err, 'loss','RMSE' 
    #     #file_id=str(uuid.uuid1())
    #     file_id=f'lstm_{self.p_type}'
    #     show_data_dict = {'oridata':self.oridata.tolist(), 'windowsize':self.windowsize, 
    #                 'ypre': self.ypre.tolist(), 'ytrue': self.ytrue.tolist(), 'predict_12': self.predict_12.tolist(),
    #                 'yerr':self.error.tolist(), 'loss': self.loss,'rmse':self.rmse, 
    #                 'time_line':self.time_line.tolist(),'time_line_validate':self.time_line_validate.tolist(),'time_line_predict': self.time_line_predict.tolist() 
    #                 } 
    #     json_str = json.dumps(show_data_dict)
    #     if not os.path.exists(f"{absolute_path}/dat/files_lstm/"):
    #         os.makedirs(f"{absolute_path}/dat/files_lstm/")
    #     fo = open(f"{absolute_path}/dat/files_lstm/{file_id}.json", "w") 
    #     fo.write(json_str) 
    #     fo.close() 

    #     self.bilstm.save(f'{absolute_path}/dat/files_lstm/{file_id}.h5') 
    #     return file_id

    # def delModel(self, file_id):
    #     mod_file_path = f'{absolute_path}/dat/files_lstm/{file_id}.h5'
    #     dat_file_path = f"{absolute_path}/dat/files_lstm/{file_id}.json"
    #     if os.path.exists(mod_file_path): 
    #         os.remove(mod_file_path) 
    #     if os.path.exists(dat_file_path): 
    #         os.remove(dat_file_path) 
    #     pass   

    # def loadModel(self):
    #     from keras.models import load_model
    #     import json
    #     self.bilstm = load_model(f'{absolute_path}/dat/files_lstm/{self.file_id}.h5')
    #     fo = open(f"{absolute_path}/dat/files_lstm/{self.file_id}.json", "r") 
    #     dat_json = fo.read()
    #     show_data_dict = json.loads(dat_json)
    #     fo.close()
    #     self.oridata = np.array(show_data_dict['oridata'])
    #     self.windowsize = np.array(show_data_dict['windowsize'])
    #     self.ypre = np.array(show_data_dict['ypre'])
    #     self.ytrue = np.array(show_data_dict['ytrue']) 
    #     self.error =np.array( show_data_dict['yerr'])
    #     self.loss =  np.array(show_data_dict['loss'])
    #     self.rmse =  np.array(show_data_dict['rmse']) 
    #     self.predict_12 = np.array(show_data_dict['predict_12']) 
    #     self.time_line= np.array(show_data_dict['time_line']) 
    #     self.time_line_validate= np.array(show_data_dict['time_line_validate']) 
    #     self.time_line_predict = np.array(show_data_dict['time_line_predict']) 

class LossHistory(Callback):
    def __init__(self,status_callback):
        super().__init__()
        self.status_callback = status_callback
    def on_train_begin(self, logs={}):
        self.losses = [] 
    def on_epoch_end(self, epoch, logs={}):
        #print("epoc: {} , {}".format(epoch, logs))
        self.status_callback(epoch, logs['loss'], logs['root_mean_squared_error'])


if __name__ == '__main__':
    pass
    # print(SaleModel.meta['total']['y_names'][0])
#     losses =[]
#     rmses =[]
    def status_callback(inx, loss, RMSE):
        # losses.append(loss)
        # rmses.append(RMSE)
        print("epoc{}: {} , {}".format(inx, loss, RMSE))
   #path = 'D:/workspace/Works/Jupyter_Works/zwd/AlCigaarette/template/template_lstm.xlsx'
    sale_model  = SaleModel() 
    sale_model.load_data()
    sale_model.train(status_callback)
    sale_model.validate()
    sale_model.predict()
    file_id = sale_model.saveModel() 
#加载已训练的模型
    # model = SaleModel.loadModel('lstm_total')
    # print(model.predict_12)
    # model.predict()
    # print(model.predict_12)
  