import sys
from PyQt5.QtCore import QThread, pyqtSignal, QTimer 
from PyQt5.QtWidgets import QMainWindow, QApplication, QGridLayout , QWidget, QProgressBar, QLabel
from BiLSTM import SaleModel
import numpy as np
from figures import Figure_Line  

class LSTMwidget(QMainWindow):
    progress_signal = pyqtSignal(str,str,int)
    model_train_status_signal = pyqtSignal(int,float,float) # index, loss, rmse
    model_train_finish_signal = pyqtSignal(str)
    def __init__(self, name=None, path = None, file_id = None, p_type='total', callback=None):
        super(LSTMwidget, self).__init__()
        self.p_type = p_type
        self.name =name or f"LSTM{SaleModel.meta[p_type]['y_names'][0]}"
        self.path = path
        self.file_id = file_id
        self.setWindowTitle(f'{self.name}曲线')
        self.resize(1200, 900)  
        self.initUI()
        self.progress_signal.connect(self.update_progress_status_ui)
        self.model_train_status_signal.connect(self.update_train_status)
        self.model_train_finish_signal.connect(self.train_finish)
                
        #self.model_build_job()
        run_able =self.model_build_job
        if file_id:
            run_able = self.model_load_job
        self.worker = TrainWorker(run_able)
        self.worker.start() 
        self.callback=callback

    def initUI(self): 
        widget = QWidget()
        self.fig_layout = QGridLayout(self)
        widget.setLayout(self.fig_layout)
        self.setCentralWidget(widget) 
        self.createStatusBar()

        self.lossFigure = Figure_Line(title=f'{self.name}-LSTM模型收敛曲线', xlabel='epoch', ylabel='loss&RMSE')
        self.fig_layout.addWidget(self.lossFigure,1,1)

        self.dataSetFigure = Figure_Line(title=f'{self.name}-数据集', xlabel='时间（月）', ylabel=f'{self.name}')
        self.fig_layout.addWidget(self.dataSetFigure,2,1)

        self.predictFigure = Figure_Line(title=f'{self.name}-数据预测', xlabel='时间（月）', ylabel=f'{self.name}')
        self.fig_layout.addWidget(self.predictFigure,1,2)

        self.errFigure = Figure_Line(title=f'{self.name}-预测误差', xlabel='时间（月）', ylabel='误差')
        self.fig_layout.addWidget(self.errFigure,2,2)

    def createStatusBar(self):
        self.statusBar().showMessage('准备中...')
        self.progressBar = QProgressBar()
        self.label = QLabel()
        self.label2 = QLabel()
        self.statusBar().addPermanentWidget(self.label)
        self.statusBar().addPermanentWidget(self.label2)
        self.statusBar().addPermanentWidget(self.progressBar) 
        self.progressBar.setGeometry(0, 0, 100, 3)
        self.progressBar.setRange(0, 100) # 设置进度条的范围
        self.updateProgress("正在处理",'初始化',5)

    def update_progress_status_ui(self,msg1, msg2, stepValue):
        self.label.setText(msg1)
        self.label2.setText(msg2)
        self.progressBar.setValue(stepValue)
        if(stepValue==100):
            timer = QTimer(self)
            timer.timeout.connect(lambda:self.statusBar().hide()) 
            timer.start(1000)
    
    def updateProgress(self,msg1, msg2, stepValue):
        self.progress_signal.emit(msg1,msg2,stepValue)

       

    def model_build_job(self): 
        self.updateProgress("LSTM模型构建",'模型初始化',10) 
        self.model = SaleModel(p_type=self.p_type) 
        self.updateProgress("LSTM模型构建",'加载训练数据',20) 
        self.model.load_data()
        self.create_train_line()
        self.updateProgress("LSTM模型构建",'模型训练',30)
        status_callback = self.add_train_status_data 
        self.model.train(status_callback)
        self.updateProgress("LSTM模型构建",'模型验证',80)
        self.model.validate()
        self.updateProgress("LSTM模型构建",'模型预测',80)
        self.model.predict_onTrain()
        self.model.predict()
        self.updateProgress("LSTM模型构建",'模型保存',90)
        self.saveDat()
        self.model_train_finish_signal.emit('finish') 

    def model_load_job(self): 
        self.updateProgress("LSTM模型加载",'数据读取',20) 
        try:
            self.model = SaleModel.loadModel(self.file_id)
            self.updateProgress("LSTM模型加载",'加载成功',40) 
            self.create_train_line(loss=self.model.loss, rmse=self.model.rmse) 
            self.updateProgress("LSTM模型加载",'完成',80) 
            self.train_finish('finish')
        except Exception as e:
            self.updateProgress("LSTM模型加载",f'失败{str(e)}',40) 
        #self.model_train_finish_signal.emit('finish') 

    def saveDat(self):
        #'lngExtent':self.lngExtent,'latExtent':self.latExtent, 'cellSizeCoord':self.cellSizeCoord,'center':self.center,
        file_id = self.model.saveModel()
        dat ={'name':self.name,
              'type':'lstm',
              'file_id': file_id}
        data_json_file = dataOper.addInxFile(type1='sale_predict', name=self.name, dict_dat=dat) #type = 'brand'
        if self.callback :
            self.callback(self.name, data_json_file)
     

    #创建一个groupbox, 用来画训练过程的动态曲线
    def create_train_line(self, loss=[], rmse=[]):   # ['loss','RMSE']
        self.lossFigure.add_line('loss', np.array(range(len(loss))), np.array(loss)) 
        self.lossFigure.add_line('RMSE', np.array(range(len(rmse))), np.array(rmse))

    def add_train_status_data(self, inx, loss, RMSE):
        self.updateProgress("LSTM模型构建",f'模型训练{inx}/{self.model.epoch_max}',30+int(inx*50/self.model.epoch_max))
        self.model_train_status_signal.emit(inx, loss, RMSE) 

    def update_train_status(self, inx, loss, RMSE):
        self.lossFigure.add_data(inx,{'loss':loss, 'RMSE':RMSE}) 
    
    def train_finish(self, status):
        print(f"finish:{status}")
        X = np.arange(0, len(self.model.oridata), 1)
        self.plot_pre_dataset()  
        self.plot_predict_err()
        self.updateProgress("LSTM模型",'完成',100)
    
    def plot_pre_dataset(self): 
        model = self.model
        seq = np.concatenate((model.time_line, model.time_forword_12))
        self.dataSetFigure.set_xticks(range(0,len(seq),3),[seq[i] for i in range(0,len(seq),1) if i%3==0 ],rotation=60,fontsize=8)
        self.dataSetFigure.add_line("原始数据集输出", model.time_line, model.oridata,style={'ls':'','marker':'o'})
        self.dataSetFigure.add_line("训练集上预测", model.time_line[model.windowsize:-12], model.predictY_on_trainX,style={'ls':'--','marker':''})
        self.dataSetFigure.add_line("验证预测输出", model.time_line_predict, np.concatenate((model.predictY_on_testX,model.predictY_12)))
        

    def plot_predict_err(self):
        model = self.model
        self.predictFigure.add_line("测试数据实际值", model.time_line_validate, model.ytrue_on_testX, style={'ls':'','marker':'o'}) 
        self.predictFigure.add_line("测试数据预测值", model.time_line_validate, model.predictY_on_testX) 
        self.errFigure.add_line("预测误差（条）", model.time_line_validate, model.error)
        seq = model.time_line_validate
        self.predictFigure.set_xticks(range(0,len(seq),3),[seq[i] for i in range(0,len(seq),1) if i%3==0 ],rotation=10,fontsize=8)
        self.errFigure.set_xticks(range(0,len(seq),3),[seq[i] for i in range(0,len(seq),1) if i%3==0 ],rotation=10,fontsize=8)

class TrainWorker(QThread): 
    def __init__(self, job):
        super(TrainWorker,self).__init__()  
        self.job = job 
    def run(self):  
        self.job()

if __name__ == '__main__':
    app = QApplication(sys.argv) 
    p_type='size_thin'
     
    # def status_callback(inx, loss, RMSE):
    #     print("epoc{}: {} , {}".format(inx, loss, RMSE))
    #     mainMindow.add_data(inx, loss, RMSE) 
    #mainMindow.initModel(path=path) 
    #mainMindow.initModel() 
    # worker = TrainWorker(mainMindow.model_build_job)
    # worker = TrainWorker(mainMindow.model_load_job)
    # worker.start()
    # p_type='price_13_18'
    mainMindow = LSTMwidget(p_type=p_type)  
    # mainMindow = LSTMwidget(p_type=p_type, file_id=f"lstm_{p_type}") 
    mainMindow.show()
    sys.exit(app.exec_())


