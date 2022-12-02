import math

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from DataWidget import DataCombineModel
from aero_model.PreTrain import *
from aero_model.TrainedModel import TrainedModel
from aero_model.Transfer import *
from data_oper import *
from figures import Figure_Line


class ModelingDialog(QDialog):
    model_train_status_signal = pyqtSignal(int, float, float)

    def __init__(self, finish_callback=None):
        super(ModelingDialog, self).__init__()
        self.finish_callback = finish_callback
        self.setMinimumSize(900, 550)
        self.absolute_path = os.path.split(sys.argv[0])[0]
        self.setWindowTitle('建模')
        self.name = '2D无粘模型'
        self.type = 'direct_modeling'  # 'transfer_modeling'
        self.data_set = ''
        self.data_type = None
        self.x_field = []
        self.y_field = []
        self.all_fields = []
        self.iter_nums = 200
        self.init_UI()
        self.__init_data_model_fields()
        self.model_train_status_signal.connect(self.update_train_status)
        self.setWindowIcon(QIcon(f'{self.absolute_path}/res/modeling.png'))

    def init_UI(self):
        # 横向布局
        layout_out = QHBoxLayout()
        self.setLayout(layout_out)
        # 第一列  收敛曲线
        self.lossFigure = Figure_Line(title=f'{self.name}-收敛曲线', xlabel='epoch', ylabel='loss&RMSE')
        layout_out.addWidget(self.lossFigure, stretch=3)
        # 第二列 参数配置
        layout_out.addWidget(self.__init_param_frame(), stretch=2)

    def __init_param_frame(self) -> 'QFrame':
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        vbox_layout = QVBoxLayout()
        vbox_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        frame.setLayout(vbox_layout)
        vbox_layout.addWidget(QLabel("模型参数配置"))

        layout_line = QHBoxLayout()
        btn_1 = QRadioButton('直接训练')
        btn_1.setChecked(True)
        btn_1.toggled.connect(lambda: self.btnstate('direct_modeling'))
        layout_line.addWidget(btn_1)
        self.btn_2 = QRadioButton('迁移训练')
        self.btn_2.toggled.connect(lambda: self.btnstate('transfer_modeling'))
        layout_line.addWidget(self.btn_2)
        vbox_layout.addLayout(layout_line)

        layout_line = QHBoxLayout()
        self.box_datset = QComboBox()
        self.box_datset.currentIndexChanged.connect(self.change_dat_event)

        # self.data_type = node['data_type']
        layout_line.addWidget(QLabel("训练集"))
        layout_line.addWidget(self.box_datset, stretch=1)
        vbox_layout.addLayout(layout_line)

        self.widge_model_chose = QWidget()
        layout_line = QHBoxLayout()
        layout_line.setContentsMargins(0, 0, 0, 0)
        self.widge_model_chose.setLayout(layout_line)
        layout_line.addWidget(QLabel("预训练模型"))
        self.box_pre_model = QComboBox()
        self.box_pre_model.currentIndexChanged.connect(self.show_x_y)

        # 从目录添加预训练模型 =============================
        # absolute_path = os.path.split(sys.argv[0])[0]
        # filePath = f'{absolute_path}/dat/model/'
        # for file in os.listdir(filePath):
        #     self.box_pre_model.addItem(file)
        # ============================================

        layout_line.addWidget(self.box_pre_model, stretch=1)
        vbox_layout.addWidget(self.widge_model_chose)
        self.widge_model_chose.hide()

        layout_line = QHBoxLayout()
        self.xfield_list = ComboList("自变量")
        self.yfield_list = ComboList("应变量")
        # self.__init_xy_fields()
        layout_line.addWidget(self.xfield_list)
        layout_line.addWidget(self.yfield_list)

        vbox_layout.addLayout(layout_line)

        layout_line = QHBoxLayout()
        layout_line.addWidget(QLabel("模型名称"))
        self.model_name_line = QLineEdit(self.name)
        layout_line.addWidget(self.model_name_line, stretch=1)
        vbox_layout.addLayout(layout_line)

        layout_line = QHBoxLayout()
        layout_line.addWidget(QLabel("迭代次数"))
        self.iter_nums_line = QLineEdit(str(self.iter_nums))
        layout_line.addWidget(self.iter_nums_line, stretch=1)
        vbox_layout.addLayout(layout_line)

        # vbox_layout.addStretch(1) 
        layout_line = QHBoxLayout()
        layout_line.setAlignment(Qt.AlignmentFlag.AlignRight)

        btn_train = QPushButton(" 开始训练 ")
        btn_train.clicked.connect(self.inspect_and_run)
        btn_cancel = QPushButton(" 取消 ")
        btn_cancel.clicked.connect(self.stop_work)
        layout_line.addWidget(btn_train)
        layout_line.addWidget(btn_cancel)
        vbox_layout.addLayout(layout_line)
        return frame

    # def createStatusBar(self):
    #     self.statusBar().showMessage('准备中...')
    #     self.progressBar = QProgressBar()
    #     self.label = QLabel()
    #     self.label2 = QLabel()
    #     self.statusBar().addPermanentWidget(self.label)
    #     self.statusBar().addPermanentWidget(self.label2)
    #     self.statusBar().addPermanentWidget(self.progressBar)
    #     self.progressBar.setGeometry(0, 0, 100, 3)
    #     self.progressBar.setRange(0, 100)  # 设置进度条的范围
    #     self.updateProgress("正在处理", '初始化', 5)

    def btnstate(self, b_type):  # 输出按钮1与按钮2的状态，选中还是没选中
        self.type = b_type
        if b_type == 'transfer_modeling':
            self.widge_model_chose.show()
        else:
            self.widge_model_chose.hide()

    def __init_data_model_fields(self):
        tree_data = TreeData.instance()
        for node in tree_data.datas['data']:
            self.box_datset.addItem(node['name'], userData=node)
        self.box_pre_model.addItem('-请选择-')
        for node_modeling in tree_data.datas['modeling']:
            self.box_pre_model.addItem(node_modeling['name'])
        pass

    def __init_xy_fields(self):
        self.data_model = DataCombineModel.loadModel(self.data_type)
        self.file_path = self.data_model.getPath()
        self.all_fields = self.data_model.fields()
        self.xfield_list.setItems(self.all_fields)
        self.yfield_list.setItems(self.all_fields)

    def change_dat_event(self, e):
        current_dat = self.box_datset.currentData()
        if current_dat is not None:
            self.data_type = current_dat['data_type']
            self.__init_xy_fields()
        pass

    def inspect_and_run(self):
        model_name = self.model_name_line.text()
        self.name = model_name
        if os.path.exists(f"{absolute_path}/dat/model/{model_name}/"):
            QMessageBox().information(self, "提示", "模型名称重复！")
        else:
            if self.type == "transfer_modeling":
                self.run_able = self.Transfer_model_build_job
            else:
                self.run_able = self.PreTrain_model_build_job
            self.work = TrainWorker(self.run_able)
            self.work.start()

    def stop_work(self):
        self.work.terminate()
        QMessageBox().information(self, "提示", "已取消模型训练！")

    def PreTrain_model_build_job(self):
        x_count = self.xfield_list.chosed_listView.count()
        y_count = self.yfield_list.chosed_listView.count()
        x_name = []
        y_name = []
        for i in range(x_count):
            x_name.append(self.xfield_list.chosed_listView.item(i).text())
        for i in range(y_count):
            y_name.append(self.yfield_list.chosed_listView.item(i).text())

        epochs = int(self.iter_nums_line.text())
        self.iter_nums = epochs
        model_name = self.model_name_line.text()
        df = self.data_model.df_combined
        print('x_name:', x_name)
        print('y_name:', y_name)
        print('epochs', epochs)
        self.iter_nums = epochs

        self.preTrainModel = PreTrainModel()
        self.preTrainModel.data_init(df, x_name, y_name)
        self.create_train_line()
        self.preTrainModel.model_bulid()
        self.preTrainModel.model_train(epochs, self.add_train_status_data)
        self.preTrainModel.model_save(model_name)

    def Transfer_model_build_job(self):
        x_count = self.xfield_list.chosed_listView.count()
        y_count = self.yfield_list.chosed_listView.count()
        x_name = []
        y_name = []
        for i in range(x_count):
            x_name.append(self.xfield_list.chosed_listView.item(i).text())
        for i in range(y_count):
            y_name.append(self.yfield_list.chosed_listView.item(i).text())
        epochs = int(self.iter_nums_line.text())
        model_name = self.model_name_line.text()
        pre_train_model = self.box_pre_model.currentText()
        df = self.data_model.df_combined
        print('x_name:', x_name)
        print('y_name:', y_name)
        print('epochs', epochs)
        # print("path:", self.file_path)
        print("pre_train_model:", pre_train_model)
        self.iter_nums = epochs
        epochs = math.ceil(epochs / 2)

        self.transferModel = TransferModel()
        self.transferModel.model_load_job(pre_train_model)
        self.transferModel.data_init(df, x_name, y_name)
        self.create_train_line()
        self.transferModel.model_bulid()
        self.transferModel.model_train(epochs, self.add_train_status_data)
        self.transferModel.fine_tune(epochs, self.add_train_status_data)
        self.transferModel.model_save(model_name)

    # 创建一个groupbox, 用来画训练过程的动态曲线
    def create_train_line(self, loss=[], vloss=[]):  # ['loss','RMSE']
        self.lossFigure.add_line('loss', np.array(range(len(loss))), np.array(loss))
        self.lossFigure.add_line('vloss', np.array(range(len(vloss))), np.array(vloss))

    def add_train_status_data(self, inx, loss, vloss, base_epoch=0):
        self.lossFigure.add_data(inx + base_epoch, {'loss': loss, 'vloss': vloss})
        print(f'模型训练{inx}/{self.iter_nums}')
        # self.updateProgress("LSTM模型构建",f'模型训练{inx}/{self.model.epoch_max}',30+int(inx*50/self.model.epoch_max))
        self.model_train_status_signal.emit(inx, loss, vloss)  # 向UI线程发送数据

    def update_train_status(self, inx, loss, vloss):  # 更新UI
        print(inx, '===', self.iter_nums)
        if inx + 1 == self.iter_nums:
            QMessageBox().information(self, "提示", "模型训练完成！")
            if (self.finish_callback):
                self.finish_callback(self.name)

    # def updateProgress(self,msg1, msg2, stepValue):
    #     self.progress_signal.emit(msg1,msg2,stepValue)

    # 模型自变量、因变量展示
    def show_x_y(self):
        model_name = self.box_pre_model.currentText()
        print(model_name)
        if model_name == '-请选择-':
            print('===')
            pass
        else:
            self.model = TrainedModel()
            self.model.model_load_job(model_name)
            self.x_name = self.model.trained_model.x_name
            self.y_name = self.model.trained_model.y_name
            print('x_name:', self.x_name)
            print('y_name:', self.y_name)
            self.xfield_list.chosed_listView.clear()
            self.yfield_list.chosed_listView.clear()
            self.xfield_list.chosed_listView.addItems(self.x_name)
            self.yfield_list.chosed_listView.addItems(self.y_name)


class ComboList(QWidget):
    def __init__(self, name, strs=[]) -> None:
        super().__init__()
        self.init_UI(name, strs)

    def init_UI(self, name, strs):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.box = QComboBox()
        self.box.currentIndexChanged.connect(self.change_field_event)
        self.box.addItems(strs)
        layout_line = QHBoxLayout()
        layout_line.addWidget(QLabel(name))
        layout_line.addWidget(self.box, stretch=1)
        layout.addLayout(layout_line)

        self.chosed_listView = QListWidget()
        layout.addWidget(self.chosed_listView)
        # self.name_listView.clicked.connect(self.name_item_click)
        self.chosed_listView.setContextMenuPolicy(Qt.CustomContextMenu)  # type: ignore # 
        self.chosed_listView.customContextMenuRequested[QPoint].connect(self.item_menu)  # type: ignore

    def item_menu(self, point):
        print(point)
        pop_menu = QMenu(self.chosed_listView)
        delete_item = QAction("删除")
        clear_item = QAction("清空")
        delete_item.triggered.connect(self.delete_action)
        clear_item.triggered.connect(self.chosed_listView.clear)
        pop_menu.addActions([delete_item, clear_item])
        pop_menu.exec_(QCursor.pos())

    def change_field_event(self, e):
        currentText = self.box.currentText()
        for x in range(self.chosed_listView.count()):
            if self.chosed_listView.item(x).text() == currentText:
                return
        self.chosed_listView.addItem(currentText)
        print('选取到了：', currentText)

    def setItems(self, items, clear=True):
        if clear:
            self.box.clear()
        self.chosed_listView.clear()
        self.box.addItems(items)
        self.box.setCurrentIndex(-1)
        self.chosed_listView.clear()

    def delete_action(self):
        row = self.chosed_listView.currentIndex().row()
        self.chosed_listView.takeItem(row)


class TrainWorker(QThread):
    def __init__(self, job):
        super(TrainWorker, self).__init__()
        self.job = job

    def run(self):
        self.job()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    radioDemo = ModelingDialog()
    radioDemo.show()
    if radioDemo.exec_():
        print(radioDemo.name)
        print(radioDemo.type)
