import datetime
import os
import shutil
import sys
import uuid

from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
import os
import sys
from figures import *

absolute_path = os.path.split(sys.argv[0])[0]


class DataWidget(QMainWindow):
    progress_signal = pyqtSignal(str, str, int)
    model_train_status_signal = pyqtSignal(int, float, float)  # index, loss, rmse
    model_train_finish_signal = pyqtSignal(str)

    @staticmethod
    def clear(type):
        DataCombineModel.loadModel(type).clear()

    def __init__(self, type=None, callback=None):
        # self.absolute_path = os.path.split(sys.argv[0])[0].replace('\\','/')
        super(DataWidget, self).__init__()
        self.absolute_path = os.path.split(sys.argv[0])[0]
        self.type = type
        self.name = self.type
        self.model = DataCombineModel.loadModel(self.type)
        self.name = f"{self.name}"
        self.setWindowTitle(f'{self.name}')
        self.resize(1200, 900)
        self.initUI()
        # 
        self.progress_signal.connect(self.update_progress_status_ui)
        self.model_train_finish_signal.connect(self.train_finish)
        # run_able =self.model_build_job
        # # if file_id:
        # #     run_able = self.model_load_job
        # self.worker = TrainWorker(self.init_data_bymodel)
        # self.worker.start()
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.init_data_bymodel())
        timer.start(100)

        self.callback = callback

    def initUI(self):
        widget = QWidget()
        main_layout = QHBoxLayout()

        widget.setLayout(main_layout)
        self.setCentralWidget(widget)
        self.createStatusBar()
        # 左右布局
        left = self.ui_left()
        right = self.ui_right()

        main_layout.addWidget(left)
        main_layout.addWidget(right)

    def ui_left(self):
        frame01 = QFrame()
        frame01.setFrameShape(QFrame.StyledPanel)

        layout_left_top = QVBoxLayout()
        frame01.setLayout(layout_left_top)

        lb1 = QLabel(f"{self.name}-数据")
        lb1.setAlignment(Qt.AlignmentFlag.AlignTop)
        lb1.setStyleSheet(
            "QLabel{color:rgb(0,0,0);font-size:20px;font-weight:bold;font-family:Arial;}")
        layout_left_top.addWidget(lb1)

        self.table_datas = Data_Table()
        layout_left_top.addWidget(self.table_datas, stretch=1)
        return frame01

    def ui_right(self):
        frame01 = QFrame()
        frame01.setMaximumWidth(400)
        frame01.setFrameShape(QFrame.StyledPanel)
        vbox = QVBoxLayout()
        frame01.setLayout(vbox)

        self.label_top = QLabel(f"{self.name}数据操作")
        vbox.addWidget(self.label_top)

        vbox.addWidget(QLabel("数据文件:"))
        self.file_lineEdit = QLineEdit(self)
        layout_h = QHBoxLayout()
        layout_h.addWidget(self.file_lineEdit, stretch=1)
        self.file_browse_btn = QPushButton(self)
        self.file_browse_btn.setText("浏览/上传")
        self.file_browse_btn.setIcon(QIcon(f'{self.absolute_path}/res/上传.png'))
        self.file_browse_btn.clicked.connect(lambda: self.browse_file())
        layout_h.addWidget(self.file_browse_btn)
        vbox.addLayout(layout_h)

        self.label_table = QLabel(f"数据列表")
        vbox.addWidget(self.label_table)
        self.table_history = Data_Table(titles=["文件名", "上传时间", ''])
        self.table_history.setColumnHidden(2, True)
        vbox.addWidget(self.table_history, stretch=1)
        vbox.setAlignment(Qt.AlignmentFlag.AlignTop)

        btn_combine = QPushButton(" 合并入库 ")
        btn_combine.setIcon(QIcon(f'{self.absolute_path}/res/数据入库.png'))
        btn_combine.clicked.connect(self.combine)
        btn_download = QPushButton(" 下载 ")
        btn_download.setIcon(QIcon(f'{self.absolute_path}/res/下载.png'))
        btn_download.clicked.connect(self.download)
        layout_btn = QHBoxLayout()
        layout_btn.addWidget(btn_combine)
        layout_btn.addWidget(btn_download)
        vbox.addLayout(layout_btn)
        return frame01

    def combine(self):
        self.model.combine_file("utf-8")
        self.table_datas.set_data_df(self.model.df_combined)

    def download(self):
        if self.model.df_combined is None:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("未进行入库操作！")
            msgBox.setWindowTitle("下载提示")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.exec()
        else:
            options = QFileDialog.Options()
            # options |= QFileDialog.DontUseNativeDialog
            fileName, _ = QFileDialog.getSaveFileName(self, caption="保存CSV文件", directory="",
                                                      filter="CSV文件(*.csv);;ALL(*.*)", options=options)
            if fileName:
                self.model.df_combined.to_csv(fileName, encoding="utf-8")

    def init_data_bymodel(self):
        self.updateProgress("正在处理", '加载数据', 25)
        model = self.model
        for file_info in model.file_list:
            print(file_info)
            self.table_history.append_row(list(file_info.values()))
        self.updateProgress("正在处理", '加载数据', 50)
        if model.df_combined is not None:
            print(file_info)
            self.table_datas.set_data_df(self.model.df_combined)
        self.updateProgress("正在处理", '加载数据', 100)

    def browse_file(self):
        files, filetype = QFileDialog.getOpenFileNames(
            self, "选择CSV文件", "./", "CSV文件(*.csv);;ALL(*.*)")
        if (len(files) > 0):
            self.file_lineEdit.setText(str(files))
            for file_name in files:
                file_info = self.model.upload_file(file_path=file_name)
                self.table_history.append_row(list(file_info.values()))
            self.model.saveModel()

    def creat_by_file(self, file_name):
        file_info = self.model.upload_file(file_path=file_name)
        self.table_history.append_row(list(file_info.values()))
        self.model.saveModel()

    def createStatusBar(self):
        self.statusBar().showMessage('准备中...')
        self.progressBar = QProgressBar()
        self.label = QLabel()
        self.label2 = QLabel()
        self.statusBar().addPermanentWidget(self.label)
        self.statusBar().addPermanentWidget(self.label2)
        self.statusBar().addPermanentWidget(self.progressBar)
        self.progressBar.setGeometry(0, 0, 100, 3)
        self.progressBar.setRange(0, 100)  # 设置进度条的范围
        self.updateProgress("正在处理", '初始化', 5)

    def update_progress_status_ui(self, msg1, msg2, stepValue):
        self.label.setText(msg1)
        self.label2.setText(msg2)
        self.progressBar.setValue(stepValue)
        if (stepValue == 100):
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self.statusBar().hide())
            timer.start(1000)

    def updateProgress(self, msg1, msg2, stepValue):
        self.progress_signal.emit(msg1, msg2, stepValue)

    def train_finish(self, status):
        self.updateProgress("组合模型", '完成', 100)


class TrainWorker(QThread):
    def __init__(self, job):
        super(TrainWorker, self).__init__()
        self.job = job

    def run(self):
        self.job()


class DataCombineModel:
    def __init__(self, data_type) -> None:
        self.data_type = data_type
        self.df_combined: pd.DataFrame = None
        self.file_list = []
        pass

    def combine_file(self, encoding):
        frames = []
        for file_info in self.file_list:
            save_name = file_info['save_name']
            frames.append(
                pd.read_csv(f'{absolute_path}/dat/files_dat_combine/upload/{save_name}.csv', encoding=encoding))
        self.df_combined = pd.concat(frames)
        self.saveModel()

    def fields(self):
        if self.df_combined is not None and self.df_combined.any:
            return list(self.df_combined)
        else:
            return []

    def upload_file(self, file_path):
        fileName = os.path.basename(file_path)
        now_time = datetime.datetime.now()
        now_time_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
        save_name = str(uuid.uuid1())
        if not os.path.exists(f"{absolute_path}/dat/files_dat_combine/upload/"):
            os.makedirs(f"{absolute_path}/dat/files_dat_combine/upload/")
        shutil.copy(file_path, f'{absolute_path}/dat/files_dat_combine/upload/{save_name}.csv')
        file_info = {'file_name': fileName, 'upload_time': now_time_str, 'save_name': save_name}
        self.file_list.append(file_info)
        return file_info

    def saveModel(self):
        import pickle
        file_id = f"combine_{self.data_type}"
        if not os.path.exists(f"{absolute_path}/dat/files_dat_combine/"):
            os.makedirs(f"{absolute_path}/dat/files_dat_combine/")
        path = f"{absolute_path}/dat/files_dat_combine/{file_id}.pkl"
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()
        return file_id

    def clear(self):
        file_id = f"combine_{self.data_type}"
        path = f"{absolute_path}/dat/files_dat_combine/{file_id}.pkl"
        if os.path.exists(path):
            os.remove(path)
        for file_info in self.file_list:
            save_name = file_info['save_name']
            path = f'{absolute_path}/dat/files_dat_combine/upload/{save_name}.csv'
            if os.path.exists(path):
                os.remove(path)

    def getPath(self):
        file_id = f"combine_{self.data_type}"
        path = f"{absolute_path}/dat/files_dat_combine/{file_id}.pkl"
        for file_info in self.file_list:
            save_name = file_info['save_name']
            path = f'{absolute_path}/dat/files_dat_combine/upload/{save_name}.csv'
            if os.path.exists(path):
                return path
            else:
                print('未找到该文件')

    @staticmethod
    def loadModel(data_type) -> 'DataCombineModel':
        file_id = f"combine_{data_type}"
        import pickle
        try:
            path = f"{absolute_path}/dat/files_dat_combine/{file_id}.pkl"
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        except Exception as e:
            obj = DataCombineModel(data_type=data_type)
        return obj


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainMindow = DataWidget()  # 新的训练
    mainMindow.show()
    sys.exit(app.exec_())
