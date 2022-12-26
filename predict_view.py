# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PIL import Image
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog

from ComboCheckBox import CheckBox
from DataWidget import *
from aero_model.TrainedModel import TrainedModel
from data_oper import *
from figure_3D import *
from plot_view import Ui_plot_view
from util import *

absolute_path = os.path.split(sys.argv[0])[0]
os.environ["PATH"] += os.pathsep + absolute_path + '/Graphviz/bin'


class Ui_widget_5(QWidget):

    def __init__(self, name=None):
        super().__init__()
        self.df = None
        self.absolute_path = os.path.split(sys.argv[0])[0]
        self.name = name or '-请选择-'
        self.setupUi()
        self.model.df = None
        self.color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf']

    def setupUi(self):
        widget_5 = self
        widget_5.setObjectName("widget_5")
        widget_5.resize(821, 563)
        self.gridLayout = QtWidgets.QGridLayout(widget_5)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.frame = QtWidgets.QFrame(widget_5)
        self.frame.setAutoFillBackground(False)
        self.frame.setMaximumWidth(400)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setObjectName("frame")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.textEdit = QtWidgets.QLineEdit(self.frame)
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout.addWidget(self.textEdit)
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_5.addLayout(self.verticalLayout_2)
        self.line = QtWidgets.QFrame(self.frame)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_5.addWidget(self.line)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setBold(True)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.comboBox = QtWidgets.QComboBox(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_4.addWidget(self.comboBox)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_3.addWidget(self.label_6)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.widget = QtWidgets.QListWidget(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(20, 40))
        self.widget.setObjectName("widget")
        self.horizontalLayout_5.addWidget(self.widget)
        self.widget_3 = QtWidgets.QListWidget(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setMinimumSize(QtCore.QSize(20, 40))
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_5.addWidget(self.widget_3)
        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.line_3 = QtWidgets.QFrame(self.frame)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")

        self.summary_label = QtWidgets.QLabel(self.frame)
        # self.summary_label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setBold(True)
        self.summary_label.setFont(font)
        self.summary_label.setObjectName("summary_label")
        self.summary = QtWidgets.QLabel(self.frame)
        self.summary.setAlignment(Qt.AlignCenter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        # self.summary.setSizePolicy(sizePolicy)
        self.verticalLayout.addWidget(self.line_3)
        self.verticalLayout.addWidget(self.summary_label)
        self.verticalLayout.addWidget(self.summary)
        self.verticalLayout_5.addLayout(self.verticalLayout)
        self.line_2 = QtWidgets.QFrame(self.frame)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_5.addWidget(self.line_2)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_5.addWidget(self.pushButton_4)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.pushButton_6 = QtWidgets.QPushButton(self.frame)
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_6.addWidget(self.pushButton_6)
        self.pushButton_5 = QtWidgets.QPushButton(self.frame)
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout_6.addWidget(self.pushButton_5)
        self.verticalLayout_5.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7.addWidget(self.frame)
        self.horizontalLayout_7.setStretch(0, 1)
        self.tabWidget = QtWidgets.QTabWidget(widget_5)
        # self.tabWidget.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabWidget.setObjectName("tabWidget")
        self.tabWidget.setTabPosition(QTabWidget.TabPosition.East)
        self.horizontalLayout_9.addWidget(self.tabWidget)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_9.setStretch(0, 2)  # 第0编号的组件stretch为2
        self.horizontalLayout_9.setStretch(1, 1)  # 第1编号的组件stretch为1
        self.gridLayout.addLayout(self.horizontalLayout_9, 0, 0, 1, 1)

        self.retranslateUi(widget_5)
        self.tabWidget.setCurrentIndex(0)
        # QtCore.QMetaObject.connectSlotsByName(widget_5)
        self.connection(widget_5)
        self.__init_data_model_fields()

    def retranslateUi(self, widget_5):
        _translate = QtCore.QCoreApplication.translate
        widget_5.setWindowTitle(_translate("widget_5", "模型预测"))
        self.label.setText(_translate("widget_5", "数据文件："))
        self.pushButton.setText(_translate("widget_5", "浏览/上传"))
        self.pushButton.setIcon(QIcon(f'{self.absolute_path}/res/上传.png'))
        self.label_3.setText(_translate("widget_5", "模型参数配置:"))
        self.label_4.setText(_translate("widget_5", "选择预测模型"))
        self.label_5.setText(_translate("widget_5", "自变量："))
        self.label_6.setText(_translate("widget_5", "因变量："))
        self.summary_label.setText(_translate("widget_5", "模型结构:"))
        self.pushButton_4.setText(_translate("widget_5", "开始预测"))
        self.pushButton_4.setIcon(QIcon(f'{self.absolute_path}/res/预测.png'))
        self.pushButton_6.setText(_translate("widget_5", "预测结果下载"))
        self.pushButton_6.setIcon(QIcon(f'{self.absolute_path}/res/下载.png'))
        self.pushButton_5.setText(_translate("widget_5", "可视化展示"))
        self.pushButton_5.setIcon(QIcon(f'{self.absolute_path}/res/绘图.png'))

    # 方法
    def connection(self, widget_5):
        self.pushButton.clicked.connect(lambda: self.browse_file(widget_5))
        self.comboBox.currentIndexChanged.connect(self.show_x_y)
        self.tabWidget.setTabsClosable(True)  # 给每个tab添加关闭按钮
        self.tabWidget.tabCloseRequested.connect(self.tabclose)
        self.pushButton_4.clicked.connect(self.predict_work)
        self.pushButton_6.clicked.connect(self.download)
        self.pushButton_5.clicked.connect(self.plot_work)

    def browse_file(self, widget_5):
        file, filetype = QFileDialog.getOpenFileName(widget_5, "选择CSV文件", "./", "CSV文件(*.csv);;ALL(*.*)")
        # 防止取消后程序崩溃，保证第二个参数不为空再继续
        if filetype:
            self.textEdit.clear()
            self.textEdit.setText(str(file))
            self.df = found_df(str(file))
            self.df2 = self.df
            self.tab = Data_Table()
            self.tab.set_data_df(df=self.df)
            self.tabWidget.addTab(self.tab, QIcon(f'{self.absolute_path}/res/data.png'), "数据展示")
            self.tabWidget.setCurrentWidget(self.tab)

    # 模型自变量、因变量展示
    def show_x_y(self):
        model_name = self.comboBox.currentText()
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
            self.widget.clear()
            self.widget_3.clear()
            self.widget.addItems(self.x_name)
            self.widget_3.addItems(self.y_name)
            # summary展示
            self.summary.clear()
            path = 'model.png'
            # 提前缩放图片，防止图片模糊======================
            # 获取屏幕分辨率
            screenRect = QApplication.desktop().screenGeometry()
            # 图片想要的占比
            y = screenRect.height() * 0.5
            print(y)
            img = Image.open(path)
            width, height = img.size
            size = (width, y)
            img.thumbnail(size)
            img.save(path, 'png')

            image = QtGui.QPixmap(path)
            self.summary.setPixmap(image)
            # self.screen_df = self.model.df
            # self.summary.setScaledContents(True)  # 自适应宽高

    def __init_data_model_fields(self):
        tree_data = TreeData.instance()
        for node_modeling in tree_data.datas['modeling']:
            self.comboBox.addItem(node_modeling['name'])
        self.comboBox.setCurrentText(self.name)
        pass

    # 关闭tab选项卡
    def tabclose(self, index):
        self.tabWidget.removeTab(index)

    def predict_work(self):
        if self.df is None:
            QMessageBox().information(self, "提示", "请上传数据！")
        elif self.widget.count() == 0:
            QMessageBox().information(self, "提示", "请选择模型！")
        else:
            self.model.predict(self.df)
            self.model.result_to_df()
            QMessageBox().information(self, "提示", "预测完成！")

            # 添加筛选行
            widget = QWidget()
            self.screen_horizontalLayout = []
            screen_horizontalLayout = QtWidgets.QHBoxLayout()
            self.screen_horizontalLayout.append(screen_horizontalLayout)
            verticalLayout = QtWidgets.QVBoxLayout()
            self.add_screen()
            self.screen_button = QPushButton()
            self.screen_button.setText('筛选')
            self.screen_button.setIcon(QIcon(f'{self.absolute_path}/res/筛选.png'))
            self.screen_button.clicked.connect(self.screen_work)
            self.screen_horizontalLayout[0].addWidget(self.screen_button)
            self.tab2 = Data_Table()
            self.tab2.set_data_df(df=self.model.df)
            for i in range(len(self.screen_horizontalLayout)):
                verticalLayout.addLayout(self.screen_horizontalLayout[i])
            verticalLayout.addWidget(self.tab2)
            widget.setLayout(verticalLayout)
            self.tabWidget.addTab(widget, QIcon(f'{self.absolute_path}/res/预测.png'), "预测结果")
            self.tabWidget.setCurrentWidget(widget)
            self.screen_df = self.model.df
            self.x_len = len(self.model.trained_model.x_name)
            print(self.x_len)

    def download(self):
        save_path, save_type = QFileDialog.getSaveFileName(self, "./", "./result.csv", "csv(*.csv)")
        print(save_type)
        # 防止取消后程序崩溃，保证第二个参数不为空再继续
        if save_type:
            self.model.df.to_csv(str(save_path), sep=',', index=False, header=True, encoding='utf_8_sig')

    def plot_work(self):
        self.plot = Ui_plot_view(self.x_name, self.y_name, self.finish_callback)
        self.plot.show()

    def finish_callback(self, plot_type, group_type, x_name, y_name, z_name, group_name):
        self.plot_type = plot_type
        self.group_type = group_type
        self.plot_x_name = x_name
        self.plot_y_name = y_name
        self.plot_z_name = z_name
        self.plot_group_name = group_name
        if self.model.df is None:
            QMessageBox().information(self, "提示", "请先预测再绘图！")
        else:
            # plot
            if self.plot_type == '二维绘图' and self.group_type is False:
                self.plot_widget = QWidget()
                self.plot_gridLayout = QGridLayout()
                self.plot_2d()
                self.plot_widget.setLayout(self.plot_gridLayout)
                self.tabWidget.addTab(self.plot_widget, QIcon(f'{self.absolute_path}/res/绘图.png'), f'预测结果-{plot_type}')
                self.tabWidget.setCurrentWidget(self.plot_widget)
            elif self.plot_type == '二维绘图' and self.group_type is True:
                self.plot_widget = QWidget()
                self.plot_gridLayout = QGridLayout()
                self.plot_2d_group()
                self.plot_widget.setLayout(self.plot_gridLayout)
                self.tabWidget.addTab(self.plot_widget, QIcon(f'{self.absolute_path}/res/绘图.png'), f'预测结果-{plot_type}')
                self.tabWidget.setCurrentWidget(self.plot_widget)
            elif self.plot_type == '三维绘图' and self.group_type is False:
                self.plot_widget = QWidget()
                self.plot_gridLayout = QGridLayout()
                self.plot_3d()
                self.plot_widget.setLayout(self.plot_gridLayout)
                self.tabWidget.addTab(self.plot_widget, QIcon(f'{self.absolute_path}/res/绘图.png'), f'预测结果-{plot_type}')
                self.tabWidget.setCurrentWidget(self.plot_widget)
            elif self.plot_type == '三维绘图' and self.group_type is True:
                self.plot_widget = QWidget()
                self.plot_gridLayout = QGridLayout()
                self.plot_3d_group()
                self.plot_widget.setLayout(self.plot_gridLayout)
                self.tabWidget.addTab(self.plot_widget, QIcon(f'{self.absolute_path}/res/绘图.png'), f'预测结果-{plot_type}')
                self.tabWidget.setCurrentWidget(self.plot_widget)

    def add_screen(self):
        self.name_list = list(self.model.df)
        self.screen_box = []
        for i in range(len(self.name_list)):
            items = self.model.df[self.name_list[i]].to_numpy()
            items = np.unique(items)
            name = self.name_list[i]
            box = CheckBox(name, items.tolist())
            if len(self.screen_horizontalLayout) > i / 5:
                self.screen_horizontalLayout[len(self.screen_horizontalLayout) - 1].addWidget(box)
                self.screen_box.append(box)
            else:
                screen_horizontalLayout = QtWidgets.QHBoxLayout()
                screen_horizontalLayout.setAlignment(Qt.AlignLeft)
                self.screen_horizontalLayout.append(screen_horizontalLayout)
                self.screen_horizontalLayout[len(self.screen_horizontalLayout) - 1].addWidget(box)
                self.screen_box.append(box)

    def screen_work(self):
        self.screen_df = self.model.df
        self.df2 = self.df
        for i in range(len(self.name_list)):
            text_data = self.screen_box[i].comboCheckBox.get_selected()
            math_score = []
            for j in range(len(text_data)):
                math_score.append(eval(text_data[j]))
            self.screen_df = self.screen_df[self.screen_df[self.name_list[i]].isin(math_score)]
            if i < self.x_len:
                self.df2 = self.df2[self.df2[self.name_list[i]].isin(math_score)]
        self.tab2.clear()
        self.tab2.set_data_df(self.screen_df)

    def plot_2d(self):
        for i in range(len(self.plot_x_name)):
            df = self.screen_df
            df.sort_values(by=self.plot_x_name[i], inplace=True, ascending=True)
            input_df = self.df2
            input_df.sort_values(by=self.plot_x_name[i], inplace=True, ascending=True)
            for j in range(len(self.plot_y_name)):
                color = 0
                figure = Figure_Line(title=f'预测结果({self.plot_x_name[i]}--{self.plot_y_name[j]})',
                                     xlabel=f'{self.plot_x_name[i]}',
                                     ylabel=f'{self.plot_y_name[j]}')
                x = df[self.plot_x_name[i]].to_numpy()
                y = df[self.plot_y_name[j]].to_numpy()
                print(j % 10)
                figure.add_line(f'预测数据', x, y, style={'ls': '-.', 'marker': '', 'color': self.color[color % 10]})
                try:
                    x = input_df[self.plot_x_name[i]].to_numpy()
                    y = input_df[self.plot_y_name[j]].to_numpy()
                except:
                    color += 1
                    self.plot_gridLayout.addWidget(figure, i, j)
                    continue
                else:
                    if x is not None and y is not None:
                        figure.add_line(f'原始数据', x, y, style={'ls': '', 'marker': 'o', 'color': self.color[color % 10]})
                color += 1
                self.plot_gridLayout.addWidget(figure, i, j)

    def plot_3d(self):
        pass

    def plot_2d_group(self):

        z = self.screen_df[self.plot_group_name].to_numpy()
        z = np.unique(z)
        for i in range(len(self.plot_x_name)):
            df = self.screen_df
            df.sort_values(by=self.plot_x_name[i], inplace=True, ascending=True)
            input_df = self.df2
            input_df.sort_values(by=self.plot_x_name[i], inplace=True, ascending=True)
            for j in range(len(self.plot_y_name)):
                color = 0
                figure = Figure_Line(title=f'预测结果({self.plot_x_name[i]}--{self.plot_y_name[j]})',
                                     xlabel=f'{self.plot_x_name[i]}',
                                     ylabel=f'{self.plot_y_name[j]}')
                for k in range(len(z)):
                    num = z[k]
                    name = self.plot_group_name
                    df1 = df[df[name[0]] == num]
                    x = df1[self.plot_x_name[i]].to_numpy()
                    y = df1[self.plot_y_name[j]].to_numpy()
                    if x is not None and y is not None:
                        figure.add_line(f'预测数据[{self.plot_group_name[0]}={z[k]}]', x, y,
                                        style={'ls': '-.', 'marker': '', 'color': self.color[color % 10]})
                    df2 = input_df[input_df[name[0]] == num]
                    try:
                        x = df2[self.plot_x_name[i]].to_numpy()
                        y = df2[self.plot_y_name[j]].to_numpy()
                    except:
                        color += 1
                        continue
                    else:
                        if x is not None and y is not None:
                            figure.add_line(f'原始数据[{self.plot_group_name[0]}={z[k]}]', x, y,
                                            style={'ls': '', 'marker': 'o', 'color': self.color[color % 10]})
                        color += 1
                self.plot_gridLayout.addWidget(figure, i, j)

    def plot_3d_group(self):

        z = self.screen_df[self.plot_group_name].to_numpy()
        z = np.unique(z)
        for i in range(len(self.plot_x_name)):
            df = self.screen_df
            df.sort_values(by=self.plot_x_name[i], inplace=True, ascending=True)
            input_df = self.df2
            input_df.sort_values(by=self.plot_x_name[i], inplace=True, ascending=True)
            for j in range(len(self.plot_y_name)):
                color = 0
                figure = Figure_Line_3D(title=f'预测结果({self.plot_x_name[i]}--{self.plot_y_name[j]})',
                                        xlabel=f'{self.plot_group_name[0]}',
                                        ylabel=f'{self.plot_x_name[i]}',
                                        zlabel=f'{self.plot_y_name[j]}')
                for k in range(len(z)):
                    num = z[k]
                    name = self.plot_group_name
                    df1 = df[df[name[0]] == num]
                    x = df1[self.plot_x_name[i]].to_numpy()
                    y = df1[self.plot_y_name[j]].to_numpy()
                    group = df1[self.plot_group_name[0]].to_numpy()
                    if x is not None and y is not None:
                        figure.add_line(f'预测数据[{self.plot_group_name[0]}={z[k]}]', group, x, y,
                                        style={'ls': '-.', 'marker': '', 'color': self.color[color % 10]})
                    df2 = input_df[input_df[name[0]] == num]
                    try:
                        x = df2[self.plot_x_name[i]].to_numpy()
                        y = df2[self.plot_y_name[j]].to_numpy()
                    except:
                        color += 1
                        continue
                    else:
                        if x is not None and y is not None:
                            figure.add_line(f'原始数据[{self.plot_group_name[0]}={z[k]}]', group, x, y,
                                            style={'ls': '', 'marker': 'o', 'color': self.color[color % 10]})
                        color += 1
                self.plot_gridLayout.addWidget(figure, i, j)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_widget_5()
    ui.show()
    sys.exit(app.exec_())
