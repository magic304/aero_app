# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '1.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget, QMessageBox

from modingDialog import ComboList


class Ui_plot_view(QWidget):

    def __init__(self, x_name=None, y_name=None, finish_callback=None):
        super().__init__()
        plot_view = self
        self.finish_callback = finish_callback
        self.x_name = x_name or ["1", '2', '3']
        self.y_name = y_name or ["1", '2', '3']
        plot_view.setObjectName("plot_view")
        plot_view.resize(800, 400)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(plot_view)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.setting_frame = QtWidgets.QFrame(plot_view)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.setting_frame.sizePolicy().hasHeightForWidth())
        self.setting_frame.setSizePolicy(sizePolicy)
        self.setting_frame.setMinimumSize(QtCore.QSize(400, 0))
        self.setting_frame.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.setting_frame.setObjectName("setting_frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.setting_frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radioButton_2d = QtWidgets.QRadioButton(self.setting_frame)
        self.radioButton_2d.setChecked(True)
        self.radioButton_2d.setObjectName("radioButton_2d")
        self.horizontalLayout_2.addWidget(self.radioButton_2d)
        self.radioButton_3d = QtWidgets.QRadioButton(self.setting_frame)
        self.radioButton_3d.setObjectName("radioButton_3d")
        self.horizontalLayout_2.addWidget(self.radioButton_3d)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.x_combox = ComboList("x轴")
        self.horizontalLayout.addWidget(self.x_combox)
        self.y_combox = ComboList("y轴")
        self.horizontalLayout.addWidget(self.y_combox)
        self.z_combox = ComboList("分组")
        self.horizontalLayout.addWidget(self.z_combox)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.line2 = QtWidgets.QFrame(self.setting_frame)
        self.line2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line2.setObjectName("line")
        self.verticalLayout.addWidget(self.line2)
        self.pushButton_plot = QtWidgets.QPushButton(self.setting_frame)
        self.pushButton_plot.setObjectName("pushButton_plot")
        self.horizontalLayout_4.addWidget(self.pushButton_plot)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5.addWidget(self.setting_frame)

        self.retranslateUi(plot_view)
        self.radioButton_2d.clicked.connect(self.z_combox.hide)  # type: ignore
        self.radioButton_3d.clicked.connect(self.z_combox.show)  # type: ignore
        self.pushButton_plot.clicked.connect(self.get_setting)
        # QtCore.QMetaObject.connectSlotsByName(plot_view)

    def retranslateUi(self, plot_view):
        _translate = QtCore.QCoreApplication.translate
        plot_view.setWindowTitle(_translate("plot_view", "可视化展示设置"))
        self.radioButton_2d.setText(_translate("plot_view", "二维视图"))
        self.radioButton_3d.setText(_translate("plot_view", "分组绘图"))
        self.pushButton_plot.setText(_translate("plot_view", "绘图"))

        self.z_combox.hide()
        self.__init_xy_fields()

    def __init_xy_fields(self):
        if self.x_name is None or self.y_name is None:
            pass
        else:
            self.x_combox.setItems(self.x_name, clear=False)
            self.x_combox.setItems(self.y_name, clear=False)
            self.y_combox.setItems(self.x_name, clear=False)
            self.y_combox.setItems(self.y_name, clear=False)
            self.z_combox.setItems(self.x_name, clear=False)
            self.z_combox.setItems(self.y_name, clear=False)

    def get_setting(self):

        self.x_name = None
        self.y_name = None
        self.z_name = None

        if self.radioButton_2d.isChecked():
            self.plot_type = '二维绘图'
        elif self.radioButton_3d.isChecked():
            self.plot_type = '分组绘图'

        print('plot_type:', self.plot_type)
        x_count = self.x_combox.chosed_listView.count()
        x_name = []
        for i in range(x_count):
            x_name.append(self.x_combox.chosed_listView.item(i).text())
        self.x_name = x_name
        print('self.x_name:', self.x_name)

        y_count = self.y_combox.chosed_listView.count()
        y_name = []
        for i in range(y_count):
            y_name.append(self.y_combox.chosed_listView.item(i).text())
        self.y_name = y_name
        print('self.y_name:', self.y_name)

        z_count = self.z_combox.chosed_listView.count()
        z_name = []
        for i in range(z_count):
            z_name.append(self.z_combox.chosed_listView.item(i).text())
        self.z_name = z_name
        print('self.z_name:', self.z_name)

        if self.plot_type == '二维绘图' and (len(self.x_name) == 0 or len(self.y_name) == 0):
            QMessageBox().information(self, "提示", "请输入x轴与y轴数据！")
        elif self.plot_type == '分组绘图' and (len(self.x_name) == 0 or len(self.y_name) == 0 or len(self.z_name) == 0):
            QMessageBox().information(self, "提示", "请输入x轴、y轴与分组数据！")
        elif len(self.z_name) > 1:
            QMessageBox().information(self, "提示", "分组数量不能大于1！")
        else:
            if (self.finish_callback):
                self.finish_callback(self.plot_type, self.x_name, self.y_name, self.z_name)
                self.close()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_plot_view()
    ui.show()
    sys.exit(app.exec_())
