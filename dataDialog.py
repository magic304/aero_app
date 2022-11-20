import os
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class DataDialog(QDialog):

    def __init__(self, parent=None):
        super(DataDialog, self).__init__(parent)
        self.setMinimumSize(500, 0)
        self.absolute_path = os.path.split(sys.argv[0])[0]
        self.setWindowTitle('增加数据集')
        self.setWindowIcon(QIcon(f'{self.absolute_path}/res/data.png'))
        self.name = '2D无粘数据集'
        self.type = '2d_non_viscous'
        self.file = ''

        # 纵向布局
        layout_out = QVBoxLayout()
        # 第一行 水平布局
        layout_row1 = QHBoxLayout()
        layout_row1.addWidget(QLabel("数据集名称："))
        self.loading_label = QLabel(self)
        layout_row1.addWidget(self.loading_label)
        layout_out.addLayout(layout_row1)

        # 第二行
        self.set_name_text = QLineEdit()
        layout_out.addWidget(self.set_name_text)
        self.set_name_text.textChanged.connect(lambda: self.btnstate(self.set_name_text.text()))

        # 第三行
        layout_row3 = QHBoxLayout()
        buttonBox = QDialogButtonBox(parent=self)
        buttonBox.setOrientation(Qt.Horizontal)  # 设置为水平方向
        buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)  # 确定和取消两个按钮 
        buttonBox.accepted.connect(self.accept)  # 确定
        buttonBox.rejected.connect(self.reject)  # 取消
        layout_row3.addWidget(buttonBox)
        layout_out.addLayout(layout_row3)

        self.setLayout(layout_out)

    def btnstate(self, b_type):  # 输出按钮1与按钮2的状态，选中还是没选中
        self.type = b_type
        self.name = self.set_name_text.text()

    def browse_file(self):
        self.file, filetype = QFileDialog.getOpenFileName(self, "选择xls文件", "./", "excel文件(*.xls *.xlsx);;ALL(*.*)")
        self.file_lineEdit.setText(str(self.file))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    radioDemo = DataDialog()
    radioDemo.show()
    if radioDemo.exec_():
        print(radioDemo.name)
        print(radioDemo.type)
        print(radioDemo.file)
