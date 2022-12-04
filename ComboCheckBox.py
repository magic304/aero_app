import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *


class CheckBox(QWidget):
    def __init__(self, name=None, items=None):
        super().__init__()
        self.name = name or '123'
        self.items = items or ['111', '222', '333']
        self.__init_UI()

    def __init_UI(self):
        horizontalLayout = QtWidgets.QHBoxLayout()
        self.label = QLabel()
        self.setMaximumWidth(250)
        self.label.setText(self.name)
        self.comboCheckBox = ComboCheckBox(self.items)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.comboCheckBox.setSizePolicy(sizePolicy)
        horizontalLayout.addWidget(self.label)
        horizontalLayout.addWidget(self.comboCheckBox)
        self.setLayout(horizontalLayout)


class ComboCheckBox(QComboBox):
    def __init__(self, items=None):
        """
        initial function
        :param items: the items of the list
        """
        super(ComboCheckBox, self).__init__()
        self.items = items  # items list
        self.box_list = []  # selected items
        self.text = QLineEdit()  # use to selected items
        self.text.setReadOnly(True)
        q = QListWidget()

        self.box_list.append(QCheckBox())
        self.box_list[0].setChecked(True)
        self.box_list[0].setText('全选')
        item = QListWidgetItem(q)
        q.setItemWidget(item, self.box_list[0])
        self.box_list[0].stateChanged.connect(self.all_choose)

        for i in range(1, len(self.items) + 1):
            self.box_list.append(QCheckBox())
            self.box_list[i].setChecked(True)
            self.box_list[i].setText(f'{self.items[i - 1]}')
            item = QListWidgetItem(q)
            q.setItemWidget(item, self.box_list[i])
            self.box_list[i].stateChanged.connect(self.show_selected)
        self.setLineEdit(self.text)
        self.setModel(q.model())
        self.setView(q)
        self.show_selected()
        self.text.setCursorPosition(0)

    def get_selected(self) -> list:
        """
        get selected items
        :return:
        """
        ret = []
        for i in range(1, len(self.items) + 1):
            if self.box_list[i].isChecked():
                ret.append(self.box_list[i].text())
        return ret

    def show_selected(self):
        """
        show selected items
        :return:
        """
        self.text.clear()
        ret = ','.join(self.get_selected())
        self.text.setText(ret)

    def all_choose(self):
        if self.box_list[0].isChecked():
            for i in range(1, len(self.items) + 1):
                self.box_list[i].setChecked(True)
        else:
            for i in range(1, len(self.items) + 1):
                self.box_list[i].setChecked(False)
        self.text.setCursorPosition(0)


class UiMainWindow(QWidget):
    def __init__(self):
        super(UiMainWindow, self).__init__()
        self.setWindowTitle('Test')
        self.resize(600, 400)
        a = [1, 2, 3]
        str2 = ''.join(str(i) for i in a)
        print(str2)
        print(len(str2))
        combo = CheckBox(items=a)
        layout = QVBoxLayout()
        layout.addWidget(combo)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = UiMainWindow()
    ui.show()
    sys.exit(app.exec_())
