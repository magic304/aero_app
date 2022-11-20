# from SaleLSTMWidget import LSTMwidget
import ctypes

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget, QAction, QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem, QMessageBox

from DataWidget import *
from dataDialog import *
from modingDialog import *
from predict_view import Ui_widget_5
# from pydot import graphviz

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")
import os
absolute_path = os.path.split(sys.argv[0])[0]
os.environ["PATH"] += os.pathsep + absolute_path+'/Graphviz/bin'  #注意修改你的路径


class MainWindow(QMainWindow):
    meta_node_types = ['data', 'modeling']
    meta_node_names = ['数据', '模型']
    meta_node_icos = ['data.png', 'modeling.png']

    def __init__(self):
        super().__init__()
        self.absolute_path = os.path.split(sys.argv[0])[0].replace('\\', '/')
        self.tree_root_nodes = {}
        self.initUI()

    def initUI(self):
        splitter = QSplitter(self)

        self.treeview = QTreeWidget(headerHidden=True)
        self.treeview.setMinimumWidth(200)
        self.treeview.setMaximumWidth(300)
        self.treeview.setColumnCount(1)
        self.treeview.itemClicked.connect(self.onTreeClicked)
        self.treeview.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeview.customContextMenuRequested.connect(self.rightClickMenu)
        self.treeview.setHeaderLabels(['操作树'])
        self.loadTree()

        # treeview.setMaximumSize(250,768)
        self.tabwidget = QTabWidget()
        self.tabwidget.setTabsClosable(True)
        self.tabwidget.tabCloseRequested.connect(self.closeTab)
        # self.connect(self.tabwidget,SIGNAL("tabCloseRequested(int)"),self.closeTab)
        self.tabwidget.setTabShape(QTabWidget.TabShape.Triangular)
        self.tabwidget.setMinimumWidth(800)
        self.tabwidget.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self.treeview)
        splitter.addWidget(self.tabwidget)
        self.setCentralWidget(splitter)
        # statusbar = self.statusBar()

        toolbar = self.addToolBar('工具条')
        dataAction = QAction(QIcon(f'{self.absolute_path}/res/data.png'), '数据', self)
        dataAction.setShortcut('Ctrl+Q')
        dataAction.setStatusTip('数据')
        dataAction.triggered.connect(self.dataActionClicked)
        toolbar.addAction(dataAction)

        modelingAction = QAction(QIcon(f'{self.absolute_path}/res/modeling.png'), '建模', self)
        modelingAction.setShortcut('Ctrl+W')
        modelingAction.setStatusTip('建模')
        modelingAction.triggered.connect(self.modelingActionClicked)
        toolbar.addAction(modelingAction)

        self.setGeometry(30, 30, 1200, 900)
        self.setWindowTitle('进气道数据/建模平台')
        self.setWindowIcon(QIcon(f'{self.absolute_path}/res/data.png'))
        self.center()
        self.show()

    def closeTab(self, currentIndex):
        self.tabwidget.removeTab(currentIndex)

    def dataActionClicked(self):
        print("dataActionClicked")
        dialog = DataDialog()
        if dialog.exec_():
            try:
                node_type = 'data'
                # file = dialog.file
                data_type = dialog.type
                if self.tree_dat.is_exist(node_type, data_type):
                    QMessageBox().information(self, "提示", "该数据集已存在！")
                    self.tree_root_nodes[node_type].setExpanded(True)
                    return
                name = dialog.name
                dat_widge = DataWidget(type=data_type)
                # dat_widge.creat_by_file(file_name=file)
                self.tabwidget.addTab(dat_widge, f"{name}")
                self.tabwidget.setCurrentWidget(dat_widge)
                self.tree_dat.add_node(parent_type=node_type, name=name, data_type=data_type)
                leaf_node = self.add_tree_node(node_type=node_type, node_name=name, node_dat_file='',
                                               data_type=data_type)
                self.tree_root_nodes[node_type].setExpanded(True)
                currNode = self.treeview.currentItem()
                if currNode:
                    currNode.setSelected(False)
                leaf_node.setSelected(True)
            except Exception as e:
                print(str(e))

    def modelingActionClicked(self):
        print("modelingActionClicked")
        dialog = ModelingDialog(self.modeling_finish_callback)
        dialog.exec_()

    def modeling_finish_callback(self, modeling_name):
        node_type = 'modeling'
        if self.tree_dat.is_exist(node_type, modeling_name):
            QMessageBox().information(self, "提示", "该数据集已存在！")
            self.tree_root_nodes[node_type].setExpanded(True)
            return
        # dat_widge = DataWidget(type=data_type) 
        # self.tabwidget.addTab(dat_widge, f"{name}")
        # self.tabwidget.setCurrentWidget(dat_widge)
        self.tree_dat.add_node(parent_type=node_type, name=modeling_name)
        leaf_node = self.add_tree_node(node_type=node_type, node_name=modeling_name)
        self.tree_root_nodes[node_type].setExpanded(True)
        currNode = self.treeview.currentItem()
        if currNode:
            currNode.setSelected(False)
        leaf_node.setSelected(True)

    def center(self):
        qr = self.frameGeometry()  # 获得窗口
        cp = QDesktopWidget().availableGeometry().center()  # 获得屏幕中心点
        qr.moveCenter(cp)  # 显示到屏幕中心
        self.move(qr.topLeft())

    def loadTree(self):
        self.tree_dat = TreeData.instance()
        self.treeview.clear()

        for inx in range(len(MainWindow.meta_node_types)):
            node_type = MainWindow.meta_node_types[inx]
            root_node = QTreeWidgetItem(self.treeview)
            root_node.setText(0, MainWindow.meta_node_names[inx])
            root_node.setIcon(0, QIcon(f'{self.absolute_path}/res/{MainWindow.meta_node_icos[inx]}'))
            self.tree_root_nodes[node_type] = root_node
            if node_type in self.tree_dat.datas:
                model_inxs = self.tree_dat.datas[node_type]
                for model_inx in model_inxs:
                    print(model_inx)
                    node_name = model_inx['name']
                    node_dat_file = model_inx['data_file_name']
                    data_type = model_inx['data_type']
                    self.add_tree_node(node_type, node_name, node_dat_file, data_type)

    def add_tree_node(self, node_type, node_name, node_dat_file='', data_type='') -> QTreeWidgetItem:
        root_node = self.tree_root_nodes[node_type]
        leaf_node = QTreeWidgetItem(root_node)
        leaf_node.setText(0, node_name)
        leaf_node.setText(1, node_dat_file)
        leaf_node.setText(2, node_type)
        leaf_node.setText(3, data_type)
        return leaf_node

        # node_brand.setIcon(0,QIcon(f'{self.absolute_path}/res/data.png'))

    def onTreeClicked(self, qmodeLindex):
        item = self.treeview.currentItem()
        name = item.text(0)
        dat_file_name = item.text(1)
        node_type = item.text(2)
        data_type = item.text(3)
        if name and node_type:
            self.showInTab(name, node_type, data_type, dat_file_name)

    def rightClickMenu(self, pos):
        currNode = self.treeview.currentItem()
        name = currNode.text(0)
        dat_file_name = currNode.text(1)
        node_type = currNode.text(2)
        data_type = currNode.text(3)
        if currNode.parent():
            seq_inx = currNode.parent().indexOfChild(currNode)
            if name and node_type:
                reply = QMessageBox.question(self, 'Message', f'确定删除{name}?', QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.No)
                if reply == QMessageBox.Yes:
                    try:
                        parent1 = currNode.parent()
                        parent1.removeChild(currNode)
                        self.tree_dat.remove_node(node_type, int(seq_inx))  # 删除数据目录
                        if node_type == 'data':
                            DataWidget.clear(data_type)  # 删除数据文件
                        if node_type == 'modeling':
                            self.model_clear(name)
                    except Exception as e:
                        print(e)

    # 删除目录中的model
    def model_clear(self, name):
        path = f'{absolute_path}/dat/model/{name}/'
        shutil.rmtree(path)

    def showInTab(self, name, node_type, data_type, dat_file_name):
        print("showInTab: {},{},{}".format(name, node_type, data_type, dat_file_name))

        count = self.tabwidget.count()
        for inx in range(count):
            str = self.tabwidget.tabText(inx)
            if str == name:
                self.tabwidget.setCurrentIndex(inx)
                return

        if node_type == 'data':
            dc_widge = DataWidget(type=data_type)
            self.tabwidget.addTab(dc_widge, f"{name}")
            self.tabwidget.setCurrentWidget(dc_widge)
        elif node_type == 'modeling':
            print("modeling")
            # md_widget = QtWidgets.QWidget()
            md_widget = Ui_widget_5(name)
            # md_ui.setupUi(md_widget)
            self.tabwidget.addTab(md_widget, f"{name}")
            print(name)
            self.tabwidget.setCurrentWidget(md_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())
