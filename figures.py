import matplotlib
import numpy as np
import pandas as pd
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 解决坐标轴中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号不显示的问题


class Data_Table(QTableWidget):
    def __init__(self, parent=None, titles=None):
        super().__init__(parent=parent)
        if titles:
            self.setColumnCount(len(titles))
            self.setHorizontalHeaderLabels(titles)
        # self.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)  
        self.checkbox_array = []  # 存放checkbox
        self.checkbox_array_index = []  # 存放选中index
        self.checkbox_array_id = []  # 存放待筛选消费者id

    def get_select(self):
        colomn = self.columnCount()
        row_list = set()
        for i in self.selectionModel().selection().indexes():
            row_list.add(i.row())
        # print(row_list)
        select_data = []
        for row in row_list:
            row_data = [self.item(row, p).text() for p in range(colomn)]
            select_data.append(row_data)
        print(select_data)

    def set_data(self, table_data, table_header=None, vertical_header=None):
        self.clearContents()
        input_table_rows = table_data.shape[0]
        input_table_colunms = table_data.shape[1]

        # 读取表格，转换表格,给tablewidget设置行列表头
        self.setColumnCount(input_table_colunms)
        self.setRowCount(input_table_rows)
        if table_header:
            self.setHorizontalHeaderLabels(table_header)
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 给tablewidget设置行列表头
        if vertical_header:
            self.setVerticalHeaderLabels(vertical_header)
        # 遍历表格每个元素，同时添加到tablewidget中
        for i in range(input_table_rows):
            for j in range(input_table_colunms):
                input_table_items = f'{table_data[i, j]}'
                if isinstance(table_data[i, j], float):
                    input_table_items = f'{table_data[i, j]: .4}'
                newItem = QTableWidgetItem(input_table_items)
                newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.setItem(i, j, newItem)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        print(self.viewport().size())
        # if self.width() > self.viewport().size().width():
        #     self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        #     print(self.width(),self.viewport().size().width())
        # else:
        #     self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def set_data_df(self, df: pd.DataFrame, table_header=None, vertical_header=None):
        table_data = df.values
        if table_header is None:
            table_header = df.columns.to_list()
        if vertical_header is None:
            vertical_header = [f"{inx}" for inx in df.index]  # df.index.to_list()
        self.set_data(table_data, table_header=table_header, vertical_header=vertical_header)

    def append_row(self, row_dat):
        currentRowCount = self.rowCount()  # necessary even when there are no rows in the table
        self.insertRow(currentRowCount)
        for i in range(0, len(row_dat)):
            # self.insertRow(currentRowCount, i, QTableWidgetItem(f"{row_dat[i]}"))
            self.setItem(currentRowCount, i, QTableWidgetItem(f"{row_dat[i]}"))
        # self.render()

    def set_data_checkbok(self, table_data, table_header=None, vertical_header=None, check_index=None):
        self.clearContents()
        input_table_rows = table_data.shape[0]
        input_table_colunms = table_data.shape[1]
        self.checkbox_array_id = np.array(check_index.index)
        # 读取表格，转换表格,给tablewidget设置行列表头
        self.setColumnCount(input_table_colunms)
        self.setRowCount(input_table_rows)
        if table_header:
            self.setHorizontalHeaderLabels(table_header)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 给tablewidget设置行列表头
        if vertical_header:
            self.setVerticalHeaderLabels(vertical_header)
            self.verticalHeader().setVisible(False)  # 隐藏表头
            for i in range(len(vertical_header)):
                self.checkbox = QCheckBox()
                self.checkbox_array.append(self.checkbox)
                self.setCellWidget(i, 0, self.checkbox)
            # 遍历表格每个元素，同时添加到tablewidget中
        for i in range(input_table_rows):
            for j in range(input_table_colunms):
                if (isinstance(table_data[i, j], str)):
                    input_table_items = f'{table_data[i, j]}'
                else:
                    input_table_items = f'{table_data[i, j]:.4f}'
                # input_table_items = f'{table_data[i, j]: .4f}'
                newItem = QTableWidgetItem(input_table_items)
                newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.setItem(i, j, newItem)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()

    def checkbox_select(self):
        checkboxs = {}
        for i in range(len(self.checkbox_array)):
            checkboxs[i] = self.checkbox_array[i].isChecked()
        for j in checkboxs:
            if checkboxs[j]:
                self.checkbox_array_index.append(self.checkbox_array_id[j])
        print(self.checkbox_array_index)

    def set_picture(self, path=None):
        self.setRowCount(1)
        self.setColumnCount(1)
        self.setHorizontalHeaderLabels([""])
        self.setVerticalHeaderLabels([""])
        self.reset()
        pix = QPixmap(path)
        lab1 = QLabel(self)
        lab1.setPixmap(pix)
        self.setCellWidget(0, 0, lab1)
        self.resizeColumnToContents(0)
        self.resizeRowToContents(0)
        print("hllo")


class Latex_Canvas(FigureCanvas):
    def __init__(self, latex_str=None, parent=None, width=1, height=1):
        self.figure = Figure(figsize=(width, height), dpi=120)
        super(Latex_Canvas, self).__init__(self.figure)
        if latex_str:
            self.renderTex(latex_str)

    def renderTex(self, mathTex, fs=20):
        self.figure.patch.set_facecolor('none')
        # ---- plot the mathTex expression ----
        ax = self.figure.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_facecolor('none')

        t = ax.text(0, 0.1, mathTex, ha='left', va='bottom', fontsize=fs)
        # ---- fit figure size to text artist ----
        # renderer = self.get_renderer()
        # fwidth, fheight = self.figure.get_size_inches()
        # fig_bbox = self.figure.get_window_extent(renderer)
        # text_bbox = t.get_window_extent(renderer)
        # tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
        # tight_fheight = text_bbox.height * fheight / fig_bbox.height
        # self.figure.set_size_inches(tight_fwidth, tight_fheight)
        self.draw()


class Figure_Line(FigureCanvas):
    """ 创建折线画板类 """
    colors = ['red', 'black', 'blue', 'brown', 'green']

    def __init__(self, parent=None, width=4.3, height=3.8, title='曲线', xlabel='x轴', ylabel='y轴'):
        self.fig = Figure(figsize=(width, height))  # , dpi=120
        super(Figure_Line, self).__init__(self.fig)
        self.lines = {}
        self.ymin = None
        self.ymax = None
        self.xmin = None
        self.xmax = None
        self.ax = self.fig.add_subplot(111)  # 111表示1行1列，第一张曲线图
        self.ax.grid(True)  # 添加网格
        self.ax.set_title(title)  # 设置标题
        self.ax.set_xlabel(xlabel)  # 设置坐标名称
        self.ax.set_ylabel(ylabel)
        # self.setContentsMargins(20,20,20,20)
        # xticks(range(0,len(self.time_line),3),[self.time_line[i] for i in range(0,len(self.time_line),1) if i%3==0 ],rotation=90,fontsize=8)
        self.init_toolbar(self)
        self.fig.tight_layout(pad=1.2)

    def set_xticks(self, ticks=None, labels=None, rotation=None,
                   fontsize=None):  # range(0,len(self.time_line),3),[self.time_line[i] for i in range(0,len(self.time_line),1) if i%3==0 ],rotation=90,fontsize=8
        self.ax.set_xticks(ticks, labels, rotation=rotation, fontsize=fontsize)

    def init_toolbar(self, parent=None):
        self.naviBar = NavigationToolbar(self, parent, coordinates=False)  # 创建工具栏
        # self.naviBar.setSizePolicy(QSizePolicy.Policy.MinimumExpanding,QSizePolicy.Policy.Minimum)
        self.naviBar.setMaximumWidth(self.width() / 9)
        self.naviBar.setMovable(True)
        self.mpl_connect("scroll_event", self.do_scrollZoom)  # 支持鼠标滚轮缩放
        self.blitted_cursor = BlittedCursor(self.ax)
        self.mpl_connect('motion_notify_event', self.blitted_cursor.on_mouse_move)

    def do_scrollZoom(self, event):  # 通过鼠标滚轮缩放
        ax = event.inaxes  # 产生事件axes对象
        if ax == None:
            return
        self.naviBar.push_current()
        xmin, xmax = ax.get_xbound()
        xlen = xmax - xmin
        ymin, ymax = ax.get_ybound()
        ylen = ymax - ymin

        xchg = event.step * xlen / 20
        xmin = xmin + xchg
        xmax = xmax - xchg
        ychg = event.step * ylen / 20
        ymin = ymin + ychg
        ymax = ymax - ychg
        ax.set_xbound(xmin, xmax)
        ax.set_ybound(ymin, ymax)
        event.canvas.draw()

    def add_bar(self, name, x_data, y_data, color=None, newAx=None):
        if newAx:
            newAx = self.ax.twinx()  # instantiate a second axes that shares the same x-axis
            line = newAx.bar(x_data, y_data, color=color)
            self.lines[name] = line[0]
        else:
            line = self.ax.bar(x_data, y_data, color=color)
            self.lines[name] = line[0]
            # self.ax.add_line(line[0]) 
            if (x_data.size > 0 and y_data.size > 0):
                self.update_xylim(x_data, y_data)
            # 添加图例
        self.ax.legend(list(self.lines.values()), list(self.lines.keys()))
        self.draw()

    def add_line(self, name, x_data, y_data, style={'ls': '-', 'marker': None, 'color': None}, newAx=None):
        if newAx:
            newAx = self.ax.twinx()  # instantiate a second axes that shares the same x-axis
            line = newAx.plot(x_data, y_data, linestyle=style['ls'], marker=style['marker'], color=style['color'], markerfacecolor='white')
            self.lines[name] = line[0]
            newAx.set_ylabel(name, color=style['color'])
            from matplotlib import ticker
            newAx.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
            newAx.tick_params(axis='y', colors=style['color'])
            # newAx.add_line(line[0]) 
        else:
            line = self.ax.plot(x_data, y_data, linestyle=style['ls'], marker=style['marker'], color=style['color'], markerfacecolor='white')
            self.lines[name] = line[0]
            # self.ax.add_line(line[0]) 
            if (x_data.size > 0 and y_data.size > 0):
                self.update_xylim(x_data, y_data)
            # 添加图例
        self.ax.legend(list(self.lines.values()), list(self.lines.keys()), fontsize=8)
        self.draw()
        # self.ax.legend([self.line, self.line2], ['sinx', 'cosx'])  # 添加图例

    def updateData(self, name, x_data, y_data):
        if name in self.lines:
            line = self.lines[name]
            line.set_xdata(x_data)
            line.set_ydata(y_data)
        self.draw()

    def add_data(self, x_point, y_dict):  # x, y={name:value}
        for name in y_dict.keys():
            if name in self.lines:
                line = self.lines[name]
                x_data = np.append(line.get_xdata(), x_point)
                y_data = np.append(line.get_ydata(), y_dict[name])
                line.set_xdata(x_data)
                line.set_ydata(y_data)
                self.update_xylim(x_data, y_data)
        self.draw()

    # 更新图的x\y轴范围
    def update_xylim(self, x_data, y_data):
        if self.xmin is None:
            self.xmin = np.min(x_data)
        if self.xmax is None:
            self.xmax = np.max(x_data)
        self.xmin = min(self.xmin, np.min(x_data))
        self.xmax = max(self.xmax, np.max(x_data))
        self.ax.set_xlim(self.xmin - abs(0.05 * self.xmin), self.xmax + abs(0.05 * self.xmax))
        if self.ymin is None:
            self.ymin = np.min(y_data)
        if self.ymax is None:
            self.ymax = np.max(y_data)
        self.ymin = min(self.ymin, np.min(y_data))
        self.ymax = max(self.ymax, np.max(y_data))
        self.ax.set_ylim(self.ymin - abs(0.1 * self.ymin), self.ymax + abs(0.1 * self.ymax))

        # ymin=np.min(y_data)
        # self.ymin = self.ymin if  ymin>self.ymin else ymin   
        # ymax=np.max(y_data)
        # self.ymax = self.ymax if  ymax<self.ymin else ymax


class BlittedCursor:
    """
    A cross hair cursor using blitting for faster redraw.
    """

    def __init__(self, ax):
        self.ax = ax
        self.background = None
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.65, 0.95, '', transform=ax.transAxes)
        self._creating_background = False
        ax.figure.canvas.mpl_connect('draw_event', self.on_draw)

    def on_draw(self, event):
        self.create_new_background()

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def create_new_background(self):
        if self._creating_background:
            # discard calls triggered from within this function
            return
        self._creating_background = True
        self.set_cross_hair_visible(False)
        self.ax.figure.canvas.draw()
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.set_cross_hair_visible(True)
        self._creating_background = False

    def on_mouse_move(self, event):
        if self.background is None:
            self.create_new_background()
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.restore_region(self.background)
                self.ax.figure.canvas.blit(self.ax.bbox)
        else:
            self.set_cross_hair_visible(True)
            # update the line positions
            x, y = event.xdata, event.ydata
            self.horizontal_line.set_ydata(y)
            self.vertical_line.set_xdata(x)
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))

            self.ax.figure.canvas.restore_region(self.background)
            self.ax.draw_artist(self.horizontal_line)
            self.ax.draw_artist(self.vertical_line)
            self.ax.draw_artist(self.text)
            self.ax.figure.canvas.blit(self.ax.bbox)


# class SvgWidget(QSvgWidget):

#     def paintEvent(self, event: QtGui.QPaintEvent) -> None: 
#         renderer = self.renderer()
#         if renderer != None:
#             painter = QPainter(self)
#             size = renderer.defaultSize()
#             ratio = size.height()/size.width()
#             length = min(self.width(), self.height())
#             renderer.render(painter, QRectF(0, 0, length, ratio * length))
#             painter.end()
class RadarMap(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(RadarMap, self).__init__(self.fig)
        self.axes0 = self.fig.add_subplot(111, polar=True)

    def radarMap(self, values, feature, values_2, labels=['零售户均值', '零售户']):
        self.axes0.clear()

        maxValue = int(np.ceil(np.max(values_2) * 12)) / 10
        maxValue = max(1.2, maxValue)
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        values_2 = np.concatenate((values_2, [values_2[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        feature = np.concatenate((feature, [feature[0]]))

        line1, = self.axes0.plot(angles, values, 'o-', linewidth=2, label=labels[0])
        self.axes0.fill(angles, values, alpha=0.25)
        line2, = self.axes0.plot(angles, values_2, 's-', linewidth=2, label=labels[1])
        self.axes0.fill(angles, values_2, alpha=0.25)
        self.axes0.set_thetagrids(angles * 180 / np.pi, feature)
        self.axes0.set_ylim(0, maxValue)
        self.axes0.legend(handles=[line1, line2], labels=labels, loc=(0.9, .95))
        self.draw()


class PictureLabel(QtWidgets.QLabel):
    def __init__(self, text: str = None, pixmap: QtGui.QPixmap = None):
        super().__init__()
        self._pixmap = None
        text is not None and self.setText(text)
        pixmap is not None and self.setPixmap(pixmap)

    def setPixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self.repaint()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if self._pixmap is not None:
            imageWidth, imageHeight = self._pixmap.width(), self._pixmap.height()
            labelWidth = self.width()
            ratio = labelWidth / imageWidth
            # ratio = min(labelWidth / imageWidth, labelHeight / imageHeight)
            newWidth, newHeight = int(imageWidth * ratio), int(imageHeight * ratio)
            self.setFixedHeight(newHeight)
            newPixmap = self._pixmap.scaledToWidth(newWidth, Qt.TransformationMode.FastTransformation)
            x, y = abs(newWidth - labelWidth) // 2, abs(newHeight - newHeight) // 2
            QtGui.QPainter(self).drawPixmap(x, y, newPixmap)


from PyQt5.QtCore import Qt, QSortFilterProxyModel
from PyQt5.QtWidgets import QCompleter, QComboBox


class ExtendedComboBox(QComboBox):
    def __init__(self, parent=None):
        super(ExtendedComboBox, self).__init__(parent)

        self.setEditable(True)

        # add a filter model to filter matching items
        self.pFilterModel = QSortFilterProxyModel(self)
        self.pFilterModel.setFilterCaseSensitivity(Qt.CaseInsensitive)  # 设定为大小写不敏感
        self.pFilterModel.setSourceModel(self.model())

        # add a completer, which uses the filter model
        self.completer = QCompleter(self.pFilterModel, self)
        # self.completer.popup().setStyleSheet("font:30px ;")#通过pop.setstylesheet()函数设置弹窗样式，字体、背景色等等。
        # always show all (filtered) completions
        self.completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.setCompleter(self.completer)

        # connect signals
        self.lineEdit().textEdited.connect(self.pFilterModel.setFilterFixedString)
        self.completer.activated.connect(self.on_completer_activated)

    # on selection of an item from the completer, select the corresponding item from combobox
    def on_completer_activated(self, text):
        if text:
            index = self.findText(text)
            self.setCurrentIndex(index)
            self.activated[str].emit(self.itemText(index))

    # on model change, update the models of the filter and completer as well
    def setModel(self, model):
        super(ExtendedComboBox, self).setModel(model)
        self.pFilterModel.setSourceModel(model)
        self.completer.setModel(self.pFilterModel)

    # on model column change, update the model column of the filter and completer as well
    def setModelColumn(self, column):
        self.completer.setCompletionColumn(column)
        self.pFilterModel.setFilterKeyColumn(column)
        super(ExtendedComboBox, self).setModelColumn(column)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QMainWindow, QApplication, QGridLayout, QWidget
    import sys

    app = QApplication(sys.argv)
    mainMindow = QMainWindow()

    mainMindow.resize(500, 200)
    widget = QWidget()
    fig_layout = QGridLayout(mainMindow)
    widget.setLayout(fig_layout)
    mainMindow.setCentralWidget(widget)

    data_table = Data_Table()
    data = np.array([[1, 2, 3], [4, 5, 6]])
    data_table.set_data(table_data=data, table_header=['A', 'B', 'C'], vertical_header=['hell', 'wo'])
    fig_layout.addWidget(data_table)

    # latex_str = '$C_{soil}=(1 - n) C_m + \\theta_w C_w$'
    # latex_str = '$k_{soil}=\\frac{\\sum f_j k_j \\theta_j}{\\sum f_j \\theta_j}$'
    # latex_str = '公式\n $\\lambda_{soil}=k_{soil} / C_{soil}$'
    # latex_canvas = Latex_Canvas( parent=mainMindow)
    # latex_canvas.renderTex(latex_str)
    # fig_layout.addWidget(latex_canvas)

    LineFigure = Figure_Line(mainMindow)
    fig_layout.addWidget(LineFigure)
    # 准备数据，绘制曲线
    x_data = np.arange(-4, 4, 0.02)
    y_data = np.sin(x_data)
    y2_data = np.cos(x_data)
    LineFigure.add_line('sinx', x_data, y_data)
    LineFigure.add_line('cosx', x_data, y2_data)

    mainMindow.show()
    sys.exit(app.exec_())
