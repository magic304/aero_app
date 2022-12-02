import matplotlib
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 解决坐标轴中文显示问题
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号不显示的问题


class Figure_Line(FigureCanvas):
    """ 创建折线画板类 """
    colors = ['red', 'black', 'blue', 'brown', 'green']

    def __init__(self, parent=None, width=4.3, height=3.8, title='曲线', xlabel='x轴', ylabel='y轴'):
        self.fig = Figure(figsize=(width, height))  # , dpi=120
        super(Figure_Line, self).__init__(self.fig)
        self.lines = {}
        self.ymin = 0
        self.ymax = 1
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
        self.naviBar.setMaximumWidth(self.width() / 4)
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
            line = newAx.plot(x_data, y_data, linestyle=style['ls'], marker=style['marker'], color=style['color'])
            self.lines[name] = line[0]
            newAx.set_ylabel(name, color=style['color'])
            from matplotlib import ticker
            newAx.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
            newAx.tick_params(axis='y', colors=style['color'])
            # newAx.add_line(line[0])
        else:
            line = self.ax.plot(x_data, y_data, linestyle=style['ls'], marker=style['marker'], color=style['color'])
            self.lines[name] = line[0]
            # self.ax.add_line(line[0])
            if (x_data.size > 0 and y_data.size > 0):
                self.update_xylim(x_data, y_data)
            # 添加图例
        self.ax.legend(list(self.lines.values()), list(self.lines.keys()))
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
        self.ax.set_xlim(self.xmin, self.xmax)

        self.ymin = min(self.ymin, np.min(y_data))
        self.ymax = max(self.ymax, np.max(y_data))
        self.ax.set_ylim(self.ymin, self.ymax + 0.2 * self.ymax)
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
