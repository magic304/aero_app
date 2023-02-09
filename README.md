# aero_app

# 气动力数据/建模平台

## 介绍

用于可视化气动力数据建模的pyqt程序，算法使用LSTM神经网络进行气动力数据建模，并集成了迁移学习算法，也可用于一般的非线性拟合任务使用。功能包括数据上传、数据合并、数据下载、模型超参数设置、动态展示收敛曲线、模型预测、预测结果可视化等。

![动画](https://s2.loli.net/2023/02/09/N5FHyUvrtgEZxRW.gif)

![动画1](https://s2.loli.net/2023/02/09/YHMsTf8hSypnZb6.gif)

## 软件架构

- 建模算法基于TensorFlow、Keras深度学习框架
- 可视化界面基于Pyqt5实现
- 绘图功能基于Matpoltlib

## 使用说明

1. 将项目下载至本地
2. 运行app.py文件

## 功能介绍

### 一、数据功能

- 点击工具栏的数据功能按钮，输入新数据集的名称，点击`ok`创建一个新的数据集：

  ![image-20230209171813334](https://s2.loli.net/2023/02/09/cXSseCdBAlzDn7H.png)

  ![image-20230209171859877](https://s2.loli.net/2023/02/09/3zdhqsgN92LmPaT.png)

- 创建完成，点击创建的数据集，点击右上角的`浏览/上传`，上传所需数据文件，目前仅支持csv文件。可以选择多个csv文件，点击合并入库后会将数据合并在一起：

  ![image-20230209172120432](https://s2.loli.net/2023/02/09/aODr9Jnhx3sC4wd.png)

- 右键数据集可进行删除操作。

### 二、建模功能

- 点击主界面上方建模模块按钮，弹出建模窗口：

  ![image-20230209194516089](https://s2.loli.net/2023/02/09/y31toNB56FfbTW7.png)

- 进行建模工作前，用户需要对建模参数进行设置。若已存在与模型名称相同的模型，会弹出‘模型名称重复’提醒：

  ![image-20230209194619843](https://s2.loli.net/2023/02/09/y1mUDpPASTeaGbt.png)

  ![image-20230209194718436](https://s2.loli.net/2023/02/09/GcBHzuM2WOiDmeQ.png)

- 参数设置好后，点击`开始训练`进行建模。建模过程中可以动态观察网络训练的收敛曲线：

  ![动画](https://s2.loli.net/2023/02/09/N5FHyUvrtgEZxRW.gif)

- 模型会自动保存到系统中。并在左侧显示。点击左侧已训练好的模型，进行模型预测功能：

  ![image-20230209200256482](https://s2.loli.net/2023/02/09/jVSoWRFe4wJagUT.png)

- 上传需要预测的数据，点击开始预测，等待弹出预测完成：

  ![image-20230209200401585](https://s2.loli.net/2023/02/09/5PtZaqK6InFcAw7.png)

- 点击`ok`,会显示预测后的数据表，上方会有筛选功能：

  ![image-20230209200501402](https://s2.loli.net/2023/02/09/CeuBxZL6pwUIj32.png)

- 可以点击右下角将预测结果下载。如需可视化，点击可视化展示按钮，弹出可视化设置弹窗：

  ![image-20230209200612373](https://s2.loli.net/2023/02/09/dfrE29MJI1SBhtk.png)

- 绘图效果图下

  - 二维绘图设置（不分组）及绘图效果

    ![image-20230209200839679](https://s2.loli.net/2023/02/09/EoityPlhgBN3cAY.png)

    ![image-20230209200858400](https://s2.loli.net/2023/02/09/7xMcbQUpfG9jnih.png)

  - 二维绘图设置（分组）及绘图效果

    ![image-20230209201035003](https://s2.loli.net/2023/02/09/sbKAjaI6rXiW4ZP.png)

    ![image-20230209200949431](https://s2.loli.net/2023/02/09/JP9RFmQyt71h8Wz.png)

  - 三维绘图设置及效果（三维绘图需要分组）

    ![image-20230209201050578](https://s2.loli.net/2023/02/09/a1RP79D8orEuwIS.png)

    ![动画1](https://s2.loli.net/2023/02/09/YHMsTf8hSypnZb6.gif)

## 其他

如果需要更改模型，在`aero_app`->`aero_model`的`PreTrain.py`(直接训练/预训练模型)或`Transfer.py`（迁移学习模型）中修改网络模型即可。