# 交通标志识别_CNN卷积神经网络实现)
此项目来自优达学城-自动驾驶车辆课程项目，实现方法通过卷积神经网络（LeNet）算法。

整个项目分为以下几部分：
---
- 1.数据导入与分析
- 2.搭建卷积神经网络模型实现
- 3.测试集上验证识别效果
- 4.可视化显示不同卷积层结果

## 1.数据导入与分析

本项目用到的交通标图片来源于网站：[German Traffic Sign Benchmarks](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

项目中用到的图片压缩包文件下载地址为，需要下载并解压：
- https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
- https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
- https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip


### 训练集数据文件夹结构

解压GTSRB_Final_Training_Images.zip文件到指定位置。得到文件夹结构：...\GTSRB\Final_Training\Images\...
训练集（Training）中图片总共有43种交通标志，对应解压后的...\Images文件夹下的43个子文件夹。每一个文件夹内的图片(.ppm格式)对应一种类型的交通标志（例如stop single），同时每一个文件夹内有一个.CSV问价存储记录了图片文件的相关信息。文件夹结构如下：

```
Training_Images
    + GTSRB
     + Final_Training
        + Images
            + 00000
                + 00000_00000.ppm
                + 00000_00001.ppm
                ...    
                + GT-00000.csv
            + 00001
                + 00000_00000.ppm
                + 00000_00001.ppm
                ...    
                + GT-00001.csv
            ...
```
注意：所有的图片格式为[PPM](https://blog.csdn.net/kinghzkingkkk/article/details/70226214)格式。需要借助Python的 `matplotlib` 与 `pillow` 库进行图片处理。如果想直接打开查看图片可能需要借助[其他软件](http://www.4qx.net/Extension_DaQuan.php?name=ppm)。

## 2.模型实现
### LeNet-5 模型架构

这里用到的时是[LeNet-5](http://www.tensornews.cn/lenet/)模型。它是第一个成功应用于数字识别问题的卷积神经网络。LeNet-5模型结构图如下：

![LeNet](/lenet.png)

来源: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

针对本项目，这里用到的模型在在原始LeNet模型基础上有调整：

#### 输入

输入为32x32x3(RGB - 3通道)图片

#### 架构
第一层：
- 卷积，输出节点矩阵为28x28x6;
- 激活函数：ReLU；
- 池化，过滤器大小为2x2,长宽步长为2，输出矩阵大小为14x14x6

第二层：
- 卷积，输出节点矩阵为10x10x16;
- 激活函数：ReLU；
- 池化，过滤器大小为2x2,长宽步长为2，输出矩阵大小为5x5x16
- Flatten

第三层：
- 全连接（Fully Connected），输出节点个数为120;
- 激活函数：ReLU；

第四层：
- 全连接（Fully Connected），输出节点个数为84;
- 激活函数：ReLU；

第五层：
- 全连接（Fully Connected）.输出节点个数为43

#### 输出

第二次全连接之后的43种交通标志分类

|Layer                       | Shape    |
|----------------------------|:--------:|
|Input                       | 32x32x3  |
|Convolution (valid, 5x5x6)  | 28x28x6  |
|Max Pooling (valid, 2x2)    | 14x14x6  |
|Activation  (ReLU)          | 14x14x6  |
|Convolution (valid, 5x5x16) | 10x10x16 |
|Max Pooling (valid, 2x2)    | 5x5x16   |
|Activation  (ReLU)          | 5x5x16   |
|Flatten                     | 400      |
|Dense                       | 120      |
|Activation  (ReLU)          | 120      |
|Dense                       | 43       |
|Activation  (Softmax)       | 43       |

## 3.实现效果验证分析
导入测试数据集之外新的图片验证识别准确率
### 测试集数据

测试图片位于文件夹Test_Images/GTSRB/Final_Test内
```
Test_Images
    +GTSRB
         + Final_Test
            + Images
                + 00000.ppm
                + 00001.ppm
                + ...
                + GT-final_test.csv      # 扩展的注释，包括图片分类id
                + GT-final_test.test.csv
```

GT-final_test.csv文件是单独下载，包含测试集图片label信息

## 4.可视化
可视化显示CNN神经网络不同层的输出

第一层：
![conv1](/conv1.jpg)
第二层：
![conv2](/conv2.jpg)
