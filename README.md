# PSPNet_VOC2012_AUG

# 环境配置：

python == 3.8

tensorflow == 2.4.1

keras == 2.4.3

如果想切换到1.6.0的tensoflow版本，匹配python==3.6，需要在代码这个tf命令中进行修改：

tf.image.resize(x, (K.int_shape(y)[1], K.int_shape(y)[2]))改成tf.image.resize_images

# 运行：

直接运行main.py即可。

1、img文件夹：原始数据集，内含11355张rgb图片，20类目标 + 1类背景。

2、cls文件夹：原始数据集，内含11355个语义分割.mat文件，标记信息用数字0-21表示。

数据集下载：https://blog.csdn.net/Twilight737?spm=1018.2226.3001.5343&type=download

3、demo文件夹：test_data的效果演示图，可以看出训练出来的非常不错，所有像素点分类精度接近93%。

4、download_weights.h5：迁移学习，网上下载backbone的部分权重。

5、mobile_netv2.py：特征提取网络，PSPNet的backbone部分。

6、psp_model.py：PSPNet整体结构。

7、Logs文件夹：记录每大轮下每个epoch训练好的权重文件。

# 实验效果：

最佳训练权重 best_val_accuracy = 0.92490，测试集图片中所有像素点的分类精度可达93%。

VOC12_AUG数据集，语义分割效果较佳。

