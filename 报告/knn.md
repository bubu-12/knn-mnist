# KNN实现手写数字识别

姓名：###

学号：***

时间：2023/9/15

## 1. KNN算法

### 1.1 KNN算法介绍

KNN即**k-nearest neighbor，是一种基本分类方法，也称k-近邻法。**

定义：如果一个样本在特征空间中的k个最相似（即特征空间最邻近）的样本中的大多数属于某一个类别，则该样本也属于这个类别。

### 1.2 数学模型

输入：训练集train， 测试集test。

输出：test中每一个实例所属类别，总正确率 = 预测正确数 / test总规模。

## 2. 数据集介绍

本次使用的数据集为机器学习中常见的MNIST手写数据集，地址：[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)， 示例见下图。训练集共60000张照片，每张为2828像素。每一个像素点取值介于0-255之间，0为纯黑，255为纯白，下图由28*28的二维灰度矩阵决定。

![本地图片](train_10.png)

## 3. KNN实现

### 3.1 思路介绍

本实验采取的距离度量为欧氏距离。对于每一个test实例，分别计算它与train中每一个实例的欧式距离，取最近的前k个，对它们标签进行投票，test预测值即票数最多的标签。最后与test给出的标签进行比较，得到预测正确数。

### 3.2 具体实现

**数据集处理**

样本的每一个实例均为28*28的二维矩阵，为方便欧氏距离的计算，将样本展开为长度为784的一维向量。同时对数据进行归一化处理，对样本空间内每一个向量，同一位置处采取Z-Score标准化（标准差标准化）处理。公式：

$X_{\text{normalized}} = \frac{X - \mu}{\sigma}$

其中$\mu$是均值，$\sigma$是标准差。从而将数据缩放为具有均值为0和标准差为1的标准正态分布。

**距离计算**

对于样本实例P = (x1, y1, z1, ...) , Q = (x2, y2, z2, ...)。

欧氏距离：

$d_{\text{Euclidean}} = \sqrt{{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2 + \ldots}}$

曼哈顿距离：

$d_{\text{Manhattan}} = |x_2 - x_1| + |y_2 - y_1| + |z_2 - z_1| + \ldots $

**k的选取**

本实验k首先在较大范围内，较大跨度进行实验，同时为减少时间，取test前1000个数据进行实验。随后确定k在一个较小范围内有预测表现较好，再缩小范围，找到较好的k值。

## 4. 实验结果及分析

### 4.1 实验结果

距离公式选取欧氏距离，k在(5, 30, 2)，即跨度为2，范围为5-30的区间内遍历的结果如下图。

![本地图片](k_5_30.png "size = 1000，k = (3, 30 ,2)")

距离选择曼哈顿距离，k在(3, 21, 2)，即跨度为2，范围为3-21的区间内遍历的结果如下图。

![本地图片](manhattan_k_3_21.png)

由上可知，距离公式选取欧氏距离表现更好。以下结果均是采用欧氏距离。

可以观察到，k在7附近模型表现较好，故将k在(3,10)遍历寻优，结果如下图。

![本地图片](k_3_10.png "size = 1000, k = (3, 10)")

### 4.2 结果分析

最终，k = 7的时候模型表现最好，将测试集全部代入，求得最终结果96.94%。 没有选择4的原因是防止模型过拟合。

## 5. 总结与反思

### 5.1 本次实验遇到问题

1. python的for循环运行速度太慢，需要利用numpy的广播性，减少对for循环的使用。同时，为防止numpy太大，将test分块处理，每次取出100个样本进行预测，最终取平均值。
2. 数据的预处理很重要，在未进行归一化之前，模型结果在0.27左右，表现异常糟糕。而Z-Score标准化在保留数据分布信息的前提下，减少了异常值的影响， 提高了模型的收敛速度与稳定性，同时降低过拟合风险。
3. 训练集和测试集规模较大，可先取部分测试集数据进行处理，得到k表现较好的值，然后对测试集全体进行预测。

### 5.2 反思

1. knn算法要对每一个test样本，求出其与所有训练样本的距离，计算非常慢。本实验中，将test全部预测需要36min，这个时间还会随着训练数据的规模增大，数据的维度增大而递增。大规模的数据，应当考虑选择其它分类方式。
2. knn对于特征的尺度较敏感，因此需要先对数据进行标准化、归一化处理，保证各个维度上对距离的影响是均衡的。做实验时对于此问题较晚意识到，造成时间上的浪费。
3. 距离的选择以及k的选择很重要。可以看出，曼哈顿距离的表现弱于欧几里得距离，而后者的计算复杂性更高。同时，k值的表现在大于7之后开始下降。因此，knn实现过程中要首先通过选取部分test数据，选择表现更好的距离计算方式以及k值。

## 6 代码

运行环境

* python 3.9
* numpy 1.23.5
* tqdm 4.64.1

```python
import numpy as np
from tqdm import tqdm


# 读取.idx3-ubyte 文件
def load_idx3_data(file_path):
    with open(file_path, 'rb') as f:
        # 读取文件头信息
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # 读取图像数据
        image_data = np.fromfile(f, dtype=np.uint8)

        # 将二维矩阵平铺为一维向量
        image_data = image_data.reshape(num_images, num_rows*num_cols)

    return image_data


# 读取.idx1-ubyte 文件
def load_idx1_data(file_path):
    with open(file_path, 'rb') as f:
        # 读取文件头数据
        magic_number = int.from_bytes(f.read(4), 'big')
        num_items = int.from_bytes(f.read(4), 'big')

        # 读取标签数据
        label_data = np.fromfile(f, dtype=np.uint8)

    return label_data


# 数据归一化处理
def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalize_data = (data - mean) / std

    return normalize_data


# 曼哈顿距离计算
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


# 欧氏距离计算
def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


# knn实现， batch_size为每次取出测试数据的规模
def knn(train_images, test_images, train_labels, test_labels, k, batch_size=100):
    num_test = len(test_images)
    accuracy_result = []

    for batch_start in tqdm(range(0, num_test, batch_size), desc='out'):
        batch_end = min(batch_start + batch_size, num_test)
        test_batch = test_images[batch_start:batch_end]

        batch_accuracies = []
        i = 0
        for test_sample in test_batch:
            # 计算测试样本与所有训练样本的曼哈顿距离
            # distances = np.sum(np.abs(train_images - test_sample), axis=1)
            distances = np.array([euclidean_distance(test_sample, train_sample) for train_sample in train_images])

            # 找到K个最近邻的训练样本的索引
            nearest_neighbors_indices = np.argpartition(distances, k)[:k]

            # 获取K个最近邻的训练标签
            nearest_labels = train_labels[nearest_neighbors_indices]

            # 对K个最近邻的训练标签进行投票
            predicted_label = np.argmax(np.bincount(nearest_labels))
            # print(nearest_labels, test_labels[batch_start + i])
            # 检查预测是否正确
            correct = predicted_label == test_labels[batch_start + i]
            i += 1
            batch_accuracies.append(correct)

        # 计算每个批次的准确率
        batch_accuracy = np.mean(batch_accuracies)
        accuracy_result.append(batch_accuracy)

    return np.mean(accuracy_result)


if __name__ == "__main__":
    # 读取图像数据文件
    train_image = load_idx3_data('train-images.idx3-ubyte')
    test_image = load_idx3_data('t10k-images.idx3-ubyte')

    # 图像数据归一化
    nor_data = normalize(np.concatenate((train_image, test_image), axis=0))
    nor_train, nor_test = nor_data[:60000], nor_data[60000:70000]

    # 读取数值数据
    train_data = load_idx1_data('train-labels.idx1-ubyte')
    test_data = load_idx1_data('t10k-labels.idx1-ubyte')

    result = []
    for k in range(3, 10):
        p = knn(nor_train, nor_test, train_data, test_data, k)
        print(k, ':', p)
        result.append(p)

    print(result)
```
