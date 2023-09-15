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
            distances = np.array([manhattan_distance(test_sample, train_sample) for train_sample in train_images])

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
    for k in range(3, 4):
        p = knn(nor_train, nor_test[:1000], train_data, test_data[:1000], k)
        print(k, ':', p)
        result.append(p)

    print(result)