import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 读取图像数据并预处理
def preprocess_image(img):
    # 灰度处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 模糊处理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    edges = cv2.Canny(blurred, 30, 150)
    # 获取特征向量
    features = edges.flatten()
    return features

# 读取图像数据
images = [cv2.imread('image1.jpg'), cv2.imread('image2.jpg'), ...]
labels = [1, 2, ...] # 图像的标签

# 预处理数据并转换为特征向量
X = [preprocess_image(img) for img in images]

# 将数据分为训练集和测试集
X_train = X[:int(len(X) * 0.8)]
y_train = labels[:int(len(labels) * 0.8)]
X_test = X[int(len(X) * 0.8):]
y_test = labels[int(len(labels) * 0.8):]

# 使用KNN算法进行训练
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# 使用测试数据评估模
