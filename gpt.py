import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier

def create_data(n):
    # 随机生成正弦波
    x = np.linspace(0, 10, n)
    sin_y = np.sin(x) + random.uniform(-0.5, 0.5)
    # 随机生成直线
    line_y = x + random.uniform(-0.5, 0.5)
    # 组合成数据集
    X1 = np.concatenate((x.reshape(-1, 1), sin_y.reshape(-1, 1)), axis=1)
    X2 = np.concatenate((x.reshape(-1, 1), line_y.reshape(-1, 1)), axis=1)
    X = np.concatenate((X1, X2), axis=0)
    # 标记为正弦波和直线，正弦波为0，直线为1
    y = np.concatenate((np.zeros(n), np.ones(n)), axis=0)
    return X, y

def train_test_split(X, y, ratio):
    n = X.shape[0]
    # 计算分割的索引
    split_index = int(n * ratio)
    # 分割训练集和测试集
    X_train = X[:split_index, :]
    X_test = X[split_index:, :]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    n = 10000
    X, y = create_data(n)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_predict = knn.predict(X_test)

    accuracy = np.mean(y_predict == y_test)
    print("Accuracy:", accuracy)
