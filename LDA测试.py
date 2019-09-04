import numpy as np
import os
from scipy.misc import imread

orlpath = "ORL/ORL"
import cv2
def img2vector(image):
    img = cv2.imread(image, 0)  # 读取图片
    rows, cols = img.shape  #获取图片的像素
    imgVector = np.zeros((1, rows * cols))
    imgVector = np.reshape(img, (1, rows * cols))#使用imgVector变量作为一个向量存储图片矢量化信息，初始值均设置为0
    return imgVector

def load_orl(k):#参数K代表选择K张图片作为训练图片使用
    '''
    对训练数据集进行数组初始化，用0填充，每张图片尺寸都定为112*92,
    现在共有40个人，每个人都选择k张，则整个训练集大小为40*k,112*92
    '''
    image_path=[]
    train_face = np.zeros((40 * k, 112 * 92))
    train_label = np.zeros(40 * k)  # [0,0,.....0](共40*k个0)
    test_face = np.zeros((40 * (10 - k), 112 * 92))
    test_label = np.zeros(40 * (10 - k))
    np.random.seed(0)
    sample = np.random.rand(10).argsort() + 1  # 随机排序1-10 (0-9）+1
    for i in range(40):  # 共有40个人
        people_num = i + 1
        for j in range(10):  # 每个人都有10张照片
            image = orlpath + '/s' + str(people_num) + '/' + str(sample[j]) + '.jpg'
            image_path.append(image)
            # 读取图片并进行向量化            
            img = img2vector(image)
            if j < k:
                # 构成训练集
                train_face[i * k + j, :] = img
                train_label[i * k + j] = people_num
            else:
                # 构成测试集
                test_face[i * (10 - k) + (j - k), :] = img
                test_label[i * (10 - k) + (j - k)] = people_num

    return train_face, train_label, test_face, test_label,image_path



def PCA(data, r):#降低到r维
    data = np.float32(np.mat(data))
    rows, cols = np.shape(data)
    data_mean = np.mean(data, 0)  # 对列求平均值
    A = data - np.tile(data_mean, (rows, 1))  # 将所有样例减去对应均值得到A
    C = A * A.T  # 得到协方差矩阵
    D, V = np.linalg.eig(C)  # 求协方差矩阵的特征值和特征向量
    V_r = V[:, 0:r]  # 按列取前r个特征向量
    V_r = A.T * V_r  # 小矩阵特征向量向大矩阵特征向量过渡
    for i in range(r):
        V_r[:, i] = V_r[:, i] / np.linalg.norm(V_r[:, i])  # 特征向量归一化

    final_data = A * V_r
    return final_data, data_mean, V_r


def LDA(X_train, y_train, X_test, k):

    m , n = X_train.shape
    class_wise_mean = []
    for i in range(1,41):
        idx = np.where(y_train==i)
        class_wise_mean.append(np.mean(X_train[idx], axis=0))

    within_SM = np.zeros((n, n))
    
    for i, mean_vector in zip(range(1, 41), class_wise_mean):
        class_wise_M = np.zeros((n, n))
        idx = np.where(y_train==i)
        for img in X_train[idx]:
            img, mean_vector = img.reshape(n, 1), mean_vector.reshape(n, 1)
            class_wise_M += (img - mean_vector).dot((img - mean_vector).T)
        within_SM += class_wise_M

    total_mean = np.mean(X_train, axis=0)
    between_SM = np.zeros((n, n))
    for i, mean_vector in enumerate(class_wise_mean):
        idx = np.where(y_train==i+1)
        cnt = X_train[idx].shape[0]
        mean_vector = mean_vector.reshape(n, 1)
        total_mean = total_mean.reshape(n, 1)
        between_SM += cnt * (mean_vector - total_mean).dot((mean_vector - total_mean).T)
    a=np.linalg.inv(within_SM).dot(between_SM)
    evals, evecs = np.linalg.eigh(a)
    idx = np.argsort(evals)[::-1][:k]
    evecs = evecs[:, idx]

    return np.dot(evecs.T, X_train.T).T, np.dot(evecs.T, X_test.T).T
# In[1]: 对原始数据先pca再lda
# PCA


# 利用训练集的矩阵变换文件，将测试集样本进行pca降维
training_data,  training_labels,testing_data, testing_labels,path = load_orl(7)
training_data,data_mean,V_r=PCA(training_data,279)

temp_face = testing_data - np.tile(data_mean, (testing_data.shape[0], 1))
testing_data = temp_face * V_r  # 得到测试脸在特征向量下的数据

# LDA
#LDA.LDA(training_data, training_labels, testing_data, testing_labels, 7)
train,test=LDA(training_data, training_labels, testing_data, 70)

# In[2]: 对原始数据直接lda
training_data,  training_labels,testing_data, testing_labels,path = load_orl(7)
train,test=LDA(training_data, training_labels, testing_data, 70)






