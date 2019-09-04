# -*- coding: utf-8 -*-
"""
@author: KLY
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']#有这个画图的时候才能显示中文
# In[1]:
# 每个图片样本是矩阵，pca处理的样本是向量，因此需要将图片转换为向量
def img2vector(image):
    img = cv2.imread(image, 0)  # 读取图片
    rows, cols = 112,92  
    img=cv2.resize(img,(cols,rows)) #将图片转换为同等大小
    imgVector = np.zeros((1, rows * cols))
    imgVector = np.reshape(img, (1, rows * cols))#使用imgVector变量作为一个向量存储图片矢量化信息，初始值均设置为0
    return imgVector
# In[2]:
# 读入人脸库,每个人随机选择k张作为训练集,其余构成测试集
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

# In[3]:
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

# 定义LDA算法
def LDA(X_train, y_train, k):

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

    return np.dot(evecs.T, X_train.T).T, evecs
def cosdist(A,B):#余弦相似度    # 参考于  https://blog.csdn.net/chary8088/article/details/74580238
    num = np.dot(A,B) #若为行向量则 A * B.T
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom #余弦值
    sim = 0.5 + 0.5 * cos #归一化
    return sim
# In[4]:
# 测试整个测试集的精准度
print('提取文件')
print('——————————————————————————————————————')

orlpath = "ORL/ORL"
r=10                         # 可以分别设置r（降到多少维)
result=[]
train_face, train_label, test_face, test_label,image_path = load_orl(7)  # 得到数据集
num_train = train_face.shape[0]  # 训练脸总数
num_test = test_face.shape[0]  # 测试脸总数
# 直接计算原始数据的LDA速度特别慢，而且容易出现奇异值，因此我们先用pca降到279，在此基础上用lda将原始数据降到r维
# 文件->LDA测试.py，有直接lda与pca+lda的对比

# 利用PCA算法将训练样本降到r维
data_train_new, data_mean, V_r = PCA(train_face, 279)
# 利用训练集的矩阵变换文件，将测试集样本进行pca降维
temp_face = test_face - np.tile(data_mean, (num_test, 1))
data_test_new = temp_face * V_r  # 得到测试脸在特征向量下的数据
#将得到的数据集转换为np数组，以便于后续处理
data_test = np.array(data_test_new)  # mat change to array
data_train= np.array(data_train_new)
# lda
data_train_new,evecs=LDA(data_train, train_label, r)
data_test_new=np.dot(evecs.T, data_test.T).T

print('计算测试集准确率')
print('——————————————————————————————————————')

# 测试准确度
true_num = 0

for i in range(num_test):
    testFace = data_test_new[i, :]
    #计算测试脸到每一张训练脸的距离
    diffMat = data_train_new - np.tile(testFace, (num_train, 1))  
    sqDiffMat = diffMat ** 2                                  
    sqDistances = sqDiffMat.sum(axis=1)  # 采用欧式的是欧式距离
    
    
    sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
    indexMin = sortedDistIndicies[0]  # 距离最近的索引
    result.append(train_label[indexMin])
    if train_label[indexMin] == test_label[i]:
        true_num += 1
    else:
        pass
    
    accuracy = float(true_num) / num_test
plt.figure()
plt.scatter(np.linspace(0,num_test,num_test),result,c='red',label='预测标签')
plt.scatter(np.linspace(0,num_test,num_test),test_label,label='测试集真实标签')
plt.legend()
plt.title('测试集分类结果')
plt.show()


print('当降维到%d时,测试集人脸识别精度为: %.2f%%' % (r,accuracy * 100))
# In[5]:
'''
#选择两张图片进行对比，看是否为同一个人的照片
#目标脸
root = tk.Tk()
root.withdraw()
file_path1 = filedialog.askopenfilename()# 选择图片

#==============================================================================
# file_path1="ORL/ORL/s1/1.jpg"
#==============================================================================
label1=int(file_path1.split('/')[-2][1:]) # 目标脸所属类别
img_object=cv2.imread(file_path1)# 读取图片

fig=plt.figure()
a=fig.add_subplot(1,2,1)
a.imshow(img_object)#显示图片
plt.xlabel('目标脸')
# In[6]:
# 待测脸
root = tk.Tk()
root.withdraw()
file_path2 = filedialog.askopenfilename()# 选择图片
#==============================================================================
# file_path2="ORL/ORL/s1/10.jpg"
#==============================================================================
img_object=cv2.imread(file_path2)# 读取图片，依次对待测图片进行pca与lda
test_face = img2vector(file_path2)
temp_face = test_face - data_mean
data_test_new = temp_face * V_r  # 得到测试脸在特征向量下的数据
data_test_new = np.array(data_test_new)  # mat change to array

data_test_new=np.dot(evecs.T, data_test_new.T).T


#计算待测脸到每一张训练脸的距离。来判断其所属类别
diffMat = data_train_new - np.tile(data_test_new, (num_train, 1))  
sqDiffMat = diffMat ** 2                                  
sqDistances = sqDiffMat.sum(axis=1)  # 采用欧式的是欧式距离
sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
indexMin = sortedDistIndicies[0]  # 距离最近的索引
label2=train_label[indexMin]



a=fig.add_subplot(1,2,2)
a.imshow(img_object)#显示图片
plt.xlabel('待测脸')
if label1==label2:
    plt.suptitle('匹配成功，是同一个人')
    print('匹配成功，是同一个人')
else:
    plt.suptitle('匹配失败，不是同一个人')
    print('匹配失败，不是同一个人')
'''

# In[5]:
#选择两张图片进行对比，看是否为同一个人的照片
#目标脸
root = tk.Tk()
root.withdraw()
file_path1 = filedialog.askopenfilename()# 选择图片

img_object=cv2.imread(file_path1)# 读取图片
test_face1 = img2vector(file_path1)

fig=plt.figure()
a=fig.add_subplot(1,2,1)
a.imshow(img_object)#显示图片
plt.xlabel('目标脸')
# In[6]:
# 待测脸
root = tk.Tk()
root.withdraw()
file_path2 = filedialog.askopenfilename()# 选择图片

img_object=cv2.imread(file_path2)# 读取图片，依次对待测图片进行pca与lda
test_face2 = img2vector(file_path2)
test_face=np.vstack((test_face1,test_face2))

temp_face = test_face - data_mean
data_test_new = temp_face * V_r  # 得到测试脸在特征向量下的数据
data_test_new = np.array(data_test_new)  # mat change to array

data_test_new=np.dot(evecs.T, data_test_new.T).T


#计算待测脸到每一张训练脸的距离。来判断其所属类别

distance=cosdist(data_test_new[0,:],data_test_new[1,:])#计算两张图片之间的相似度，相似度大于alpha时，判定两张图片输入同一类




alpha=0.7
a=fig.add_subplot(1,2,2)
a.imshow(img_object)#显示图片
plt.xlabel('待测脸')
if distance>alpha:
    plt.suptitle('匹配成功，是同一个人')
    print('匹配成功，是同一个人')
else:
    plt.suptitle('匹配失败，不是同一个人')
    print('匹配失败，不是同一个人')



