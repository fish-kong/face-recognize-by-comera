# -*- coding: utf-8 -*-
"""

@author: KLY
"""

from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tkinter as tk
from tkinter import filedialog
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']#有这个画图的时候才能显示中文



def img2vector(image):
    img = cv2.imread(image,0)  # 读取图片
    rows, cols = 112,92
    img=cv2.resize(img,(cols,rows)) #将图片转换为同等大小
    imgVector = np.zeros(( rows , cols))
    imgVector = np.reshape(img,  (rows , cols))#与pca lda不同的是，lbp处理的是图像矩阵，所以这里不将其变成向量
    return imgVector
# In[2]:
# 读入人脸库,每个人随机选择k张作为训练集,其余构成测试集
def load_orl(k):#参数K代表选择K张图片作为训练图片使用
    '''
    对训练数据集进行数组初始化，用0填充，每张图片尺寸都定为112*92,
    现在共有40个人，每个人都选择k张，则整个训练集大小为40*k,112*92
    '''
    image_path=[]
    train_face =[]
    train_label = np.zeros(40 * k)  # [0,0,.....0](共40*k个0)
    test_face =[]
    test_label = np.zeros(40 * (10 - k))
    np.random.seed(0)
    sample = np.random.rand(10).argsort() + 1  # 随机排序1-10 (0-9）+1
    for i in range(40):  # 共有40个人
        people_num = i + 1
        for j in range(10):  # 每个人都有10张照片
            image = orlpath + '/s' + str(people_num) + '/' + str(sample[j]) + '.jpg'
            image_path.append(image)
            img = img2vector(image)
            if j < k:
                # 构成训练集
                train_face.append(img)
                train_label[i * k + j] = people_num
            else:
                # 构成测试集
                test_face.append(img)
                test_label[i * (10 - k) + (j - k)] = people_num

    return train_face, train_label, test_face, test_label,image_path

def cosdist(A,B):#余弦相似度    # 参考于  https://blog.csdn.net/chary8088/article/details/74580238
    num = np.dot(A,B) #若为行向量则 A * B.T
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom #余弦值
    sim = 0.5 + 0.5 * cos #归一化
    return sim
# In[2]:

orlpath = "ORL/ORL"

radius =1  # LBP算法中范围半径的取值
n_points = 8 * radius # 领域像素点数
n_hist=2**n_points
# 读取图像
train_face, train_label, test_face, test_label,image_path = load_orl(7)  # 得到数据集
rows, cols = train_face[0].shape  #获取图片的像
num_train = len(train_face)  # 训练脸总数
num_test = len(test_face)  # 测试脸总数

data_train_new=np.zeros([num_train,n_hist])
for i in range(num_train):
    image=train_face[i]
    # lbp变换
    lbp = local_binary_pattern(image, n_points, radius)
    #统计图像的直方图
    max_bins = int(lbp.max() + 1)
    a, _ = (np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins)))
    a=np.array(a)#转为np矩阵
    data_train_new[i,:]=a
    
fig=plt.figure()
a=fig.add_subplot(1,3,1)
a.imshow(image)#显示图片
plt.xlabel('原始人脸')

a=fig.add_subplot(1,3,2)
a.imshow(lbp)#显示图片
plt.xlabel('lbp图片')
a=fig.add_subplot(1,3,3)
a.plot(data_train_new[-1,:])#显示图片


data_test_new=np.zeros((num_test,n_hist))
for i in range(num_test):
    image = test_face[i]
    lbp = local_binary_pattern(image, n_points, radius)
    max_bins = int(lbp.max() + 1)
    a, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    data_test_new[i,:]=np.array(a)


#将得到的数据集转换为np数组，以便于后续处理
data_test_new = np.array(data_test_new)  # mat change to array
data_train_new = np.array(data_train_new)



#计算测试脸到每一张训练脸的距离
result=[]
# 测试准确度
true_num = 0
for i in range(num_test):
    testFace = data_test_new[i]
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
print('测试集人脸识别精度为: %.2f%%' % (accuracy * 100))

plt.figure()
plt.scatter(np.linspace(0,num_test,num_test),result,c='red',label='预测标签')
plt.scatter(np.linspace(0,num_test,num_test),test_label,label='测试集真实标签')
plt.legend()
plt.title('测试集分类结果')
plt.show()

# In[3]:
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
img_object=cv2.imread(file_path2)# 读取图片
test_face = img2vector(file_path2)

data_test_new=np.zeros((1,n_hist))#对待测脸进行lbp特征提取
image = test_face
lbp = local_binary_pattern(image, n_points, radius)
a, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
data_test_new=np.array(a)

#将得到的数据集转换为np数组，以便于后续处理
data_test_new = np.array(data_test_new)  # mat change to array
#计算待测脸到每一张训练脸的距离，以判断其所属类别
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
# In[3]:

#选择两张图片进行对比，看是否为同一个人的照片
#目标脸
root = tk.Tk()
root.withdraw()
file_path1 = filedialog.askopenfilename()# 选择图片


img_object=cv2.imread(file_path1)# 读取图片
test_face1 = img2vector(file_path1)

data_test_new=np.zeros((2,n_hist))
image = test_face1
lbp1 = local_binary_pattern(image, n_points, radius)
max_bins = int(lbp1.max() + 1)
a, _ = np.histogram(lbp1, normed=True, bins=max_bins, range=(0, max_bins))
data_test_new[0,:]=a

fig=plt.figure()
a=fig.add_subplot(1,2,1)
a.imshow(img_object)#显示图片
plt.xlabel('目标脸')
# In[6]:
# 待测脸
root = tk.Tk()
root.withdraw()
file_path2 = filedialog.askopenfilename()# 选择图片

img_object=cv2.imread(file_path2)# 读取图片
test_face2 = img2vector(file_path2)

image = test_face2
lbp2 = local_binary_pattern(image, n_points, radius)
max_bins = int(lbp2.max() + 1)
a, _ = np.histogram(lbp2, normed=True, bins=max_bins, range=(0, max_bins))
data_test_new[1,:]=a


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



