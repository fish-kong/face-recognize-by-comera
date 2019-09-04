import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']#有这个画图的时候才能显示中文

'极限学习机--extreme learning machine'

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
    train_label = np.zeros(40 * k,dtype=int)  # [0,0,.....0](共40*k个0)
    test_face = np.zeros((40 * (10 - k), 112 * 92))
    test_label = np.zeros(40 * (10 - k),dtype=int)
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
# In[2]:
'极限学习机训练函数'
def train_elm(P,T,N=10,C=100000000000000,TF='sig',TYPE=1):
    # P 输入数据 n*m  n-> samples m->features
    # T 输出数据
    # N 隐含层节点
    # C 正则化参数
    # TF 隐含层激活函数
    # TYPE=1 分类任务  =0 回归任务
    n,m=P.shape
    if TYPE == 1:
        y=np.zeros([n,41])
        for i in range(n):
            y[i,T[i]]=1
        T=np.copy(y)
    
    n,c=T.shape
    '输入权重'
    Weights = 2*np.random.rand(m,N)-1
    '隐含层偏置'
    biases=np.random.rand(1,N)
    
    temp=np.matmul(P,Weights)+np.tile(biases,[n,1])
    if TF=='sig':
        H=1/(1+np.exp(temp))
    '输出权重计算'
    w_out=np.matmul(np.matmul(np.linalg.pinv(np.matmul(H.T,H)+1/C),H.T),T)
    return Weights ,biases ,w_out, TF, TYPE

def train_predic(P,Weights,biases,w_out,TF,TYPE):
    n,m=P.shape
    temp=np.matmul(P,Weights)+np.tile(biases,[n,1])
    if TF=='sig':
        H=1/(1+np.exp(temp))
    T=np.matmul(H,w_out)
    
    if TYPE==1:
        T_predict=np.argmax(T,axis=1)
    return T_predict
def compute_accuracy(T_true,T_predict,TYPE):
    if TYPE==0:
        accuracy=np.mean(np.sum(T_true-T_predict))
    if TYPE==1:
        n=0
        for i in range(len(T_true)):
           if T_true[i]==T_predict[i]:
               n=n+1
        accuracy=n/len(T_true)
    return accuracy
           
# In[3]:
print('提取文件')
print('——————————————————————————————————————')

orlpath = "ORL/ORL"
train_face, train_label, test_face, test_label,image_path = load_orl(7)  # 得到数据集
num_train = train_face.shape[0]  # 训练脸总数
num_test = test_face.shape[0]  # 测试脸总数

Weights,biases,w_out,TF,TYPE=train_elm(train_face,train_label,N=2000,C=100000,TF='sig',TYPE=1)

T_predict=train_predic(test_face,Weights,biases,w_out,TF,TYPE)
accuracy=compute_accuracy(test_label,T_predict,TYPE)


plt.figure()
plt.scatter(np.linspace(0,num_test,num_test),T_predict,c='red',label='预测标签')
plt.scatter(np.linspace(0,num_test,num_test),test_label,label='测试集真实标签')
plt.legend()
plt.title('测试集分类结果')
plt.show()

print('测试集人脸识别精度为: %.2f%%' % (accuracy * 100))
# In[4]:
# In[5]:
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




label2=train_predic(test_face,Weights,biases,w_out,TF,TYPE)



a=fig.add_subplot(1,2,2)
a.imshow(img_object)#显示图片
plt.xlabel('待测脸')
if label1==label2:
    plt.suptitle('匹配成功，是同一个人')
    print('匹配成功，是同一个人')
else:
    plt.suptitle('匹配失败，不是同一个人')
    print('匹配失败，不是同一个人')


# 极限学习机没办法对训练集样本之外的样本进行分类




