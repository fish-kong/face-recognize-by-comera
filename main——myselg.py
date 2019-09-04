# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:45:37 2019

@author: KLY
"""

import cv2
import numpy as np
import scipy.io as sio
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']#有这个画图的时候才能显示中文

P=sio.loadmat('PCA_data.mat')

def predict(data_train_new,data_mean,V_r,train_label,img):
    rows, cols = 112,92  
    test_face = np.reshape(img, (1, rows * cols))#
    temp_face = np.array(test_face - data_mean)# 对待测图片进行pca
    data_test_new =np.dot( temp_face , V_r)  # 得到测试脸在特征向量下的数据
    num_train=data_train_new.shape[0]
    #将得到的数据集转换为np数组，以便于后续处理
    #计算待测脸到每一张训练脸的距离，以判断其所属类别
    diffMat = data_train_new - np.tile(data_test_new, (num_train, 1))  
    sqDiffMat = diffMat ** 2                                  
    sqDistances = sqDiffMat.sum(axis=1)  # 采用欧式的是欧式距离
    
    sortedDistIndicies = sqDistances.argsort()  # 对向量从小到大排序，使用的是索引值,得到一个向量
    indexMin = sortedDistIndicies[0]  # 距离最近的索引
    return train_label[0,indexMin]
def detect():
  face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
  #eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
  camera = cv2.VideoCapture(0)
  counter=1

  while (True):
    ret, img = camera.read()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img, 1.3, 5)#捕获到的帧要转换为灰度图像
    counter = counter-1
    
    if counter==0:
        counter = 1
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)#将框框加进图片帧中
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]
            roi = cv2.resize(roi, (112, 92), interpolation=cv2.INTER_LINEAR)
            
            params = predict(P['train'],P['data_mean'],P['V_r'],P['train_label'],roi)
            print(params)
            if params==40:
                params='Kong lingyu'
            cv2.putText(img, str(params), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
            
            #用于检测器官
            #roi_gray = gray[y:y+h, x:x+w]        
            #eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40,40))
            #for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)

    cv2.imshow("camera", img)
    
    if cv2.waitKey(1) & 0xff == ord("q"):#输入q  结束程序
      break

  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  detect()
