# -*- encoding: utf-8 -*-
"""
@File    : creat_vsm.py
@Time    : 2020/12/30 20:42
@Author  : XD
@Email   : gudianpai@qq.com
@Software: PyCharm
"""
import numpy as np
import os
import os.path
import cv2
import scipy.io as sio

#一个函数只能完成一个功能，这样有利于组织

ImgDB_path = "F:\\CBRI\\Simple_CBIR\\ImgDB"

#得到ImgDB中所有图片的名称，并且返回一个列表
def ImgDB_names(path = ImgDB_path):
    img_names = os.listdir(path)
    return img_names

#得到每一张图片的路径
def each_img_full_path():
    each_img_full_path = []
    name = ImgDB_names()
    for name in img_names:
        img_path = os.path.join('F:\\CBRI\\Simple_CBIR\\ImgDB',name)
        each_img_full_path.append(img_path)
    return each_img_full_path

#计算每一张图片的特征
def get_hist(img):

    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV).astype(np.float)

    h = hsv_img[...,0]
    s = hsv_img[...,1]
    v = hsv_img[...,2]

    height, width = h.shape

    h = h * 2
    h[(h >=316) | (h <= 20)] = 0
    h[(h >= 21) & (h <= 40)] = 1
    h[(h >= 41) & (h <= 75)] = 2
    h[(h >= 76) & (h <= 155)] = 3
    h[(h >= 156) & (h <= 190)] = 4
    h[(h >= 191) & (h <= 270)] = 5
    h[(h >= 271) & (h <= 295)] = 6
    h[(h >= 296) & (h <= 315)] = 7

    s = s / 255
    s[(s >= 0) & (s <= 0.2)] = 0
    s[(s >= 0.2) & (s <= 0.7)] = 1
    s[(s >= 0.7) & (s <= 1)] = 2

    v = v / 255
    v[(v >= 0) & (v <= 0.2)] = 0
    v[(v >= 0.2) & (v <= 0.7)] = 1
    v[(v >= 0.7) & (v <= 1)] = 2

    g = 9 * h + 3 * s + v

    # print('h:',h)
    # print('s:',s)
    # print('v:',v)
    hist = []
    # print(np.min(g),np.max(g))
    for i in range(72):
        a = (g == i).sum()
        hist.append(a)
    return np.array(hist)

#得到所有照片的特征值

def get_all_images_72v(path):
    value = []
    n = 0
    for l in path:
        print(l)
        img = cv2.imread(l)
        print("->正在计算第:",n+1,"张图片特征值")
        v = get_hist(img)
        value.append(v)
        n += 1
    print("全部计算完毕，一共",n,"张")
    return value


#处理语义名字
def spilt_name(name):
    """
    输入参数形式：'bird_056.jpg
    返回：bird
    """
    L = [x.split("_")[0] for x in name]
    return L

def unique(name):
    L = spilt_name(name)
    unique = np.unique(L)
    return unique

def unique_to_num(name):
    L = spilt_name(name)
    U = unique(name)
    target = []
    for n in L:
        if n == U[0]:
            target.append(0)
        if n == U[1]:
            target.append(1)
        if n == U[2]:
            target.append(2)
        if n == U[3]:
            target.append(3)
        if n == U[4]:
            target.append(4)
        if n == U[5]:
            target.append(5)
        if n == U[6]:
            target.append(6)
        if n == U[7]:
            target.append(7)
        if n == U[8]:
            target.append(8)
        if n == U[9]:
            target.append(9)
    return np.array(target)

if __name__ == '__main__':
    ImgDB_path = "F:\\CBRI\\Simple_CBIR\\ImgDB"

    img_names = ImgDB_names()
    # print(img_names)

    path = each_img_full_path()
    # print(path)

    print('---------------------------------------------------------------------------------------')
    print('开始计算图片的特征向量')
    print('---------------------------------------------------------------------------------------')
    data = get_all_images_72v(path)
    name = img_names

    print('---------------------------------------------------------------------------------------')
    print("->正在处理语义名")
    print(".............")
    target = unique_to_num(name)
    print("->处理语义名完毕")
    print('---------------------------------------------------------------------------------------')

    print('---------------------------------------------------------------------------------------')
    print("-->>正在保存vsm矩阵")
    dict = dict(data = data,name = name,target = target)
    sio.savemat("F:\CBRI\Simple_CBIR\VSM.mat",dict)
    print('---------------------------------------------------------------------------------------')
    print("-->>保存完毕")


