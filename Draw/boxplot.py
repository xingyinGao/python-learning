# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:12:35 2018
路径途径加载，存在变量名
@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
#import scipy.io as scio
#from pylab import *
f=open(r'H:\aNEWpaper\san_ori\sam_water.txt')
data=[]
for line in f:
    data.append(line+'\t')
del data[0]
lis_size=len(data)
da_pro=np.zeros((lis_size,23))
j=0
for i in data:
    da_pro[j,:]=np.array(i.split())
    j=j+1 
matplotlib.rcParams['font.family'] = 'STSong'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.style']='italic'
lab=[]
for i in range(0,23):
    lab.append(1+i*16)
plt.style.use('ggplot')  #可以作出ggplot风格的图片
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
df=pd.DataFrame(da_pro*0.0001,columns=lab) 
fig=df.boxplot(notch='Ture',patch_artist='g',sym='r*',return_type='dict')
#plt.tight_layout()
#可以通过调用matplotlib.pyplot.colors()得到matplotlib支持的所有颜色
for box in fig['boxes']:
#    # 箱体边框颜色
    box.set( color='#696969', linewidth=0.5)
   # 箱体内部填充颜色
    box.set(facecolor='#696969')
for whisker in fig['whiskers']:#是指从box 到error bar之间的竖线.
    whisker.set(color='#696969', linewidth=0.8)
for cap in fig['caps']:#是指error bar横线.
    cap.set(color='#696969', linewidth=0.8)
for median in fig['medians']:#是中位值的横线, 每个median是一个Line2D对象
    median.set(color='#F0F8FF', linewidth=0.5)
for flier in fig['fliers']:#是指error bar线之外的离散点.
    flier.set(marker='o',color='#F0F8FF',alpha=0.3)       
ax=plt.gca()
ax.set_yticks(np.linspace(-0.2,1,7))  
ax.set_yticklabels(['-0.2','0','0.2','0.4','0.6','0.8','1.0']) 
x_axis = plt.gca().xaxis
for xlabel in x_axis.get_ticklabels():#获取x轴刻度元素，并操作
     #label.set_color("red")
     xlabel.set_rotation(45)
     xlabel.set_fontsize(10)
y_axis = plt.gca().yaxis     
for ylabel in y_axis.get_ticklabels():#获取x轴刻度元素，并操作
     #label.set_color("red")
     ylabel.set_rotation(45)
     ylabel.set_fontsize(10)     
#fig.patch.set_color("g")
#fig.canvas.draw()
#%matplotlib inline
#x=np.zeros((1,23))
#for i in range(0,23):
#    x[0,i]=HD[:,i].mean()
#y=np.array(lab).reshape(1,23)
#plt.plot(x,y)
#plt.axes('g')
plt.ylabel(u'NDVI value') # 这一段
plt.xlabel(u'Day Of Year')
plt.tight_layout()
plt.show()
#plt.savefig('plot123_2.png', dpi=300) #指定分辨率保存
