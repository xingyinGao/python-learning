# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 20:22:00 2019

@author: Administrator
"""
import numpy as np
import copy


class SEEDS(object):
    """
    #SLIC算法种子点生成算法；
    #类属性：
    #       self.seeds:种子点，初始为None；
    #       self.step:步长，种子点初始相隔像素个数，实例化时传入；
    #       self.image_ori:种子点生成的初始图像，实例化时传入；
    #       self.data_type:传入grad_min函数的数据的类型，默认为0，表示种子点坐标；
    #                      当其为1时，表示数组；
    #       self.method_grad:种子点类型，无需传入，默认为None；
    #       self.location_grad：特殊种子点位置，无需传入，默认为None；
    #       self.grad_style:种子点调整方式，取值为0-1，默认为0，即取梯度最小处；
    #                       当取值为1时，表示梯度最大处；
    #调用方式：
    #       sd=SEEDS(image);
    #       sd.seeds_init()：种子点初始计算；
    #       sd.seeds_adjust()：种子点调整；
    #       sd.seeds:查看种子点
    """
    
    def __init__(self,image,step=10,grad_style=0.1):
        
        self.seeds=None
        self.image_ori=image.astype(np.float32)
        self.step=step
        self.data_type=0
        self.grad_style=grad_style
        
    def grad_min(self,seed_loc):
        
        """
        #计算图像最小梯度位置;
        #输入为图像初始种子点位置索引或数组
        #返回值为图像数组中最小梯度所在位置
        """
        step_half=int(self.step/2)
        
        if self.data_type==0:
            img_sem_ori=copy.deepcopy(self.image_ori[
                            seed_loc[0]-step_half:seed_loc[0]+step_half+1,
                            seed_loc[1]-step_half:seed_loc[1]+step_half+1,
                            ...])
            locx=np.tile(np.linspace(seed_loc[0]-step_half,seed_loc[0]+step_half,
                                     self.step+1).reshape(-1,1),(1,self.step+1))
            locy=np.tile(np.linspace(seed_loc[1]-step_half,seed_loc[1]+step_half,
                                     self.step+1).reshape(1,-1),(self.step+1,1))
        elif self.data_type==1:
            return
        
        rc_loc=locx.shape
        locxy=np.array([0]*rc_loc[1]+list(zip(locx.reshape(-1).astype(int),
                       locy.reshape(-1).astype(int))),dtype=object).reshape(
                       rc_loc[0]+1,rc_loc[1])[1:,:]
                
        grad_all=0
        if len(img_sem_ori.shape)==3:
            for i in range(img_sem_ori.shape[2]):
                img_sem=img_sem_ori[:,:,i]
                grad_x=np.abs(img_sem[:,1:]-img_sem[:,:-1])
                grad_y=np.abs(img_sem[1:,:]-img_sem[:-1,:])
                grad=grad_x[:-1,:]+grad_y[:,:-1]
                grad_all=grad_all+grad
                
            grad_all=grad_all/img_sem_ori.shape[2] 
            
        else:
            grad_x=np.abs(img_sem_ori[:,1:]-img_sem_ori[:,:-1])
            grad_y=np.abs(img_sem_ori[1:,:]-img_sem_ori[:-1,:])
            grad=grad_x[:-1,:]+grad_y[:,:-1]
            grad_all=grad_all+grad    
            
        grad_range=grad_all.max()-grad_all.min()
        grad_loc=np.where(grad_all<grad_all.min()+grad_range*self.grad_style)
        
        if len(grad_loc[0])>2:
#            rd_loc=np.random.randint(0,len(grad_loc),1)[0]
            rd_loc=int(len(grad_loc)/2)
        else:
            rd_loc=0
      
        loc_x=grad_loc[0][rd_loc]
        loc_y=grad_loc[1][rd_loc]
        
        return locxy[loc_x,loc_y]

    def seeds_init(self):
        
        """
        #种子点初始化函数；
        """
        image=self.image_ori
        step_init=self.step+1
        rc=image.shape
        r_seed,c_seed=int(rc[0]/step_init),int(rc[1]/step_init)
        seeds_mat_init=np.zeros((r_seed,c_seed),object)
        for i in range(r_seed):
            for j in range(c_seed):
                seeds_mat_init[i,j]=[i*step_init+int(step_init/2),
                                      j*step_init+int(step_init/2)]
        self.seeds=seeds_mat_init.reshape(-1)
        
    def seeds_adjust(self):
        """
        #种子点根据梯度调整位置；
        """
        self.data_type=0
        self.seeds=list(map(self.grad_min,self.seeds))
        

