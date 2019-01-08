# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:50:12 2019

@author: Administrator
"""
from seeds_slic import SEEDS
import copy
import numpy as np
import random

class SLIC(SEEDS):
    """
    #超像素生成函数：
    #调用方式：
    #   sl=SLIC(img,step,grad_style)
    #   sl.main()
    #   sl.Seeds_grow()
    #属性变量：
    #self.method_grad:种子点类型，无需传入，默认为None；
    #self.dec_mat：存储结果数组；
    #self.dec_weight：像素相似性存储数组，初始均为inf；
    #self.label：超像素标签；
    #self.spa_factor：空间因子，越大，空间权重越小；
    #self.spec_factor：光谱权重，越大，光谱权重越小；
    #函数定义：
    #update_label(self,img_sem_ori,seed_loc,spa_all,loc_range):超像素标签更新函数
    #       其中，img_sem_ori为搜索窗口矩阵，seed_loc为种子点坐标，spa_all为空间权重矩阵，
    #       loc_range为坐标范围；
    #Assign(self,seed_loc):赋值函数，根据种子点类型，对搜索窗口内元素计算相似性，并贴标签；
    #       其中，seed_loc为种子点；
    #Seeds_grow(self)：判断种子点类型，并调用Assign函数
    #__init__(self,img,step=10,grad_style=0.1):类的构造函数，其中img,step,grad_style
    #       均为继承类变量赋值；
    """
    
    def __init__(self,img,step=10,grad_style=0.1):
        
        SEEDS.__init__(self,image=img,step=step,grad_style=grad_style)
        self.method_grad=0
        self.dec_mat=np.zeros((img.shape[0],img.shape[1]))###存储结果数组
        self.dec_weight=np.zeros((img.shape[0],img.shape[1]))###像素相似性存储数组
        self.dec_weight[:]=np.inf###相似性初始化
        self.label=0##超像素标签
        self.spa_factor=self.step**2
        self.spec_factor=10**2
        
    def update_label(self,img_sem_ori,seed_loc,spa_all,loc_range):
        
            label=self.label
            rc1=img_sem_ori.shape
            ###光谱权重
            spec_weigth=np.abs(img_sem_ori-np.tile(self.image_ori[seed_loc[0],
                        seed_loc[1],:].reshape(1,1,rc1[2]),(rc1[0],rc1[1],1))).mean(axis=2)
            
            ###综合权重
            weight_all=spa_all/(self.spa_factor)+spec_weigth/(self.spec_factor)
            
            ###与已有权重比较，取小者位置
            loc_change=np.where(weight_all<self.dec_weight[loc_range[0][0]:
                            loc_range[0][1],loc_range[1][0]:loc_range[1][1]])
            
            ###更新像素标签
            dec_mat_tmp=self.dec_mat[loc_range[0][0]:loc_range[0][1],
                                     loc_range[1][0]:loc_range[1][1]]
            dec_mat_tmp[loc_change]=label
            self.dec_mat[loc_range[0][0]:loc_range[0][1],
                         loc_range[1][0]:loc_range[1][1]]=dec_mat_tmp
            
            ###更新综合权重
            dec_weight_tmp=self.dec_weight[loc_range[0][0]:loc_range[0][1],
                                           loc_range[1][0]:loc_range[1][1]]
            dec_weight_tmp[loc_change]=weight_all[loc_change]
            self.dec_weight[loc_range[0][0]:loc_range[0][1],
                            loc_range[1][0]:loc_range[1][1]]=dec_weight_tmp
                            
    def Assign(self,seed_loc):
        
        rc=self.image_ori.shape
        if self.method_grad==0:###行列方向搜索半径均未超出图像范围
            img_sem_ori=copy.deepcopy(self.image_ori[
                            seed_loc[0]-self.step:seed_loc[0]+self.step+1,
                            seed_loc[1]-self.step:seed_loc[1]+self.step+1,
                            ...])
    
            spa1=np.tile(np.abs(np.linspace(-self.step,self.step,2*self.step+1
                                     )).reshape(-1,1),(1,2*self.step+1))
            spa2=np.tile(np.abs(np.linspace(-self.step,self.step,2*self.step+1
                                     )).reshape(1,-1),(2*self.step+1,1))
            spa_all=spa1**2+spa2**2
            
            loc_range=[[seed_loc[0]-self.step,seed_loc[0]+self.step+1],
                       [seed_loc[1]-self.step,seed_loc[1]+self.step+1]]
            self.update_label(img_sem_ori,seed_loc,spa_all,loc_range)
            
        elif self.method_grad==1:###行方向搜索半径超出图像范围，前半部分
            img_sem_ori=copy.deepcopy(self.image_ori[
                            :seed_loc[0]+self.step+1,
                            seed_loc[1]-self.step:seed_loc[1]+self.step+1
                            ,...])
            
            spa1=np.tile(np.abs(np.linspace(-seed_loc[0],self.step,seed_loc[0]+self.step+1
                                     )).reshape(-1,1),(1,2*self.step+1))
            spa2=np.tile(np.abs(np.linspace(-self.step,self.step,2*self.step+1
                                     )).reshape(1,-1),(seed_loc[0]+self.step+1,1))
            spa_all=spa1**2+spa2**2
            
            loc_range=[[0,seed_loc[0]+self.step+1],[seed_loc[1]-self.step,seed_loc[1]+self.step+1]]
            self.update_label(img_sem_ori,seed_loc,spa_all,loc_range)
            
        elif self.method_grad==2:###行方向搜索半径超出图像范围，后半部分
            img_sem_ori=copy.deepcopy(self.image_ori[
                            seed_loc[0]-self.step:,
                            seed_loc[1]-self.step:seed_loc[1]+self.step+1
                            ,...])
            
            spa1=np.tile(np.abs(np.linspace(-self.step,rc[0]-1-seed_loc[0],rc[0]-seed_loc[0]+self.step
                                     )).reshape(-1,1),(1,2*self.step+1))
            spa2=np.tile(np.abs(np.linspace(-self.step,self.step,2*self.step+1
                                     )).reshape(1,-1),(rc[0]-seed_loc[0]+self.step,1))
            spa_all=spa1**2+spa2**2
            
            loc_range=[[seed_loc[0]-self.step,rc[0]],[seed_loc[1]-self.step,seed_loc[1]+self.step+1]]
            self.update_label(img_sem_ori,seed_loc,spa_all,loc_range)
            
        elif self.method_grad==3:###列方向搜索半径超出图像范围，前半部分
            img_sem_ori=copy.deepcopy(self.image_ori[
                            seed_loc[0]-self.step:seed_loc[0]+self.step+1,
                            :seed_loc[1]+self.step+1,...])
    
            spa1=np.tile(np.abs(np.linspace(-self.step,self.step,2*self.step+1
                                     )).reshape(-1,1),(1,self.step+seed_loc[1]+1))
            spa2=np.tile(np.abs(np.linspace(-seed_loc[1],self.step,self.step+seed_loc[1]+1
                                     )).reshape(1,-1),(2*self.step+1,1))
            spa_all=spa1**2+spa2**2
                
            loc_range=[[seed_loc[0]-self.step,seed_loc[0]+self.step+1],
                       [0,seed_loc[1]+self.step+1]]
            self.update_label(img_sem_ori,seed_loc,spa_all,loc_range)
                            
        elif self.method_grad==4:###列方向搜索半径超出图像范围，后半部分
            img_sem_ori=copy.deepcopy(self.image_ori[
                            seed_loc[0]-self.step:seed_loc[0]+self.step+1,
                            seed_loc[1]-self.step:,...])
    
            spa1=np.tile(np.abs(np.linspace(-self.step,self.step,2*self.step+1
                                     )).reshape(-1,1),(1,rc[1]-seed_loc[1]+self.step))
            spa2=np.tile(np.abs(np.linspace(-self.step,rc[1]-seed_loc[1]-1,rc[1]-seed_loc[1]+self.step
                                     )).reshape(1,-1),(2*self.step+1,1))
            spa_all=spa1**2+spa2**2
            
            loc_range=[[seed_loc[0]-self.step,seed_loc[0]+self.step+1],
                       [seed_loc[1]-self.step,rc[1]]]
            self.update_label(img_sem_ori,seed_loc,spa_all,loc_range)
                            
        elif self.method_grad==5:###行列方向搜索半径均超出图像范围
            if self.location_grad==1:##左上角
                img_sem_ori=copy.deepcopy(self.image_ori[
                            :seed_loc[0]+self.step+1,
                            :seed_loc[1]+self.step+1,...])
    
                spa1=np.tile(np.abs(np.linspace(-seed_loc[0],self.step,seed_loc[0]+self.step+1
                                     )).reshape(-1,1),(1,self.step+seed_loc[1]+1))
                spa2=np.tile(np.abs(np.linspace(-seed_loc[1],self.step,self.step+seed_loc[1]+1
                                     )).reshape(1,-1),(seed_loc[0]+self.step+1,1))
                spa_all=spa1**2+spa2**2
                
                loc_range=[[0,seed_loc[0]+self.step+1],
                           [0,seed_loc[1]+self.step+1]]
                self.update_label(img_sem_ori,seed_loc,spa_all,loc_range)
                
            elif self.location_grad==2:##右上角
                img_sem_ori=copy.deepcopy(self.image_ori[
                            :seed_loc[0]+self.step+1,
                            seed_loc[1]-self.step:,...])
        
                spa1=np.tile(np.abs(np.linspace(-seed_loc[0],self.step,seed_loc[0]+self.step+1
                                     )).reshape(-1,1),(1,rc[1]-seed_loc[1]+self.step))
                spa2=np.tile(np.abs(np.linspace(-self.step,rc[1]-seed_loc[1]-1,rc[1]-seed_loc[1]+self.step
                                     )).reshape(1,-1),(seed_loc[0]+self.step+1,1))
                spa_all=spa1**2+spa2**2
                
                loc_range=[[0,seed_loc[0]+self.step+1],
                           [seed_loc[1]-self.step,rc[1]]]
                self.update_label(img_sem_ori,seed_loc,spa_all,loc_range)
    
            elif self.location_grad==3:##左下角
                img_sem_ori=copy.deepcopy(self.image_ori[
                            seed_loc[0]-self.step:,
                            :seed_loc[1]+self.step+1,...])
    
                spa1=np.tile(np.abs(np.linspace(-self.step,rc[0]-1-seed_loc[0],rc[0]-seed_loc[0]+self.step
                                     )).reshape(-1,1),(1,self.step+seed_loc[1]+1))
                spa2=np.tile(np.abs(np.linspace(-seed_loc[1],self.step,self.step+seed_loc[1]+1
                                     )).reshape(1,-1),(rc[0]-seed_loc[0]+self.step,1))
                spa_all=spa1**2+spa2**2
                
                loc_range=[[seed_loc[0]-self.step,rc[0]],
                           [0,seed_loc[1]+self.step+1]]
                self.update_label(img_sem_ori,seed_loc,spa_all,loc_range)
    
            elif self.location_grad==4:##右下角
                img_sem_ori=copy.deepcopy(self.image_ori[
                            seed_loc[0]-self.step:,
                            seed_loc[1]-self.step:,...])
    
                spa1=np.tile(np.abs(np.linspace(-self.step,rc[0]-1-seed_loc[0],rc[0]-seed_loc[0]+self.step
                                     )).reshape(-1,1),(1,rc[1]-seed_loc[1]+self.step))
                spa2=np.tile(np.abs(np.linspace(-self.step,rc[1]-seed_loc[1]-1,rc[1]-seed_loc[1]+self.step
                                     )).reshape(1,-1),(rc[0]-seed_loc[0]+self.step,1))
                spa_all=spa1**2+spa2**2
        
                loc_range=[[seed_loc[0]-self.step,rc[0]],
                           [seed_loc[1]-self.step,rc[1]]]
                self.update_label(img_sem_ori,seed_loc,spa_all,loc_range)
    
    def Seeds_grow(self):
        
        seeds_now=copy.deepcopy(self.seeds)
        r_image,c_image=self.image_ori.shape[:2]
        
        ###筛选行方向上搜索半径超过图像范围的种子点，前半部分
        
        index_cut=int(c_image/self.step)
        row1=np.array(list(map(lambda x:x[0]-self.step,seeds_now[:5*index_cut])))
        seeds_spec_row1=np.where(row1<0)[0]
        
        
        ###筛选行方向上搜索半径超过图像范围的种子点，后半部分
        index_acc=len(seeds_now)-5*index_cut
        row_inf=np.array(list(map(lambda x:x[0]+self.step,seeds_now[index_acc:])))
        seeds_spec_row_inf=np.where(row_inf>r_image-1)[0]+index_acc
        
         ###筛选列方向上搜索半径超过图像范围的种子点，前半部分
        col1=np.array(list(map(lambda x:x[1]-self.step,seeds_now)))
        seeds_spec_col1=np.where(col1<0)[0]
        
        ###筛选列方向上搜索半径超过图像范围的种子点，后半部分
        col_inf=np.array(list(map(lambda x:x[1]+self.step,seeds_now)))
        seeds_spec_col_inf=np.where(col_inf>c_image-1)[0]
        
        seeds_spec_all=list(seeds_spec_row1)+list(seeds_spec_row_inf)+\
                        list(seeds_spec_col1)+list(seeds_spec_col_inf)
        
        ###筛选搜索半径行列方向均不超过图像范围的种子点
        seeds_normal_loc=[loc_i for loc_i in range(len(seeds_now))
                          if loc_i not in seeds_spec_all]
        
        ###筛选行列方向均超过图像范围的种子点
        cor1_spec_loc=[_loc for _loc in seeds_spec_row1 if _loc in seeds_spec_col1]
        cor2_spec_loc=[_loc for _loc in seeds_spec_row1 if _loc in seeds_spec_col_inf]
        cor3_spec_loc=[_loc for _loc in seeds_spec_row_inf if _loc in seeds_spec_col1]
        cor4_spec_loc=[_loc for _loc in seeds_spec_row_inf if _loc in seeds_spec_col_inf]
        
        ###行方向、列方向搜索半径超过图像范围种子点二次筛选
        cor_all=cor1_spec_loc+cor2_spec_loc+cor3_spec_loc+cor4_spec_loc
        seeds_spec_row1=[_loc for _loc in seeds_spec_row1 if _loc not in cor_all]
        seeds_spec_row_inf=[_loc for _loc in seeds_spec_row_inf if _loc not in cor_all]
        seeds_spec_col1=[_loc for _loc in seeds_spec_col1 if _loc not in cor_all]
        seeds_spec_col_inf=[_loc for _loc in seeds_spec_col_inf if _loc not in cor_all]
        
        label_list0=random.sample(range(1,len(self.seeds)+1),len(self.seeds))
        label_list=copy.deepcopy(label_list0)
        
        ###
        self.method_grad=0
        for _loc_seed in seeds_normal_loc:
            self.label=label_list.pop(0)
            self.Assign(self.seeds[_loc_seed])
        
        self.method_grad=1
        for _loc_seed in seeds_spec_row1:
            self.label=label_list.pop(0)
            self.Assign(self.seeds[_loc_seed])
            
        self.method_grad=2
        for _loc_seed in seeds_spec_row_inf:
            self.label=label_list.pop(0)
            self.Assign(self.seeds[_loc_seed])
            
        self.method_grad=3
        for _loc_seed in seeds_spec_col1:
            self.label=label_list.pop(0)
            self.Assign(self.seeds[_loc_seed])
        
        self.method_grad=4
        for _loc_seed in seeds_spec_col_inf:
            self.label=label_list.pop(0)
            self.Assign(self.seeds[_loc_seed])
            
        self.method_grad=5
        self.location_grad=1
        for _loc_seed in cor1_spec_loc:
            self.label=label_list.pop(0)
            self.Assign(self.seeds[_loc_seed])
            
        self.location_grad=2
        for _loc_seed in cor2_spec_loc:
            self.label=label_list.pop(0)
            self.Assign(self.seeds[_loc_seed])
            
        self.location_grad=3
        for _loc_seed in cor3_spec_loc:
            self.label=label_list.pop(0)
            self.Assign(self.seeds[_loc_seed])
            
        self.location_grad=4
        for _loc_seed in cor4_spec_loc:
            self.label=label_list.pop(0)
            self.Assign(self.seeds[_loc_seed])
            
        
