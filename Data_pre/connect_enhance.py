# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:44:12 2019

@author: Administrator
"""
import numpy as np
import copy
import random

class Region_grow(object):
    """
    #连通性增强函数（区域增长法）：
    #调用方式：
    #       rg=Region_grow(super_pixel_image)#sl.dec_mat
    #       rg.main()
    #
    #属性变量：
    #       self.image:超像素图像；
    #       self.count：超像素最小像素集合，int类型,默认100；
    #       self_post_label:处理后超像素图像；
    #       self.relabel:重新随机超像素标签列表；
    #函数定义：
    #       __init__(self,img_label,count=100)：生长类构造函数；
    #       get_unique_index(self,uniuqe_value)：获取唯一值所有索引位置；
    #       grow(self):区域增长，不满大于count的区域块重置标签为-1；
    #       enhance(self):对未有分配标签的像素分配标签，最近原则；
    #       makeboundary(self,image_tempalte):根据标签数据绘制边界,image_tempaltem模板图像；
    #       main(self),依次调用grow(),enhance();
    """
    
    def __init__(self,img_label,count=100):
        self.image=copy.deepcopy(img_label)
        self.count=count
        self.post_label=None
        self.relabel=None
        self.boundary=None
        
    def get_unique_index(self,uniuqe_value):
        return np.where(self.image==uniuqe_value)
    
#    def assig_1v(self,loc):
#        relabel=self.relabel.pop(0)
        
#        self.post_label[loc[0],loc[1]]=1
#        self.image[loc[0],loc[1]]=relabel
 
    def grow(self):
        self.post_label=-1*np.ones(self.image.shape)
        unique_labels=np.unique(self.image.reshape(-1))[1:]
        self.relabel=random.sample(range(1,2*len(unique_labels)+1),int(1.5*len(unique_labels)))
#        print(len(self.relabel))
        loc_array=list(map(self.get_unique_index,unique_labels))
        num=1
        while loc_array:
#            print(len(loc_array))
            loc_label=loc_array.pop(0)
            list_sin_loc=list(zip(loc_label[0],loc_label[1]))
            while list_sin_loc:
#                print(len(list_sin_loc))
                region_set_init=[]
                region_set_res=[]
                sin2=list_sin_loc.pop(0)
                region_set_init.append(sin2)
                region_set_res.append(sin2)
                while region_set_init:
                    _loc=region_set_init.pop(0)
                    if _loc[1]+1<self.image.shape[1]:
                        if self.image[_loc[0],_loc[1]]==self.image[_loc[0],_loc[1]+1]:
                            if (_loc[0],_loc[1]+1) not in region_set_res:
                                region_set_init.append((_loc[0],_loc[1]+1))
                                region_set_res.append((_loc[0],_loc[1]+1))
                                
                    if _loc[1]-1>=0:
                        if self.image[_loc[0],_loc[1]]==self.image[_loc[0],_loc[1]-1]:
                            if (_loc[0],_loc[1]-1) not in region_set_res:
                                region_set_init.append((_loc[0],_loc[1]-1))
                                region_set_res.append((_loc[0],_loc[1]-1))
                                
                    if _loc[0]+1<self.image.shape[0]:
                        if self.image[_loc[0],_loc[1]]==self.image[_loc[0]+1,_loc[1]]:
                            if (_loc[0]+1,_loc[1]) not in region_set_res:
                                region_set_init.append((_loc[0]+1,_loc[1]))
                                region_set_res.append((_loc[0]+1,_loc[1]))
                                
                    if _loc[0]-1>=0:
                        if self.image[_loc[0],_loc[1]]==self.image[_loc[0]-1,_loc[1]]:
                            if (_loc[0]-1,_loc[1]) not in region_set_res:
                                region_set_init.append((_loc[0]-1,_loc[1]))
                                region_set_res.append((_loc[0]-1,_loc[1]))
                
                if len(region_set_res)>self.count:
                    print(num)
                    num=num+1
                    loc_pair_x=[x[0] for x in region_set_res]
                    loc_pair_y=[y[1] for y in region_set_res]
#                    for loc_cet in region_set_res:
                    relabel=self.relabel.pop(0)
                    self.post_label[loc_pair_x,loc_pair_y]=relabel
#                        self.assig_1v(loc_cet)
                for loc_res in region_set_res:
                    if loc_res in list_sin_loc:
                        list_sin_loc.remove(loc_res)
                        
    def enhance(self):
        
        loc_minus1=np.where(self.post_label==-1)
#        self.image[loc_minus1]=0
        loc_minus1=list(zip(loc_minus1[0],loc_minus1[1]))
        
        while loc_minus1:
            flag_all=[]
            sin_loc=loc_minus1.pop(0)
            if sin_loc[0]>0:
                flag_up=np.where(np.array(list(self.post_label[:sin_loc[0],sin_loc[1]]).reverse())!=-1)[0]
                if len(flag_up)!=0:
                    flag_all.append([flag_up[0],(sin_loc[0]-flag_up[0]-1,sin_loc[1])])
            if sin_loc[0]<self.post_label.shape[0]-1:
                flag_low=np.where(self.post_label[sin_loc[0]+1:,sin_loc[1]]!=-1)[0]
                if len(flag_low)!=0:
                    flag_all.append([flag_low[0],(flag_low[0]+sin_loc[0]+1,sin_loc[1])])
            if sin_loc[1]>0:
                flag_left=np.where(np.array(list(self.post_label[sin_loc[0],:sin_loc[1]]).reverse())!=-1)[0]
                if len(flag_left)!=0:
                    flag_all.append([flag_left[0],(sin_loc[0],sin_loc[1]-flag_left[0]-1)])
            if sin_loc[1]<self.post_label.shape[1]-1:
                flag_right=np.where(self.post_label[sin_loc[0],sin_loc[1]+1:]!=-1)[0]
                if len(flag_right)!=0:
                    flag_all.append([flag_right[0],(sin_loc[0],sin_loc[1]+1+flag_left[0])])
            
            flag_best=[fall for fall in flag_all if fall[0]==min(flag_all,key=lambda x:x[0])[0]]
            choice=random.sample(range(len(flag_best)),1)[0]
            best_index=flag_best[choice][1]
            self.post_label[sin_loc[0],sin_loc[1]]=self.post_label[best_index[0],best_index[1]]
            
    def makeboundary(self,image_tempalte):
        
        self.boundary=copy.deepcopy(image_tempalte)
        
        _row_img=self.post_label[:-1,:]-self.post_label[1:,:]
        loc_row_01=np.where(_row_img!=0)
        loc_row_02_x=[loc+1 for loc in loc_row_01[0]]
        loc_row_02=(loc_row_02_x,loc_row_01[1])
        
        _col_img=self.post_label[:,:-1]-self.post_label[:,1:]
        loc_col_01=np.where(_col_img!=0)
        loc_col_02_y=[loc+1 for loc in loc_col_01[1]]
        loc_col_02=(loc_col_01[0],loc_col_02_y)
        
        self.boundary[loc_row_01[0],loc_row_01[1],...]=[0,255,0]
        self.boundary[loc_row_02[0],loc_row_02[1],...]=[0,255,0]
        self.boundary[loc_col_01[0],loc_col_01[1],...]=[0,255,0]
        self.boundary[loc_col_02[0],loc_col_02[1],...]=[0,255,0]
        
    def main(self):
        self.grow()
        self.enhance()
