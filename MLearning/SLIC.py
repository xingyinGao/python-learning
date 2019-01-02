# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 16:51:48 2018

@author: Administrator
"""
import numpy as np
import copy
import random
from progressbar import ProgressBar
import time

def grad_min(img_sem_ori):
    """
    #计算图像最小梯度位置;
    #输入为二维图像数组
    #返回值为图像数组中最小梯度所在位置
    """
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
            
    grad_all[np.where(grad_all==0)]=np.inf
    loc=np.where(grad_all==grad_all.min())
    loc_x=loc[0][0]
    loc_y=loc[1][0]
    return [loc_x,loc_y]


def Get_index_global(loc_partial,N):
    """
    #获取全局索引函数;
    #loc_partial为图像所有种子点在所在窗口的位置索引组成的二维数组，N为窗口大小；
    #返回值为各种子点在整幅图像中位置索引组成的二维数组
    """
    loc_global=copy.deepcopy(loc_partial)
    for i in range(loc_partial.shape[0]):
        for j in range(loc_partial.shape[1]):
            loc=loc_partial[i,j]
            loc_global[i,j]=[i*N+loc[0],j*N+loc[1]]
    return loc_global


def DEC(image,N,center_point_list):
    """
    #建立像素判分函数,判断种子点附近像素归属;
    #输入为：image，原始图像；N，种子点搜索窗口大小，像素归属判断搜索窗口为其2倍；
    #       center_point_list，种子点全局位置索引；
    #输出为超像素分割图像；
    note：本判分函数由五部分组成，第一行，最后一行，第一列，最后一列，中心处种子点
          分别进行像素归属判断；
    
    """
    rc0=center_point_list.shape###种子点数组大小
    dec_mat=np.zeros((image.shape[0],image.shape[1]))###存储结果数组
    dec_weight=np.zeros((image.shape[0],image.shape[1]))###像素相似性存储数组
    dec_weight[:]=np.inf###相似性初始化
    
    ##################################################
    ###种子点贴标签，使得在一定范围内的超像素具有较明显的界限
    seeds_range=rc0[1]
    random_seeds1=random.sample(range(1,seeds_range+1),seeds_range)
    random_seeds2=random.sample(range(seeds_range+1,2*seeds_range+1),seeds_range)
    random_seeds=copy.deepcopy(random_seeds1)
    random_seeds.extend(random_seeds2)
    center_seq=[]
    
    ###每两行种子点，标签重复一次，确保相邻超像素不会有相同的标签
    for i in range(int(rc0[0]/2)):
        center_seq.extend(random_seeds)
        random_seeds1=random.sample(range(1,seeds_range+1),seeds_range)
        random_seeds2=random.sample(range(seeds_range+1,2*seeds_range+1),seeds_range)
        random_seeds=copy.deepcopy(random_seeds1)
        random_seeds.extend(random_seeds2)
        
    if rc0[0]%2!=0:
        random_seeds1=random.sample(range(1,seeds_range+1),seeds_range)
        center_seq.extend(random_seeds1)
    
    if len(center_seq)==rc0[0]*rc0[1]:
        print('<种子点编序正确>')
    
    center_seq_mat=np.array(center_seq,dtype=int).reshape(rc0[0],rc0[1])
    #################################################
    
    
    ###第一行
    print('开始像素归类...\n第一行种子点像素归并中...')
    for i in range(1,rc0[1]-1):
        loc_cen=center_point_list[0,i]
        flag=center_seq_mat[0,i]
        img_tmp=image[:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1,:]
        rc=img_tmp.shape
        
        ###空间权重
        spa_1=np.abs(np.tile(np.linspace(-N,N,2*N+1),(rc[0],1)))
        spa_2=np.abs(np.tile(np.linspace(-loc_cen[0],N,rc[0]).reshape(rc[0],1),(1,2*N+1)))
        spa_weight_ori=spa_1**2+spa_2**2
        
        ###光谱权重
        spec_weigth=((img_tmp-np.tile(image[loc_cen[0],loc_cen[1],:].reshape(1,1,rc[2]),(rc[0],rc[1],1)))**2).sum(axis=2)

        ###综合权重
        weight_all=np.sqrt(spa_weight_ori/N**2+spec_weigth/img_tmp.shape[2]**2)
        
        ###与已有权重比较，取小者位置
        loc_change=np.where(weight_all<dec_weight[:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1])
        
        ###更新像素标签
        dec_mat_tmp=dec_mat[:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1]
        dec_mat_tmp[loc_change]=flag
        dec_mat[:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1]=dec_mat_tmp
        
        ###更新综合权重
        dec_weight_tmp=dec_weight[:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1]
        dec_weight_tmp[loc_change]=weight_all[loc_change]
        dec_weight[:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1]=dec_weight_tmp
    
    ###最后一行
    print('<第一行种子点像素归并完毕>\n最后一行种子点像素归并中...')
    for i in range(1,rc0[1]-1):
        loc_cen=center_point_list[-1,i]
        flag=center_seq_mat[-1,i]
        img_tmp=image[loc_cen[0]-N:,loc_cen[1]-N:loc_cen[1]+N+1,:]
        rc=img_tmp.shape
        
        ###空间权重
        spa_1=np.abs(np.tile(np.linspace(-N,N,2*N+1),(rc[0],1)))
        spa_2=np.abs(np.tile(np.linspace(-N,rc[0]-N-1,rc[0]).reshape(rc[0],1),(1,2*N+1)))
        spa_weight_ori=spa_1**2+spa_2**2
        
        ###光谱权重
        spec_weigth=((img_tmp-np.tile(image[loc_cen[0],loc_cen[1],:].reshape(1,1,rc[2]),(rc[0],rc[1],1)))**2).sum(axis=2)
        
        ###综合权重
        weight_all=np.sqrt(spa_weight_ori/N**2+spec_weigth/img_tmp.shape[2]**2)
        
        ###与已有权重比较，取小者位置
        loc_change=np.where(weight_all<dec_weight[loc_cen[0]-N:,loc_cen[1]-N:loc_cen[1]+N+1])
        
        ###更新像素标签
        dec_mat_tmp=dec_mat[loc_cen[0]-N:,loc_cen[1]-N:loc_cen[1]+N+1]
        dec_mat_tmp[loc_change]=flag
        dec_mat[loc_cen[0]-N:,loc_cen[1]-N:loc_cen[1]+N+1]=dec_mat_tmp
        
        ###更新综合权重
        dec_weight_tmp=dec_weight[loc_cen[0]-N:,loc_cen[1]-N:loc_cen[1]+N+1]
        dec_weight_tmp[loc_change]=weight_all[loc_change]
        dec_weight[loc_cen[0]-N:,loc_cen[1]-N:loc_cen[1]+N+1]=dec_weight_tmp
   
    ###第一列
    print('<最后一行种子点像素归并完毕>\n第一列种子点像素归并中...')
    for i in range(1,rc0[0]-1):
        loc_cen=center_point_list[i,0]
        flag=center_seq_mat[i,0]
        img_tmp=image[loc_cen[0]-N:loc_cen[0]+N+1,:loc_cen[1]+N+1,:]
        rc=img_tmp.shape
        
        ###空间权重
        spa_1=np.abs(np.tile(np.linspace(-N,N,2*N+1).reshape(rc[0],1),(1,rc[1])))
        spa_2=np.abs(np.tile(np.linspace(-(rc[1]-N),N,rc[1]).reshape(1,rc[1]),(rc[0],1))) 
        spa_weight_ori=spa_1**2+spa_2**2
        
        ###光谱权重
        spec_weigth=((img_tmp-np.tile(image[loc_cen[0],loc_cen[1],:].reshape(1,1,rc[2]),(rc[0],rc[1],1)))**2).sum(axis=2)
        
        ###综合权重
        weight_all=np.sqrt(spa_weight_ori/N**2+spec_weigth/img_tmp.shape[2]**2)
        
        ###与已有权重比较，取小者位置
        loc_change=np.where(weight_all<dec_weight[loc_cen[0]-N:loc_cen[0]+N+1,:loc_cen[1]+N+1])
        
        ###更新像素标签
        dec_mat_tmp=dec_mat[loc_cen[0]-N:loc_cen[0]+N+1,:loc_cen[1]+N+1]
        dec_mat_tmp[loc_change]=flag
        dec_mat[loc_cen[0]-N:loc_cen[0]+N+1,:loc_cen[1]+N+1]=dec_mat_tmp
       
        ###更新综合权重
        dec_weight_tmp=dec_weight[loc_cen[0]-N:loc_cen[0]+N+1,:loc_cen[1]+N+1]
        dec_weight_tmp[loc_change]=weight_all[loc_change]
        dec_weight[loc_cen[0]-N:loc_cen[0]+N+1,:loc_cen[1]+N+1]=dec_weight_tmp
   
    ###最后一列
    print('<第一列种子点像素归并完毕>\n最后一列种子点像素归并中...')
    for i in range(1,rc0[0]-1):
        loc_cen=center_point_list[i,0]
        flag=center_seq_mat[i,0]
        img_tmp=image[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:,:]
        rc=img_tmp.shape
        
        ###空间权重
        spa_1=np.abs(np.tile(np.linspace(-N,N,2*N+1).reshape(rc[0],1),(1,rc[1])))
        spa_2=np.abs(np.tile(np.linspace(-N,rc[1]-N-1,rc[1]).reshape(1,rc[1]),(rc[0],1)))
        spa_weight_ori=spa_1**2+spa_2**2
        
        ###光谱权重
        spec_weigth=((img_tmp-np.tile(image[loc_cen[0],loc_cen[1],:].reshape(1,1,rc[2]),(rc[0],rc[1],1)))**2).sum(axis=2)
        
        ###综合权重
        weight_all=np.sqrt(spa_weight_ori/N**2+spec_weigth/img_tmp.shape[2]**2)
        
        ###与已有权重比较，取小者位置
        loc_change=np.where(weight_all<dec_weight[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:])
        
        ###更新像素标签
        dec_mat_tmp=dec_mat[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:]
        dec_mat_tmp[loc_change]=flag
        dec_mat[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:]=dec_mat_tmp
        
        ###更新综合权重
        dec_weight_tmp=dec_weight[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:]
        dec_weight_tmp[loc_change]=weight_all[loc_change]
        dec_weight[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:]=dec_weight_tmp
   

    ###中心处
    print('<最后一列种子点像素归并完毕>\n中心处主要部分种子点像素归并中...')
    
    ###空间权重
    spa_1=np.abs(np.tile(np.linspace(-N,N,2*N+1).reshape(1,2*N+1),(2*N+1,1)))
    spa_2=np.abs(np.tile(np.linspace(-N,N,2*N+1).reshape(2*N+1,1),(1,2*N+1)))
    spa_weight_ori=spa_1**2+spa_2**2
    
    for i in range(1,rc0[0]-1):
        for j in range(1,rc0[1]-1):
            loc_cen=center_point_list[i,j]
            flag=center_seq_mat[i,j]
            img_tmp=image[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1,:]
            rc=img_tmp.shape
#            print(i,j)
            ###光谱权重
            spec_weigth=((img_tmp-np.tile(image[loc_cen[0],loc_cen[1],:].reshape(1,1,rc[2]),(2*N+1,2*N+1,1)))**2).sum(axis=2)
            spec_weigth=spec_weigth/img_tmp.shape[2]
            
            ###综合权重
            weight_all=np.sqrt(spa_weight_ori/N**2+spec_weigth/(spec_weigth.max())**2)
            
            ###与已有权重比较，取小者位置
            loc_change=np.where(weight_all<dec_weight[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1])
            
            ###更新像素标签
            dec_mat_tmp=dec_mat[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1]
            dec_mat_tmp[loc_change]=flag
            dec_mat[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1]=dec_mat_tmp
            
            ###更新综合权重
            dec_weight_tmp=dec_weight[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1]
            dec_weight_tmp[loc_change]=weight_all[loc_change]
            dec_weight[loc_cen[0]-N:loc_cen[0]+N+1,loc_cen[1]-N:loc_cen[1]+N+1]=dec_weight_tmp
            
#            current_length=(i-1)*(rc[0]-2)+j
#            p.update(current_length)
            
    return dec_mat


def stat(links_mat):
  
    """
    #八连通性数组各元素个数统计函数；
    #输入为连通性数组；
    #返回值为个数最多的元素值；
    note：个数最多的元素应少于两个，且不为0，否则返回连通性数组原始中心值
    """
    links_mat2=copy.deepcopy(links_mat)
    links_mat1=list(copy.deepcopy(links_mat).reshape(-1))
    del links_mat1[4]
    
    data_uniuqe=list(set(links_mat1))
    sta_mat=np.zeros((len(data_uniuqe),2))
    sta_mat[:,0]=data_uniuqe
    for i in range(len(data_uniuqe)):
        
        sta_mat[i,1]=links_mat1.count(data_uniuqe[i])
    
    max_flag=sta_mat[:,1].argsort()[-1]
    
    ###个数最多的元素应少于两个，且不为0，否则返回连通性数组原始中心值
    if (sta_mat[:,1]==sta_mat[max_flag,1]).sum()<2 and sta_mat[max_flag,0]!=0:
        
        links_mat2[1,1]=sta_mat[max_flag,0]
    
    return links_mat2[1,1]


def Links_boost(decision_mat):
    """
    #连通性增强函数；
    #输入为原始超像素分割图像；
    note：超像素分割噪点一般存在在超像素图像四周，因此为减少计算量，仅对超像素图像
          四周进行连通性增强；
    """
    decision_mat[np.where(decision_mat==0)]=117
    
    dec_1row=decision_mat[:20,:]
    dec_inf_row=decision_mat[-20:,:]
    dec_1col=decision_mat[:,:20]
    dec_inf_col=decision_mat[:,-20:] 
    links_mats=[dec_1row,dec_inf_row,dec_1col,dec_inf_col]
#    links_mats_pro=[]
    
    for l_mat in links_mats:
        
        rc=l_mat.shape
        links_tmp=copy.deepcopy(l_mat)
        links_tmp=np.column_stack((np.zeros(rc[0]),links_tmp,np.zeros(rc[0])))
        links_tmp=np.row_stack((np.zeros(rc[1]+2),links_tmp,np.zeros(rc[1]+2)))
        rc=links_tmp.shape
        
        for i in range(1,rc[0]-1):
            for j in range(1,rc[1]-1):
                links_mat_pro=links_tmp[i-1:i+2,j-1:j+2]
                num_unique=np.unique(links_mat_pro)
                
                ###判断连通性数组是否存在两个及以上的不同元素，否则，无连通性增强必要；
                if num_unique.shape[0]!=1:
                    l_mat[i-1,j-1]=stat(links_mat_pro)
           
    return decision_mat


def GET_seeds(image,N):
    """
    #图像种子点获取函数；
    #image:待分割图像；
    #N：初始超像素分辨率倍数（相对于原始分辨率），即搜索窗口大小；
    #获取种子点，返回值为种子点索引
    """
    
    row=image.shape[0]
    column=image.shape[1]
    size_super_x=int(column/N)
    
    ###当列不为N的整数倍时，补零；
    if column%N>0:
        size_super_x=size_super_x+1
        image=np.column_stack((image,np.zeros((row,N-int(column%N),image.shape[2]))))
    size_super_y=int(row/N)
    
    ###当行不为N的整数倍时，补零；
    if row%N>0:
        size_super_y=size_super_y+1
        image=np.row_stack((image,np.zeros((N-int(row%N),image.shape[1],image.shape[2]))))
    
    ###初始化种子点矩阵
    center_mat_list=np.zeros((size_super_y,size_super_x),dtype=object)
   
    
    for i in range(1,size_super_y+1):
        for j in range(1,size_super_x+1):
            center_mat=image[(i-1)*N:i*N,(j-1)*N:j*N,:]
            center_mat_list[i-1,j-1]=center_mat
    
    ###种子点计算     
    center_point_list=list(map(grad_min,center_mat_list.reshape(size_super_x*size_super_y)))###获取窗口最小梯度位置
    
    flag=0
    if column%N>0:
        ###当x方向为非整数个超像素单元，y方向为整数个超像素单元时；
        ###当x方向剩余像素大于等于半个超像素大小时，视为一个超像素，此时重新计算该处种子点；
        flag=1
        center_mat_list=[]
        if column%N>=0.5*N:
            loc=[]
            for i in range(1,size_super_y+1):
                    center_mat=image[(i-1)*N:i*N,(size_super_x-1)*N:column,:]
                    center_mat_list.append(center_mat)
                    loc.append(i*size_super_x-1)
                    
            grad=list(map(grad_min,center_mat_list))
            for j in range(len(loc)):
                center_point_list[loc[j]]=grad[j]
                
            plus_0=list(np.ones((size_super_x)))
            center_point_list=np.array(plus_0+center_point_list,dtype=object).reshape(size_super_y+1,size_super_x)[1:,:]
        
        ###当x方向剩余像素小于半个超像素大小时，将剩余像素与前一种子点范围合并，并重新计算该处中子点；
        else:
            loc1=[]
            loc2=[]
            for i in range(1,size_super_y+1):
                    center_mat=image[(i-1)*N:i*N,(size_super_x-2)*N:column,:]
                    center_mat_list.append(center_mat)
                    loc1.append(i*size_super_x-1)
                    loc2.append(i*(size_super_x-1)-1)
                    
            grad=list(map(grad_min,center_mat_list))
            flag=1
            for j in range(len(loc1)):
                del center_point_list[loc1[len(loc1)-1-j]]
            for j in range(len(loc2)):
                center_point_list[loc2[j]]=grad[j]
                
            plus_0=list(np.ones((size_super_x-1)))
            center_point_list=np.array(plus_0+center_point_list,dtype=object).reshape(size_super_y+1,size_super_x-1)[1:,:]
    
    if row%N>0:
        
        ###当x方向为整数个超像素，y方向上为非整数个超像素；
        ###当y方向剩余像素数大于等于半个超像素大小时，视为一个超像素，重新计算该处种子点；
        center_mat_list=[]
        if row%N>=0.5*N:
            loc=[]
            for i in range(1,size_super_x+1):
                center_mat=image[(size_super_y-1)*N:row,(i-1)*N:i*N,:]
                center_mat_list.append(center_mat)
                loc.append((size_super_y-1)*size_super_x+i-1)
            
            grad=list(map(grad_min,center_mat_list))
            #print(size_super_x)
            if flag==1:
#                for i in range(center_point_list.shape[1]):
                center_point_list[-1,:]=grad
            else:
                for j in range(len(loc)):
                    center_point_list[loc[j]]=grad[j]
                plus_0=list(np.ones((size_super_x)))
                center_point_list=np.array(plus_0+center_point_list,dtype=object).reshape(size_super_y+1,size_super_x)[1:,:]
        
        ###当y方向剩余像素数小于半个超像素大小，将其与上一种子点范围合并，重新计算该处种子点；
        else:
            loc1=[]
            loc2=[]
            for i in range(1,size_super_x+1):
                center_mat=image[(size_super_y-2)*N:row,(i-1)*N:i*N,:]
                center_mat_list.append(center_mat)
                loc2.append((size_super_y-2)*size_super_x+i-1)
                
            grad=list(map(grad_min,center_mat_list))
            if flag==1:
                del center_point_list[-1,:]
                #for i in range(center_point_list.shape[1]):
                center_point_list[-1,:]=grad
            else:
                del center_point_list[(size_super_y-1)*size_super_x:]
                for j in range(len(loc2)):
                    center_point_list[loc2[j]]=grad[j]
                    
                plus_0=list(np.ones((size_super_x)))
                center_point_list=np.array(plus_0+center_point_list,dtype=object).reshape(size_super_y,size_super_x)[1:,:]
    
    ###当x和y方向均为行列数均为N的整数倍时，不需改变
    if row%N==0 and column%N==0:
        plus_0=list(np.ones((size_super_x)))
        center_point_list=np.array(plus_0+center_point_list,dtype=object).reshape(size_super_y+1,size_super_x)[1:,:]
        
    global_index=Get_index_global(center_point_list,N)         
    
#    global_index=global_index.reshape(global_index.shape[0]*global_index.shape[1])
#    for loc in global_index:
#        image[loc[0],loc[1],:]=255
##    print(center_point_list.shape)
#    for i in range(center_point_list.shape[0]):
#        image[i*N,:,:]=255
#    for i in range(center_point_list.shape[1]):
#        image[:,i*N,:]=255
    return  global_index
 
def seeds_reset(image,image_super,seeds_ori,N):
    
    rc_seeds=seeds_ori.shape
    seeds_new=copy.deepcopy(seeds_ori)
    d=image.shape[2]
    for i in range(1,rc_seeds[0]-1):
        for j in range(1,rc_seeds[1]-1):
            loc_seed=seeds_ori[i,j]
            seed_mat_ori=image_super[loc_seed[0]-N:loc_seed[0]+N+1,loc_seed[1]-N:loc_seed[1]+N+1]
            
            sup_traget=image_super[loc_seed[0],loc_seed[1]]
            seed_mat=np.zeros((2*N+1,2*N+1,d))
            seed_mat[np.where(seed_mat_ori==sup_traget),:]=1
            seed_mat=seed_mat*image[loc_seed[0]-N:loc_seed[0]+N+1,loc_seed[1]-N:loc_seed[1]+N+1,:]
            
            seed_loc_rel=np.array(grad_min(seed_mat))
            flag1=loc_seed[0]-N+seed_loc_rel[0]
            flag2=loc_seed[1]-N+seed_loc_rel[1]
            
            if flag1 in range(10,1020) and flag2 in range(10,569):
                seeds_new[i,j]=[loc_seed[0]-N+seed_loc_rel[0],loc_seed[1]-N+seed_loc_rel[1]]
  
    return seeds_new

def Boundary(links_mat):
    filter_mat=np.array([1,-1])
    rc=links_mat.shape
    for i in range(rc[0]-1):
        for j in range(rc[1]-1):
            links_mat[i,j]=(links_mat[i,j:j+1]*filter_mat).sum()
            
    return links_mat

def SLIC(image,N,iter_numb):
    
    i=0
    global_index=GET_seeds(image,N)
    
    while i <iter_numb:
        
        print('第%s次迭代...'%(i+1))
        decision_mat=DEC(image,N,global_index)           
              
        global_index=seeds_reset(image,decision_mat,global_index,N)
        i=i+1
    links_mat=Links_boost(copy.deepcopy(decision_mat))  
    return links_mat
