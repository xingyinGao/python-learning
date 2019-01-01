# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:53:02 2018

@author: xingyinGao
"""
import numpy as np

class Node(object):
    def __init__(self,score=None):
        """
        #树的节点构造函数，其中，
        #score表示标签平均值，当其位于叶子节点时，表示该叶子节点类别；
        #left为左子节点，初始为None；
        #right为右子节点，初始为None；
        #feature为父节点最优分割特征，初始为None；
        #split为父节点最优分割特征分割值，初始为None；
        #mse为该父节点的分割子集方差之和，初始为None；
        #loc为当前父节点所在树中的位置，初始为None；
        #调用方式：nd=Node(score),表示创建树父节点；
        note：当该父节点左右子节点不存在时，该父节点为叶子节点；
        """
        self.score=score
        self.left=None
        self.right=None
        self.feature=None
        self.split=None
        self.mse=None
        self.loc=None
        
class RegressionTree(object):
    def __init__(self):
        """
        #树的初始化；
        #root表示树，初始为None；
        #hight为树深度；
        #error为预测误差，初始为无穷大；
        #res为预测结果；
        #bias为预测偏差；
        #r2为预测结果与实际值相关性
        #rules为最终训练规则；
        """
        self.root=None
        self.height=0
        self.error=np.inf
        self.res=None
        self.bias=None
        self.r2=None
        self.rules=[1,'<']
    
    def Random_split(self,X,y=None,size=0.5,hold_method=1):
        """
        #数据集分割函数；
        #输入有：
        #     X表示数据，行为数据个体，列为数据特征,必选；
        #     y表示数据个体标签，可选，默认为None；
        #     size样本分割参数，当为0-1开区间内小数时，表示主分割子集分割比例；
        #                      当为整数时，表示主分割子集样本个数;
        #                      默认为0.5；
        #     hold_method返回值选项，当其为0时，表示仅返回主分割子集；
        #                           当其为1时，主分割子集和次分割子集均返回；
        #                           默认为1；
        #返回值：为一个包含1-4个元素的列表；
        #        当返回值选项为0时，若输入X和y中，仅存在数据X，则仅返回X主分割子集；
        #        当返回值选项为0时，若输入X和y均存在，则顺序返回X的主分割子集、对应
        #                          标签数据y的主分割子集，两个列表元素；
        #        当返回值选项为1时，若输入X和y中，仅存在数据X，则顺序返回X的主、次
        #                          分割子集，两个列表元素；
        #        当返回值选项为1时，若输入X和y均存在，则顺序返回X的主分割子集、对应
        #                          标签数据y的主分割子集、X的次分割子集、对应标签
        #                          数据y的次分割子集，四个列表元素；
        #调用方式：rt=RegressionTree();rt.Random_split(X,y=None,size=0.5,hold_method=1)
        """
        
        split_sets=[]
        rc=X.shape[0]
        if type(size)==int:
            size_split=size
        else:
            size_split=int(size*rc)
            
        train_loc=np.random.choice(range(rc),size=size_split,replace=False)
        test_loc=[i for i in range(rc) if i not in train_loc]
        train_set=X[train_loc,:]
        test_set=X[test_loc,:]
        split_sets.append(train_set)
        split_sets.append(test_set)
        
        flag=None
        if type(y)==np.ndarray:
            flag=y.any()
        elif type(y)==list:
            flag=any(y)
            
        if flag and len(y)==rc:
            train_y=y[train_loc]
            test_y=y[test_loc]
            split_sets.append(train_y)
            split_sets.append(test_y)
        else:
            print('输入错误！y存在空值、0或False，或与X维度不匹配！')
            return 
        return split_sets

    def mse_cal(self,feature_label):
        """
        #最优特征衡量函数：分割子集最小化方差方法；
        #输入：样本数据及对应标签组成的列表，样本数据为多个样本多个特征组成的数组，
               其中行为样本个体，列表示特征，标签数据为一元数组，表示每个样本的类别；
        #调用方式：rt=RegressionTree();rt.mse_cal([sampel_data,sample_label])
        """
        feature_col=feature_label[0]
        feature_label=feature_label[1]
        mse_all=[]
        unique_values=np.unique(feature_col)[1:]
        for split_xpoint in unique_values:
            split_left_loc=np.where(feature_col<split_xpoint)[0]
            split_left_yset=feature_label[split_left_loc]
            split_right_loc=np.where(feature_col>=split_xpoint)[0]
            split_right_yset=feature_label[split_right_loc]
            
            mse_left=(split_left_yset**2).sum()-(split_left_yset.sum()**2).mean()
            mse_right=(split_right_yset**2).sum()-(split_right_yset.sum()**2).mean()
            mse=mse_left+mse_right
            split_yset_loc=[split_left_loc,split_right_loc]
            mse_all.append([mse,split_xpoint,split_yset_loc])
            
        mse_min=sorted(mse_all,key=lambda mse_elm:mse_elm[0])[0]
    
        return mse_min
        
    def feature_choose(self,X,y):
        """
        #树节点处样本集分割所需最优特征选择函数；
        #X为样本数据，行为样本个体，列表示特征;
        #y为样本对应标签，为一元数组,表示每个样本的类别；
        #每个特征值域范围至少有两个不同的值;
        #调用方式：rt=RegressionTree();rt.feature_choose(X,y)
        """
        features_sets=[[X[:,i],y] for i in range(X.shape[1]) if len(set(X[:,i]))>=2]
        if features_sets==[]:
            return None
        else:
            res_mse=list(map(self.mse_cal,features_sets))
            feature_par_best=sorted(res_mse,key=lambda mse_sin:mse_sin[0])[0]
            
            ##定位最优特征位置
            loc_num=0
            for res in res_mse:
                if res[0]==feature_par_best[0]:
                    break
                else:
                    loc_num=loc_num+1
            loc_best=loc_num
        
        return feature_par_best+[loc_best]
    
    def fit(self,X,y,depth_max=5,min_split=2):
        """
        #回归树拟合函数;
        #X：输入训练样本，行为样本个体，列为特征;
        #y：训练样本对应标签;
        #depth_max:树最大深度,默认为5;
        #min_split:最小分割子集样本数目，默认为2;
        #调用方式：rt=RegressionTree();rt.fit(self,X,y,depth_max=5,min_split=2);
        #note:当分割子集（左节点或右节点）仅存在一个标签类型样本，该节点不再继续分割；
             当分割子集不满足分割最优特征筛选条件，则该节点不再继续分割；
        """
        self.root=Node()
        que=[[0,self.root,list(range(len(y)))]]
        
        while que:
            
            depth,nd,index=que.pop(0)
            print(len(y[index]))
            if depth==depth_max:
                break
            if len(index)<min_split: 
                continue
            if len(set(y[index]))==1:
                
                continue
            feature_par=self.feature_choose(X[index,:],y[index])
            if feature_par==None:
                continue
            
            nd.loc=depth+1
            nd.mse=feature_par[0]
            nd.split=feature_par[1]
            nd.feature=feature_par[3]
            
            nd.left=Node(y[index][feature_par[2][0]].mean())
            nd.right=Node(y[index][feature_par[2][1]].mean())
            
            que.append([depth+1,nd.left,feature_par[2][0]])
            que.append([depth+1,nd.right,feature_par[2][1]])
            self.height=depth
            
        return print('训练完成！')
    
    def rules_get(self,root):
        """
        #迭代方式深度遍历搜索打印规则(先序遍历);
        #输入为树实例;
        #调用方式：rt=RegressionTree();rt.rules_get(rt.root);
        """
        while root.left==None and root.right==None:
            print('RULE%s:'%(self.rules[0]))
            for rule in self.rules[2:]:
                print('|feature%s:<key%s%.4f>,<mse=%.4f>|'%(rule[1],self.rules[1],
                                                          rule[2],rule[3]))
            self.rules[0]=self.rules[0]+1
            
            if self.rules[1]=='>':
                self.rules.pop()
            return 
        self.rules.append([root.loc,root.feature,root.split,root.mse])
        self.rules[1]='<'
        self.rules_get(root.left)
        self.rules[1]='>'
        self.rules_get(root.right)
        
       
    def predict_sin(self,samp_pred):
        """
        #单样本预测 ;
        #调用方式：rt=RegressionTree();rt.predict_sin(sample_predict);
        #查看预测结果：rt=RegressionTree();rt.res;
        """
        nd=self.root
        while nd.left and nd.right:
            if samp_pred[nd.feature]<nd.split:
                nd=nd.left
            else:
                nd=nd.right
              
        self.res=nd.score  
        
        return self.res
    
    def predict_multi(self,samp_pred):
        """
        #多样本预测;
        #行为样本个体，列为样本特征;
        #调用方式：rt=RegressionTree();rt.predict_multi(samples_predict);
        #查看预测结果：rt.res;
        """
        samps=[samp_pred[i,:] for i in range(samp_pred.shape[0])]
        self.res=list(map(self.predict_sin,samps))
        
#        return self.res
    
    def error_cal(self,y):        
        """
        #计算预测误差;
        #输入为测试样本标签;
        #调用方式:rt=RegressionTree();rt.error_cal(test_label);
        #查看准确率：rt.error;
        #查看预测偏差：rt.bias;
        """
        res_pre=np.ceil(np.array(self.res))
        real_y=y
        right_loc=res_pre-real_y
        self.error=(right_loc==0).sum()/len(y)
        self.bias=np.abs(np.array(self.res)-real_y).mean()
        self.r2=np.corrcoef(self.res,real_y)
