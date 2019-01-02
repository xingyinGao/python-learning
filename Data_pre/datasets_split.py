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
