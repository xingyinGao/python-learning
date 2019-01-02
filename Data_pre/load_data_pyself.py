from sklearn import datasets

def load_data_pyself(key='iris'):
    """
    #python自带数据集加载：
    #      iris：鸢尾花数据集；
    #      breat_cancer:乳腺癌数据集；
    #      digits：手写数字数据集；
    #      diabetes：糖尿病数据集；
    #      boston：波士顿房价数据集；
    #      linnerud：体能数据集
    """
    if key=='iris':
        return datasets.load_iris()
    elif key=='breat_cancer':
        return datasets.load_breat_cancer()
    elif key=='digits':
        return datasets.load_digits()
    elif key=='diabetes':
        return datasets.load_diabetes()
    elif key=='boston':
        return datasets.load_boston()
    elif key=='linnerud':
           return datasets.load_linnerud() 
    else:
          print(key+'数据集不存在！请检查参数') 
    return  
