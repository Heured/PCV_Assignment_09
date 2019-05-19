# PCV_Assignment_09
KNN
## K邻近分类法(KNN)
  KNN是在分类方法中最简单而且用的最多的一种方法。这种算法的原理是把分类的对象与训练集中已知类标记的所有对象做对比，并由k近邻对指派到哪个类进行投票。这种方法的缺点包括需要预设k值，k值得选择会影响分类的性能；如果训练集过大，搜索起来就非常慢。  
  实现最基本的KNN形式非常简单。给定训练样本集和对应的标记列表，下面的脚本将创建两个不同的二维点集，每个点集有两类，用Pickle模块保存创建的数据：
```python
# -*- coding: utf-8 -*-
from numpy.random import randn
import pickle
from pylab import *

# create sample data of 2D points
n = 200
# two normal distributions
class_1 = 0.6 * randn(n,2)
class_2 = 1.2 * randn(n,2) + array([5,1])
labels = hstack((ones(n),-ones(n)))
# save with Pickle
#with open('points_normal.pkl', 'w') as f:
with open('points_normal_test.pkl', 'wb') as f:
    pickle.dump(class_1,f)
    pickle.dump(class_2,f)
    pickle.dump(labels,f)
# normal distribution and ring around it
print ("save OK!")
class_1 = 0.6 * randn(n,2)
r = 0.8 * randn(n,1) + 5
angle = 2*pi * randn(n,1)
class_2 = hstack((r*cos(angle),r*sin(angle)))
labels = hstack((ones(n),-ones(n)))
# save with Pickle
#with open('points_ring.pkl', 'w') as f:
with open('points_ring_test.pkl', 'wb') as f:
    pickle.dump(class_1,f)
    pickle.dump(class_2,f)
    pickle.dump(labels,f)
    
print ("save OK!")
```
  用不同的文件名运行两次，一个用来训练一个用来测试。  
  训练：points_normal_train.pkl和points_ring_train.pkl  
  测试：points_normal_test.pkl和points_ring_test.pkl  
  

  再执行以下脚本作为示例  
  
```python
# -*- coding: utf-8 -*-
import pickle
from pylab import *
from PCV.classifiers import knn
from PCV.tools import imtools

pklist=['points_normal.pkl','points_ring.pkl']

figure()

# load 2D points using Pickle

for i, pklfile in enumerate(pklist):
    # [:-4]相当于复制文件名且把文件名的.pkl去掉(.pkl刚好为4个字符)
    # 导入训练数据
    with open(pklfile[:-4]+'_train.pkl', 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    # load test data using Pickle

    # 利用训练数据创建并训练KNN分类器模型
    model = knn.KnnClassifier(labels,vstack((class_1,class_2)))
    # 导入测试数据
    with open(pklfile[:-4]+'_test.pkl', 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)

    # test on the first point在测试数据集的第一个数据点上进行测试
    print (model.classify(class_1[0]))

    #define function for plotting定义绘图函数
    def classify(x,y,model=model):
        return array([model.classify([xx,yy]) for (xx,yy) in zip(x,y)])

    # lot the classification boundary绘制分类边界
    subplot(1,2,i+1)
    imtools.plot_2D_boundary([-6,6,-6,6],[class_1,class_2],classify,[1,-1])
    titlename=pklfile[:-4]
    title(titlename)
show()
```
下图便是执行以上脚本的结果：  
  ![emmmm]()  
  
每个示例中，不同颜色代表类标记，正确分类的点用星号表示，分类错误的点用圆点表示，曲线是分类器的决策边界。  
  
  
