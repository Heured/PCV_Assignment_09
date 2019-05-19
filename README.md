# PCV_Assignment_09
KNN
## K邻近分类法(KNN)
### 简单KNN二维示例
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
  ![emmmm](https://github.com/Heured/PCV_Assignment_09/blob/master/ImgToShow/01.png)  
  
每个示例中，不同颜色代表类标记，正确分类的点用星号表示，分类错误的点用圆点表示，曲线是分类器的决策边界。  
  
  
### 用稠密SIFT作为图像特征
  稠密SIFT：VLFeat实现了一个快速密集的SIFT版本，称为vl_dsift。 该函数大致相当于在固定的比例和方向上在密集的位置上运行SIFT。 这种类型的特征描述符通常用于对象分类。  
  该算法首先将表达目标的矩形区域分成相同大小的矩形块,计算每一个小块的SIFT特征,再对各个小块的稠密SIFT特征在中心位置进行采样,建模目标的表达.然后度量两个图像区域的不相似性,先计算两个区域对应小块的Bhattacharyya距离,再对各距离加权求和作为两个区域间的距离.因为目标所在区域靠近边缘的部分可能受到背景像素的影响,而区域的内部则更一致,所以越靠近区域中心权函数的值越大.最后提出了能适应目标尺度变化的跟踪算法.实验表明,本算法具有良好的性能.   
  用如下脚本实现在整幅图像上用一个规格的网格应用SIFT描述子可得到稠密SIFT的表示形式：  
  
```python
# -*- coding: utf-8 -*-
from PCV.localdescriptors import sift, dsift
from pylab import *
from PIL import Image

dsift.process_image_dsift('./gesture/empire.jpg','empire.dsift',90,40,True)
l,d = sift.read_features_from_file('empire.dsift')
im = array(Image.open('gesture/empire.jpg'))
sift.plot_features(im,l,True)
title('dense SIFT')
show()
```
结果如图：  
  ![emmmm](https://github.com/Heured/PCV_Assignment_09/blob/master/ImgToShow/02.png)  
  
### 手势识别
  生成dSIFT特征：  
  
```python
# -*- coding: utf-8 -*-
import os
from PCV.localdescriptors import sift, dsift
from pylab import  *
from PIL import Image

imlist=['gesture/train/C-uniform02.ppm','gesture/train/B-uniform01.ppm',
        'gesture/train/A-uniform01.ppm','gesture/train/Five-uniform01.ppm',
        'gesture/train/Point-uniform01.ppm','gesture/train/V-uniform01.ppm']

figure()
for i, im in enumerate(imlist):
    print (im)
    # 获取每幅图像的稠密SIFT特征
    dsift.process_image_dsift(im, im[:-3]+'dsift',90,40,True)
    l,d = sift.read_features_from_file(im[:-3]+'dsift')
    dirpath, filename=os.path.split(im)
    im = array(Image.open(im))
    #显示手势含义title
    titlename=filename[:-14]
    subplot(2,3,i+1)
    sift.plot_features(im,l,True)
    title(titlename)
show()
```
结果：  
  ![emmmm](https://github.com/Heured/PCV_Assignment_09/blob/master/ImgToShow/03.png)  
  
  
进行分类测试：  
  
```python
# -*- coding: utf-8 -*-
from PCV.localdescriptors import dsift
import os
from PCV.localdescriptors import sift
from pylab import *
from PCV.classifiers import knn

def get_imagelist(path):
    """    Returns a list of filenames for
        all jpg images in a directory. """

    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.ppm')]

def read_gesture_features_labels(path):
    # create list of all files ending in .dsift
    featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]
    # read the features
    features = []
    for featfile in featlist:
        l,d = sift.read_features_from_file(featfile)
        features.append(d.flatten())
    features = array(features)
    # create labels
    labels = [featfile.split('/')[-1][0] for featfile in featlist]
    return features,array(labels)

def print_confusion(res,labels,classnames):
    n = len(classnames)
    # confusion matrix
    class_ind = dict([(classnames[i],i) for i in range(n)])
    confuse = zeros((n,n))
    for i in range(len(test_labels)):
        confuse[class_ind[res[i]],class_ind[test_labels[i]]] += 1
    print ('Confusion matrix for')
    print (classnames)
    print (confuse)

filelist_train = get_imagelist('gesture/train')
filelist_test = get_imagelist('gesture/test')
imlist=filelist_train+filelist_test

# process images at fixed size (50,50)
for filename in imlist:
    featfile = filename[:-3]+'dsift'
    dsift.process_image_dsift(filename,featfile,10,5,resize=(50,50))

features,labels = read_gesture_features_labels('gesture/train/')
test_features,test_labels = read_gesture_features_labels('gesture/test/')
classnames = unique(labels)

# test kNN
k = 1
knn_classifier = knn.KnnClassifier(labels,features)
res = array([knn_classifier.classify(test_features[i],k) for i in
range(len(test_labels))])
# accuracy
acc = sum(1.0*(res==test_labels)) / len(test_labels)
print ('Accuracy:', acc)

print_confusion(res,test_labels,classnames)

```
  结果：  
  
```python
Accuracy: 0.8134715025906736
Confusion matrix for
['A' 'B' 'C' 'F' 'P' 'V']
[[26.  0.  2.  0.  1.  1.]
 [ 0. 26.  2.  1.  2.  1.]
 [ 0.  0. 26.  0.  0.  0.]
 [ 0.  4.  0. 37.  0.  0.]
 [ 0.  0.  2.  0. 17.  2.]
 [ 3.  1.  1.  0. 13. 25.]]
```
结果表示精确度为81%左右，且输出了一个混淆矩阵。  
(混淆矩阵是一个可以显示每类有多少个样本被分在每一类中的矩阵，他可以显示错误的分布情况，以及哪些类是经常相互‘混淆’的。)  
从结果中的混淆矩阵可以看出识别为V的图片有相当一部分也被识别为P。  

使用自定义测试数据所得结果：  
  
![emmm]()  
  
![emmm]()  
  

