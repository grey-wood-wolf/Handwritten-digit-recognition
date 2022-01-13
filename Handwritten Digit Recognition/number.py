import os
import struct
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from sklearn import  svm
from PIL import Image


def load_mnist(path, kind='train'):
    # 路径，相当于字符串拼接
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def train(train_num):
    X_train, y_train = load_mnist('.//dataset//MNIST//raw//', kind='train')
    # 加载训练集
    #X = preprocessing.StandardScaler().fit_transform(X_train)
    X=X_train
    X_train = X[0:train_num]  # 训练60000张
    y_train = y_train[0:train_num]

    dt = datetime.now()
    print('time is ' + dt.strftime('%Y-%m-%d %H:%M:%S'))

    model_svc = svm.SVC(kernel='rbf', gamma='scale')
    model_svc.fit(X_train, y_train)

    dt = datetime.now()
    print('time is ' + dt.strftime('%Y-%m-%d %H:%M:%S'))

    return model_svc


def test(model_svc, test_num):
    test_images, test_labels = load_mnist('.//dataset//MNIST//raw//', kind='t10k')  # 加载测试集
    #x = preprocessing.StandardScaler().fit_transform(test_images)
    x=test_images
    x_test = x[0:test_num]
    y_test = test_labels[0:test_num]

    print(model_svc.score(x_test, y_test))  # 根据训练的模型，进行分类得分计算
    #return model_svc.score(x_test, y_test)
    return test_images, test_labels, x


def pred(model_svc, pred_num, test_images, test_labels, x):
    y_pred = model_svc.predict(x[9690 - pred_num:9690])  # 进行预测,能得到一个结果
    print(y_pred)

    X_show = test_images[9690 - pred_num:9690]
    #Y_show = test_labels[9690 - pred_num:9690]

    for i in range(pred_num):
        x_show = X_show[i].reshape(28, 28)
        plt.subplot(1, pred_num, i + 1)
        plt.imshow(x_show, cmap=plt.cm.gray_r)
        plt.title(str(y_pred[i]))
        plt.axis('off')
    plt.show()


model = train(20000)#训练个数
test_images, test_labels, x = test(model,9900)
pred(model,9,test_images, test_labels, x)

'''
#下列测试自己的图片(自己的数字效果不好，但用MNIST效果不错)
image_file = Image.open(".//mynum.png") # open colour image
image_file = image_file.resize((28,28))
image_file = image_file.convert('L') # convert image to black and white
image_file = np.array(image_file,dtype=np.uint8)
image_file = image_file.reshape(1,784)
mypred = model.predict(image_file)
print(mypred)
plt.imshow(Image.open(".//mynum.png"), cmap=plt.cm.gray_r)
plt.title(str(mypred[0]))
plt.show()
'''

'''
scores = []
tra=[]
for i in range(2000,60000)[::2000]:
    model = train(i)
    scores += [test(model,1000)]
    tra +=[i]
plt.scatter(tra, scores)
plt.show()
'''
