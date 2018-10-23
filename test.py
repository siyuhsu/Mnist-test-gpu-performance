import gzip
import pickle
import random
import numpy as np
from numba import jit


def load_data():
    f = gzip.open('d:\mnist.pkl.gz','rb')
    x,y,z = pickle.load(f,encoding='iso-8859-1')
    print("x = ",len(x))
    print(x)
    return (x,y,z)

def load_data_wrapper():
    tr_d,va_d,te_d = load_data()    #载入数据：训练集，验证集，测试集
    trainning_inputs = [np.reshape(x,(784,1)) for x in tr_d[0]] #训练集 x：784*1
    training_results = [vectorized_result(y) for y in tr_d[1]]  #训练集 y
    training_data = [(x,y) for x,y in zip(trainning_inputs,training_results)]   #训练集（x，y）
    validation_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]    #验证集 x：784*1
    validation_data = [(x,y) for x,y in zip(validation_inputs,va_d[1])]#验证集 （x，y）
    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]  #测试集 x: 784*1
    test_data = [(x,y) for x,y in zip(test_inputs,te_d[1])] #测试集 （x，y）
    return (training_data,validation_data,test_data)    #返回数据集：训练集，验证集，测试集

def vectorized_result(y):   #向量化标签
    e = np.zeros((10,1))
    e[y]=1.0    #初始化为1
    return e

class NetWork(object):
    def __init__(self,sizes):
        self.n_layer = len(sizes)   #隐藏层层数
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for  x,y in zip(sizes[:-1],sizes[1:])]

    def sigmoid(self,z):    #激活函数
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):      #sigmoid函数的导数
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def feedforward(self,activation):      #前向传播
        for b,w in zip(self.biases,self.weights):
            activation = self.sigmoid(np.dot(w,activation) + b)
        return activation

    def cost(self,activation ,y):   #成本函数
        activation = self.feedforward(activation)
        z = activation - y
        b = z*z
        b = sum(b)
        return b
    def cost_derivative(self,a,y):  #cost的导数
        return (a-y)

    def backPropagation(self,x,y):      #反向传播：对w,b分别求导
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]    #d'b
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]   #d'a
        activation = x
        activations=[x]
        zs = []
        for b,w in zip(self.biases, self.weights):  #对最后一层求导
            z = np.dot(w,activation) + b
            zs.append(z)
            activation= self.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1],y)*self.sigmoid_prime(zs[-1])  #d'a
        delta_nabla_b[-1]= delta    #d'b
        delta_nabla_w[-1]=  np.dot(delta,activations[-2].transpose())   #d'w

        for l in range(2,self.n_layer):     #对1-(n-1)层求导
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].transpose(),delta) * self.sigmoid_prime(z)
            delta_nabla_b[-l]=delta
            delta_nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (delta_nabla_b,delta_nabla_w)

    def update_mini_batch(self,mini_batch,eta):     #更新每一个minibatch的参数
        sum_delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        sum_delta_nabla_w = [np.zeros(w.shape)for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backPropagation(x,y)
            sum_delta_nabla_b = [db + sdb for db,sdb in zip(delta_nabla_b,sum_delta_nabla_b)]
            sum_delta_nabla_w = [dw + sdw for dw ,sdw in zip(delta_nabla_w,sum_delta_nabla_w)]
        self.biases = [b -(eta/len(mini_batch))*nb for b ,nb in zip(self.biases,sum_delta_nabla_b)]
        self.weights = [w -(eta/len(mini_batch))*nw for w,nw in zip(self.weights,sum_delta_nabla_w)]
    def evaluate(self,test_data):       #使用测试集评估预测结果
        results = [(np.argmax(self.feedforward(x)),y)  for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in results)
    def SGD(self,training_data,epchos,mini_batch_size,eta,test_data=None):  #随机梯度下降
        if test_data:
            n_test = len(test_data)
        n_data = len(training_data)

        for j in range(epchos):     #进行迭代
            random.shuffle(training_data)
            mini_batchs =[training_data[k:k+mini_batch_size] for k in range(0,n_data,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                y_result = self.evaluate(test_data)
                print("Epoches :{0}-->{1}/{2}".format(j,y_result,n_test))
            else:
                print("Epoches:{0}:-->".format(j))

mytraining_data,myvalidation_data,mytest_data = load_data_wrapper()
net = NetWork([784,512,28,10])  #一个输入层，两个隐藏层，一个输出层，维度分别为：784，512，28，10
net.SGD(mytraining_data,10,100,0.1,test_data=mytest_data)   #将训练集喂入SGD,10此迭代，每个minibatch 100组数据，共500个minibatch，学习率为0.1



#net = NetWork([2,3,3,5])
#w = net.pr()


#print(len(w))
#print(w[0])


