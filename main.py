import TwoLayerNet
import numpy as np

#load train


#load teach

#hyper
iters_num=10000
lerning_rate=0.1
batch_size=100


net=TwoLayerNet.TowLayerNet(22500,500,5)

x=[]
t=[]

x_train,t_train=net.load_train("./faces/asu/",2,np.array([1,0,0,0,0]))
x=x_train
t=t_train
#x_train,t_train=net.load_train("./faces/pato/",2,np.array([0,0,0,1,0]))
#x.extend(x_train)
#t.extend(t_train)

x=np.array(x)
t=np.array(t)

x2=np.random.rand(2,22500)
t2=np.array(([1,0,0,0,0],[0,1,0,0,0]))

print(x)
print(x2)
#print(t2)

grad=net.numericalGradient(x,t)
#print("=====result=====")
#print(net.params['W1']-lerning_rate*grad['W1'])