import TwoLayerNet
import numpy as np

#hyper
iters_num=1000
lerning_rate=0.1
batch_size=5
batch_size_asu=59
batch_size_ida=8
batch_size_michi=86
batch_size_nobu=86
batch_size_other=23
batch_size_pato=109
batch_size_ru=44
batch_size_shachi=57
batch_size_tanaka=51
batch_size_yu=36

net=TwoLayerNet.TowLayerNet(2500,100,10)

x=[]
t=[]

x_train,t_train=net.load_train("./faces50/asu/",batch_size_asu,np.array([1,0,0,0,0,0,0,0,0,0]))
x=x_train
t=t_train
x_train,t_train=net.load_train("./faces50/ida/",batch_size_ida,np.array([0,1,0,0,0,0,0,0,0,0]))
x.extend(x_train)
t.extend(t_train)
x_train,t_train=net.load_train("./faces50/michi/",batch_size_michi,np.array([0,0,1,0,0,0,0,0,0,0]))
x.extend(x_train)
t.extend(t_train)
x_train,t_train=net.load_train("./faces50/nobu/",batch_size_nobu,np.array([0,0,0,1,0,0,0,0,0,0]))
x.extend(x_train)
t.extend(t_train)
x_train,t_train=net.load_train("./faces50/other/",batch_size_other,np.array([0,0,0,0,1,0,0,0,0,0]))
x.extend(x_train)
t.extend(t_train)
x_train,t_train=net.load_train("./faces50/pato/",batch_size_pato,np.array([0,0,0,0,0,1,0,0,0,0]))
x.extend(x_train)
t.extend(t_train)
x_train,t_train=net.load_train("./faces50/ru/",batch_size_ru,np.array([0,0,0,0,0,0,1,0,0,0]))
x.extend(x_train)
t.extend(t_train)
x_train,t_train=net.load_train("./faces50/shachi/",batch_size_shachi,np.array([0,0,0,0,0,0,0,1,0,0]))
x.extend(x_train)
t.extend(t_train)
x_train,t_train=net.load_train("./faces50/tanaka/",batch_size_tanaka,np.array([0,0,0,0,0,0,0,0,1,0]))
x.extend(x_train)
t.extend(t_train)
x_train,t_train=net.load_train("./faces50/yu/",batch_size_yu,np.array([0,0,0,0,0,0,0,0,0,1]))
x.extend(x_train)
t.extend(t_train)

x=np.array(x)
t=np.array(t)

for i in range(iters_num):
    print(str(i)+"/"+str(iters_num))
    batch_mask=np.random.choice(x.shape[0],batch_size)
    x_batch=x[batch_mask]
    t_batch=t[batch_mask]
    grad=net.numericalGradient(x_batch,t_batch)
    net.params['W1']-=lerning_rate*grad['W1']
    net.params['b1']-=lerning_rate*grad['b1']
    net.params['W2']-=lerning_rate*grad['W2']
    net.params['b2']-=lerning_rate*grad['b2']
    print("Err:"+str(net.loss(x_batch,t_batch)))

net.getGradient()