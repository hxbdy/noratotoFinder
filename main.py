import TwoLayerNet
import numpy as np
from PIL import Image

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

#hyper
iters_num=50000
lerning_rate=0.01
batch_size=1

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
#net.setGradient()

x=[]
t=[]
loss=[]

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
'''
batch_mask=np.random.choice(x.shape[0],batch_size)
x_batch=x[batch_mask]
t_batch=t[batch_mask]
img=x_batch[0]*255.0
img=img.reshape(50,50)
img_show(img)
print(t_batch[0])
exit(1)
'''
for i in range(iters_num):
    print(str(i+1)+"/"+str(iters_num))
    batch_mask=np.random.choice(x.shape[0],batch_size)
    x_batch=x[batch_mask]
    t_batch=t[batch_mask]
    grad=net.gradient(x_batch,t_batch)
    net.params['W1']-=lerning_rate*grad['W1']
    net.params['b1']-=lerning_rate*grad['b1']
    net.params['W2']-=lerning_rate*grad['W2']
    net.params['b2']-=lerning_rate*grad['b2']
    l=net.loss(x_batch,t_batch)
    loss.append(l)
    print("Err:"+str(l))

net.saveLoss(loss)
net.getGradient()