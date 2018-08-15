import TwoLayerNet
import numpy as np

net=TwoLayerNet.TowLayerNet(2500,50,10)
net.setGradient()
x_test=net.load_test("./faces50/asu/",50)
x=np.array(x_test)
x_res=net.predict(x)
#print(x_res[1])
#net.judge(x_res[1])
for i in range(164):
    net.judge(x_res[i])
