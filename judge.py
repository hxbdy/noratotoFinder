import TwoLayerNet
import numpy as np

net=TwoLayerNet.TowLayerNet(2500,100,10)
net.setGradient()
x_test=net.load_test("./faces50/test/",164)
x=np.array(x_test)
x_res=net.predict(x)
#print(x_res[1])
#print(x_res)
for i in range(164):
    print(str(i)+".jpg ",end="")
    net.judge(x_res[i])
