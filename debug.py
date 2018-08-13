import TwoLayerNet
import numpy as np

net=TwoLayerNet.TowLayerNet(4,3,2)
print(net.params['W1'])
net.setGradient()
print(net.params['W1'])