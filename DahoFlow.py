import DahoLayers
import DahoLoss
import numpy as np

x = np.array([[1,2,3,4,5,6], [2,3,4,5,6,7], [3,4,5,6,7,8], [4,5,6,7,8,9]])
y = np.array([[21],[27],[33],[39]]) 

net = DahoLayers.Model([
    DahoLayers.Dense(1),
    DahoLayers.Relu()
])

net.fit(x, y, optimizer=DahoLayers.SGD(lr=0.01), loss=DahoLoss.MSE(), epochs=10)
print (net(x))
