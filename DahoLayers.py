import numpy as np
import DahoOptimizer
import DahoLoss
import tqdm

class Activation:
    def __init__(self):
        pass

class Layer:
    def __init__(self):
        pass

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def __call__(self, layers, loss):
        grad = loss.backward()
        
        for layer in tqdm.tqdm(layers[::-1]):
            grad = layer.backward(grad)
            print()
            print("Layer:",layer)
            print("grad",grad)
            if isinstance(layer, Layer):
                layer.w -= layer.w_gradient * self.lr
                layer.b -= layer.b_gradient * self.lr
        

class Model:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output
    
    def fit(self, x, y, optimizer = SGD(), loss=DahoLoss.MSE(), epochs=10):
        l=0
        for epoch in range(1, epochs + 1):
            pred = self.__call__(x)
            l = loss(pred,y)
            optimizer(self.layers, loss)
            print(f"epoch {epoch} loss {l}")

class Dense(Layer):
    def __init__(self, dense_neurons_count):
        self.dense_neurons_count = dense_neurons_count
        self.initialized = False

    def __call__(self, x):
        self.input_layer = x
        if not self.initialized:
            self.w = np.random.rand(self.input_layer.shape[-1], self.dense_neurons_count)
            self.b = np.random.rand(self.dense_neurons_count)
            self.initialized = True
        return np.matmul(self.input_layer,self.w) + self.b

    def backward(self, gradient):
        self.w_gradient = np.sum(self.input_layer.T @ gradient,axis=0)
        self.b_gradient = np.sum(gradient, axis=0)
        return gradient @ self.w.T

                
class Sigmoid(Activation):
    def __call__(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad):
        return grad * (self.output * (1 - self.output))

class Relu(Activation):
    def __call__(self, x):
        self.output = np.maximum(0, x)   
        return self.output

    def backward(self, grad):
        return grad * np.clip(self.output, 0, 1)
