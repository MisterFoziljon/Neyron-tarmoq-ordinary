import DahoLayers
import DahoLoss
import tqdm

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def __call__(self, model, loss):
        grad = loss.backward()
        
        for layer in tqdm.tqdm(model.layers[::-1]):
            print("grad:",grad)
            grad = layer.backward(grad)

            if isinstance(layer, DahoLayers.Layer):
                layer.w -= layer.w_gradient * self.lr
                layer.b -= layer.b_gradient * self.lr
