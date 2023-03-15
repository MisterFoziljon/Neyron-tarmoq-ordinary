import numpy as np
class MSE:
    def __call__(self, y_pred, y):
        self.error = y - y_pred        
        return np.mean(self.error ** 2)

    def backward(self):
        return 2 * np.mean(self.error)
