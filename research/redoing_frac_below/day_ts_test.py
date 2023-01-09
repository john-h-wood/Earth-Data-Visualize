import scipy.io as sio
import numpy as np

data = sio.loadmat('/Users/johnwood/Desktop/hello.mat', squeeze_me=True)
days = data['days']

print(np.shape(np.arange(100)))

