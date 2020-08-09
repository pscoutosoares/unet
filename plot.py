import numpy as np
import matplotlib.pyplot as plt 

data = np.load('validation_accuracy_CoopNets-CROPPED-model-4-projs.npy')
plt.plot(data)
plt.show()