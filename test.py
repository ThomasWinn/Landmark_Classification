import numpy as np
import torch

x = np.arange(64).reshape(2, 8, 4)
z = torch.from_numpy(x)
y = torch.from_numpy(x)

print(z.shape)

a = torch.flatten(y)
print(a.shape)

fl = torch.nn.Flatten()
b = fl(z)
print(b.shape)