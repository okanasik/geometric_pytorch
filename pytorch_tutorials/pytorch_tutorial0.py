from __future__ import print_function
import torch
import numpy as np

x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)
print(x.size())

y = torch.rand(5, 3)
print(x+y)
result = torch.empty(5,3)
torch.add(x,y, out=result)
print(result)

y.add_(x)
print(y)
print(x[:,1])

x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
print(x)
print(y)
print(z)

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

print(torch.cuda.is_available())

x = torch.zeros(10, 5, dtype=torch.float32)
x[0][0] = 10
x[0][4] = 10
x[1][4] = 10
y = x.clone()
y[0][0] = 3
print(x)
print(y)

z = torch.zeros(10000)
print(z)



