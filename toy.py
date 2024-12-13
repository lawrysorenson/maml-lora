import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.autograd as auto
import copy

test = nn.Parameter(torch.tensor([-0.1, 0.2, 0.3]))
loss = (test * test).sum()

sub = auto.grad(loss, test, create_graph=True, retain_graph=True)[0]

test2 = copy.deepcopy(test)
test2.requires_grad = False
test2.grad = sub

optim = SGD([test2], lr=1e-2, differentiable=True)
optim.step()

other = torch.tensor([5., 7., 9.])

# test2.backward(gradient=other)

(other * test2).sum().backward()

print(test2)
print(test2.grad)
print(test)
print(test.grad)