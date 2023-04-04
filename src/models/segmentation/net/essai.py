import torch
import numpy as np

a = torch.FloatTensor([220, -120, 6, -1])

print(torch.sigmoid(torch.mean(a)))
print(torch.mean(torch.sigmoid(a)))


