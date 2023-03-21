import torch


# ----------- ROUND IS DIFFERENTIABLE BUT GIVES ALL 0 :( ----------- #
a = torch.rand(3, 3, requires_grad=True)  # pixel values
b = torch.rand(3, 3, requires_grad=True)  # label map as a probability map
binary_b = torch.round(b)
out = binary_b*a
out = out.flatten()
out = torch.nn.Linear(in_features=9, out_features=1)(out)
out.backward()

print(a.grad)
print(b.grad)  # is all 0


# ----------- USE PROBABILITIES :) ----------- #
a = torch.rand(3, 3, requires_grad=True)  # pixel values
b = torch.rand(3, 3, requires_grad=True)  # label map as a probability map
out = a*b

out = out.flatten()
out = torch.nn.Linear(in_features=9, out_features=1)(out)
out.backward()

print(a.grad)
print(b.grad)

# Conclusion : To make everything differentiable, multiply with the probability label map instead of using the binary
# label map.
