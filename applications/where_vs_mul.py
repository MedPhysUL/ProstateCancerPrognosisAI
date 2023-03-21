import torch


# ----------- WHERE IS NOT DIFFERENTIABLE :( ----------- #
a = torch.rand(3, 3, requires_grad=True)  # pixel values
b = torch.rand(3, 3, requires_grad=True)  # label map as a probability map
binary_b = torch.round(b)
where_out = torch.where(binary_b == 1, a, 0.0)
where_out = where_out.flatten()
where_out = torch.nn.Linear(in_features=9, out_features=1)(where_out)
where_out.backward()

print(a.grad)
print(b.grad)  # is None :(


# ----------- MUL IS DIFFERENTIABLE :) ----------- #
a = torch.rand(3, 3, requires_grad=True)  # pixel values
b = torch.rand(3, 3, requires_grad=True)  # label map as a probability map
out = a*b

out = out.flatten()
out = torch.nn.Linear(in_features=9, out_features=1)(out)
out.backward()

print(a.grad)
print(b.grad)

# Conclusion : To make everything differentiable, multiply with the probability label map instead of using where with
# the binary label map.
