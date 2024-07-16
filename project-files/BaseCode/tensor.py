import torch 
import numpy as np

#data = [[1,2,3],[4,5,6]]
#x_data = torch.tensor(data)

#print(x_data)

#np_array = np.array(data)
#"x_np = torch.from_numpy(np_array)

#print(x_np)

tensor = torch.tensor([[1,2,3],[4,5,6]])

#print(tensor)

#1 = torch.cat([tensor, tensor, tensor], dim=1)
#print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
# y1 = tensor @ tensor.T
# y2 = tensor.matmul(tensor.T)
# 
# y3 = torch.rand_like(y1)
# torch.matmul(tensor, tensor.T, out=y3)
# 
# 
# # This computes the element-wise product. z1, z2, z3 will have the same value
# z1 = tensor * tensor
# z2 = tensor.mul(tensor)
# 
# z3 = torch.rand_like(tensor)
# torch.mul(tensor, tensor, out=z3)

print(tensor)
print(tensor.cpu().numpy())