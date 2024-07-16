import os
import torch
from torch import nn
import torch.backends
import torch.backends.mps
from torch.utils.data import DataLoader
from torchvision import transforms

device = (
    "cuda" if torch.cuda.is_available() # check if GPU is available
    else "mps" # use CPU in case GPU is not available
    if torch.backends.mps.is_available() # check if multi-process service is available
    else "cpu" # use CPU in case multi-process service is not available
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

model = NeuralNetwork().to(device)


X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

input_image = torch.rand(3,28,28) # Creates an image with size (3,28,28)
# color code of rgb


flatten = nn.Flatten() # This layer will flatten the image to array 784 values
flat_image = flatten(input_image) # Flattens the image
# print(flat_image.size()) # Prints the size of the flattened image

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)

#print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
#print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)

input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# print(f"Model structure: {model}\n\n")
# 
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")