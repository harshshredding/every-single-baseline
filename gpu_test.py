import torch
import torch.nn as nn
import subprocess

# Check Cuda Version
cuda_version = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE).stdout.decode('utf-8')
assert 'cuda_11.6' in cuda_version, "Doesn't use CUDA 11.6"

# Try putting a tensor on CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device", device.type)
t2 = torch.randn(1, 2).to(device)


# Try putting a model on GPU
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 2)

    def forward(self, x):
        x = self.l1(x)
        return x


model = M()  # not on cuda
model.to(device)  # is on cuda (all parameters)
assert next(model.parameters()).is_cuda, "model was not put on GPU :("
