import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device", device.type)
t2 = torch.randn(1, 2).to(device)
print(t2)


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
