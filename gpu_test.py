import torch.nn as nn
import torch.optim as optim
import torch

print("torch version: ", torch.__version__)
print("Supported architectures by cuda: ", torch.cuda.get_arch_list())
assert 'sm_86' in torch.cuda.get_arch_list()

EPOCHS = 20
BATCH_SIZE = 4096
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Model(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(in_dims, 128)
        self.layer2 = nn.Linear(128, out_dims)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return x


model = Model(20, 1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
