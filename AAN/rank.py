import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(1131, 200)  # 8 features, 16 neurons in first hidden layer
        self.fc2 = nn.Linear(200, 150) # 16 neurons in second hidden layer
        self.output = nn.Linear(150, 2) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.output(x))
        # print(x)
        return x
    
print("Loading Neural Network...")
# Create an instance of the network
net = SimpleANN()
net.