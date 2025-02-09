import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(8, 16)  # 8 features, 16 neurons in first hidden layer
        self.fc2 = nn.Linear(16, 16) # 16 neurons in second hidden layer
        self.output = nn.Linear(16, 1) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

train_df_X = pd.read_csv('data/processed/train_input_games_2022.csv')
train_df_Y = pd.read_csv('data/processed/train_output_games_2022.csv')
test_df_X = pd.read_csv('data/processed/test_input_games_2022.csv')
test_df_Y = pd.read_csv('data/processed/test_output_games_2022.csv')

# Example usage
input_size = train_df_X.shape[1]  # For example, if input is flattened MNIST image
hidden_size = 500
num_classes = 10
learning_rate = 0.001
num_epochs = 5

# Create an instance of the network
net = SimpleANN(input_size, hidden_size, num_classes)

criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    # Convert arrays to tensors
    inputs = torch.tensor(train_df_X, dtype=torch.float32)
    labels = torch.tensor(train_df_Y, dtype=torch.float32)

    # Forward pass
    outputs = net(inputs)
    loss = criterion(outputs, labels.unsqueeze(1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    y_predicted = net(torch.tensor(test_df_X, dtype=torch.float32))
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(torch.tensor(test_df_Y).unsqueeze(1)).sum() / float(test_df_Y.shape[0])
    print(f'Accuracy: {acc:.4f}')
