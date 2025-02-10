import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from alive_progress import alive_bar
from datetime import datetime

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(1107, 200)  # 8 features, 16 neurons in first hidden layer
        self.fc2 = nn.Linear(200, 100) # 16 neurons in second hidden layer
        self.output = nn.Linear(100, 1) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.output(x))
        # print(x)
        return x

print("Loading processed datasets...")
train_df_X = pd.read_csv('data/processed/train_input_games_2022.csv').values.astype('float32')
train_df_Y = pd.read_csv('data/processed/train_output_games_2022.csv').values.astype('float32')
test_df_X = pd.read_csv('data/processed/test_input_games_2022.csv').values.astype('float32')
test_df_Y = pd.read_csv('data/processed/test_output_games_2022.csv').values.astype('float32')
print("Loaded " + str(len(train_df_X)) + " training datapoints and " + str(len(test_df_X)) + " testing datapoints! (" + str((len(test_df_X)/(len(test_df_X) + len(train_df_X)))*100) + "%)")

# Example usage
input_size = train_df_X.shape[1]  # For example, if input is flattened MNIST image
hidden_size = 500
num_classes = 10
learning_rate = 0.001
num_epochs = 5

print("Creating Neural Network...")
# Create an instance of the network
net = SimpleANN()

if torch.cuda.is_available():
    print('Accelerating using CUDA!')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

net.to(device)

criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

epochs = 600
print("Starting training with " + str(epochs) + " epochs")
with alive_bar(epochs) as bar:
    for epoch in range(epochs):
        # Convert arrays to tensors
        inputs = torch.tensor(train_df_X, dtype=torch.float)
        labels = torch.tensor(train_df_Y, dtype=torch.float)

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.title(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        bar()

with torch.no_grad():
    y_predicted = net(torch.tensor(test_df_X, dtype=torch.float32))
    y_predicted_cls = y_predicted.round()

    # Ensure test_df_Y is a tensor of the correct dtype
    test_labels = torch.tensor(test_df_Y, dtype=torch.float32)

    # Calculate accuracy correctly
    acc = (y_predicted_cls == test_labels).float().mean()
    print(f'Accuracy: {acc:.4f}')

print('Exporting Trained Model...')
torch.save(net.state_dict(), f'model/trained-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}-2022-games.pt')