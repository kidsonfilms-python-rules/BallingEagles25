import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from alive_progress import alive_bar

# Define the optimized model
class RecommenderModel(nn.Module):
    def __init__(self, num_teams, embedding_dim, num_features):
        super(RecommenderModel, self).__init__()
        self.team_embeddings = nn.Embedding(num_teams, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2 + num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.4)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)

    def forward(self, team1_ids, team2_ids, features):
        team1_embedding = self.team_embeddings(team1_ids)
        team2_embedding = self.team_embeddings(team2_ids)
        combined_embedding = torch.cat([team1_embedding, team2_embedding, features], dim=1)
        hidden = F.relu(self.batch_norm1(self.fc1(combined_embedding)))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.batch_norm2(self.fc2(hidden)))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc3(hidden))
        output = self.output(hidden)
        return output

# Load processed datasets
print("Loading processed datasets...")
train_df_X = pd.read_csv('data/processed/train_input_games_2022.csv')
train_df_Y = pd.read_csv('data/processed/train_output_games_2022.csv')
test_df_X = pd.read_csv('data/processed/test_input_games_2022.csv')
test_df_Y = pd.read_csv('data/processed/test_output_games_2022.csv')

# Extract additional features
feature_columns = ['rest_days', 'travel_dist', 'home_away_NS']

# Create a mapping for team IDs
unique_teams = pd.concat([train_df_X['team1_id'], train_df_X['team2_id'], test_df_X['team1_id'], test_df_X['team2_id']]).unique()
team_id_mapping = {team_id: idx for idx, team_id in enumerate(unique_teams)}
train_df_X['team1_id'] = train_df_X['team1_id'].map(team_id_mapping)
train_df_X['team2_id'] = train_df_X['team2_id'].map(team_id_mapping)
test_df_X['team1_id'] = test_df_X['team1_id'].map(team_id_mapping)
test_df_X['team2_id'] = test_df_X['team2_id'].map(team_id_mapping)

# Convert data to tensors
train_team1 = torch.tensor(train_df_X['team1_id'].values, dtype=torch.long)
train_team2 = torch.tensor(train_df_X['team2_id'].values, dtype=torch.long)
train_features = torch.tensor(train_df_X[feature_columns].values, dtype=torch.float32)
train_labels = torch.tensor(train_df_Y['WL'].values, dtype=torch.long)

test_team1 = torch.tensor(test_df_X['team1_id'].values, dtype=torch.long)
test_team2 = torch.tensor(test_df_X['team2_id'].values, dtype=torch.long)
test_features = torch.tensor(test_df_X[feature_columns].values, dtype=torch.float32)
test_labels = torch.tensor(test_df_Y['WL'].values, dtype=torch.long)

# Update num_teams
num_teams = len(unique_teams)
num_features = len(feature_columns)
print(f"Number of unique teams: {num_teams}")

# Validate team IDs
assert train_team1.max() < num_teams, "Invalid team1_id in training set"
assert train_team2.max() < num_teams, "Invalid team2_id in training set"
assert test_team1.max() < num_teams, "Invalid team1_id in test set"
assert test_team2.max() < num_teams, "Invalid team2_id in test set"

# Create datasets and dataloaders
train_dataset = TensorDataset(train_team1, train_team2, train_features, train_labels)
test_dataset = TensorDataset(test_team1, test_team2, test_features, test_labels)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Hyperparameters
embedding_dim = 128
learning_rate = 0.0003
num_epochs = 50

# Initialize model, loss function, and optimizer
model = RecommenderModel(num_teams, embedding_dim, num_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training loop
print("Starting training...")
with alive_bar(num_epochs) as bar:
    for epoch in range(num_epochs):
        model.train()
        for batch_team1, batch_team2, batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_team1, batch_team2, batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        bar.title(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        bar()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(test_team1, test_team2, test_features)
    _, predicted_labels = torch.max(test_outputs, 1)
    correct = (predicted_labels == test_labels).sum().item()
    total = test_labels.size(0)
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

# Save the model
torch.save(model.state_dict(), 'Embedded/model/recommender_model.pt')
print("Model saved to 'Embedded/model/recommender_model.pt'")