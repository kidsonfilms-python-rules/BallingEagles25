import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#############################################
# 1. LOAD AND PREPROCESS DATA
#############################################

def load_and_aggregate_data(game_csv_path: str, regions_csv_path: str) -> pd.DataFrame:
    """
    Loads game-level data from game_csv_path and a team-region mapping from regions_csv_path.
    Computes aggregated stats per team and merges the region information.
    Returns a DataFrame with one row per team.
    """
    # Load game data. Adjust sep if necessary (e.g., comma vs. tab).
    df = pd.read_csv(game_csv_path)
    
    # Optional: print the columns to verify the header names
    # print("Columns in game data CSV:", df.columns.tolist())
    
    # Convert game_date if the column exists. If not, skip this step.
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    
    # Create win/loss indicators
    df['win'] = (df['team_score'] > df['opponent_team_score']).astype(int)
    df['loss'] = (df['team_score'] < df['opponent_team_score']).astype(int)
    
    # Create a home win indicator (only count as home win if playing at home)
    df['home_win'] = ((df['home_away'] == "home") & (df['team_score'] > df['opponent_team_score'])).astype(int)
    
    # Compute margin of victory for each game
    df['margin'] = df['team_score'] - df['opponent_team_score']
    
    # Group by team to compute aggregated stats
    agg_dict = {
        'win': 'sum',
        'loss': 'sum',
        'home_win': 'sum',
        'margin': 'mean',             # average margin of victory
        'team_score': 'mean',         # average points scored
        'opponent_team_score': 'mean' # average points allowed
    }
    
    team_stats = df.groupby('team').agg(agg_dict).reset_index()
    team_stats.rename(columns={
        'win': 'wins',
        'loss': 'losses',
        'home_win': 'home_wins',
        'margin': 'avg_margin',
        'team_score': 'avg_points_for',
        'opponent_team_score': 'avg_points_against'
    }, inplace=True)
    
    # Load team-region mapping CSV
    regions_df = pd.read_csv(regions_csv_path)
    # print("Columns in team regions CSV:", regions_df.columns.tolist())
    
    # Merge the region information into our aggregated DataFrame
    team_stats = pd.merge(team_stats, regions_df, on='team', how='left')
    
    # Check for teams without a region
    missing_regions = team_stats[team_stats['region'].isna()]
    if not missing_regions.empty:
        print("Warning: The following teams have no region assigned:")
        print(missing_regions['team'].tolist())
    
    return team_stats


#############################################
# 2. BUILD A DATASET FOR PYTORCH
#############################################

class TeamDataset(torch.utils.data.Dataset):
    """
    A simple PyTorch Dataset for team-level features.
    In this example, we'll use a few aggregated stats to predict a proxy target for team strength.
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        self.df = df
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        self.X = torch.tensor(self.df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(self.df[target_col].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#############################################
# 3. DEFINE A NEURAL NETWORK FOR TEAM STRENGTH
#############################################

class TeamStrengthModel(nn.Module):
    def __init__(self, input_dim):
        super(TeamStrengthModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)  # 8 features, 16 neurons in first hidden layer
        self.fc2 = nn.Linear(200, 150) # 16 neurons in second hidden layer
        self.output = nn.Linear(150, 1) # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.output(x))
        # print(x)
        return x


#############################################
# 4. TRAIN THE MODEL (OPTIONAL STEP)
#############################################

def train_team_strength_model(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Trains a neural network on aggregated team data to predict a "team strength" measure.
    In this example, we use 'wins' as a simple proxy for team strength.
    """
    # Select features (you can expand this list with more relevant stats)
    feature_cols = ['avg_margin', 'avg_points_for', 'avg_points_against']  
    target_col = 'wins'
    
    dataset = TeamDataset(team_stats, feature_cols, target_col)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = TeamStrengthModel(input_dim=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss = {loss.item():.4f}")
    
    # Use the trained model to generate a team strength rating for each team
    X_all = torch.tensor(team_stats[feature_cols].values, dtype=torch.float32)
    with torch.no_grad():
        strength_preds = model(X_all).squeeze().numpy()
    
    team_stats['nn_strength'] = strength_preds
    return team_stats


#############################################
# 5. RANK TEAMS PER REGION
#############################################

def rank_teams(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts teams based on:
      1) Win-loss record (wins descending, losses ascending)
      2) Home-court performance (home_wins descending)
      3) Neural network team strength (nn_strength descending)
    """
    team_stats_sorted = team_stats.sort_values(
        by=['wins', 'losses', 'home_wins', 'nn_strength'],
        ascending=[False, True, False, False]
    ).reset_index(drop=True)
    
    return team_stats_sorted

def get_top_16_per_region(team_stats_sorted: pd.DataFrame):
    """
    Selects the top 16 teams per region from the sorted DataFrame.
    """
    top_16_list = []
    for region in team_stats_sorted['region'].unique():
        region_teams = team_stats_sorted[team_stats_sorted['region'] == region]
        top_16 = region_teams.head(16)
        top_16_list.append(top_16)
    return pd.concat(top_16_list, ignore_index=True)


#############################################
# 6. MAIN SCRIPT EXAMPLE
#############################################

def main():
    game_csv_path = "data/games_2022.csv"      # Path to your game-level CSV file
    regions_csv_path = "data/Team Region Groups.csv"       # Path to your team-region CSV file
    
    # 1. Load and aggregate game data, then merge region information
    team_stats = load_and_aggregate_data(game_csv_path, regions_csv_path)
    
    # 2. (Optional) Train a neural network to compute a "team strength" rating
    team_stats = train_team_strength_model(team_stats)
    
    # 3. Rank teams by wins, losses, home wins, and NN strength
    team_stats_sorted = rank_teams(team_stats)
    
    # 4. Get the top 16 teams per region
    top_16_df = get_top_16_per_region(team_stats_sorted)
    
    # Print the results
    print("=== Top 16 Teams Per Region ===")
    print(top_16_df[['team', 'region', 'wins', 'losses', 'home_wins', 'nn_strength']])
    
    # Optionally, save the results to a CSV file
    top_16_df.to_csv("top_16_teams_per_region.csv", index=False)

if __name__ == "__main__":
    main()
