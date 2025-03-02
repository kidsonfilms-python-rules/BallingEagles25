import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from math import exp
import matplotlib.pyplot as plt

#############################################
# 1. LOAD & AGGREGATE TEAM-LEVEL DATA
#############################################

def load_and_aggregate_data(game_csv_path: str, regions_csv_path: str) -> pd.DataFrame:
    """
    Loads game-level data and aggregates per team.
    
    Computes:
      - Basic indicators (wins, losses, margin)
      - An "adjusted_win" (1.0 for away wins, 0.5 for home wins)
      - A composite target:
           composite_strength = (avg_points_for - avg_points_against) 
                               + 10*(win_percentage - 0.5)
         where win_percentage = adjusted_win / (wins+losses)
    
    Then merges the team-region mapping.
    """
    df = pd.read_csv(game_csv_path)
    print("Columns in game data CSV:", df.columns.tolist())
    
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    
    # Compute outcomes and margins
    df['win'] = (df['team_score'] > df['opponent_team_score']).astype(int)
    df['loss'] = (df['team_score'] < df['opponent_team_score']).astype(int)
    df['home_win'] = ((df['home_away'] == "home") & (df['win'] == 1)).astype(int)
    df['margin'] = df['team_score'] - df['opponent_team_score']
    
    # Adjusted win: full credit for away wins, half credit for home wins
    df['adjusted_win'] = df.apply(lambda row: 1.0 if (row['win'] == 1 and row['home_away'] != "home")
                                  else (0.5 if (row['win'] == 1 and row['home_away'] == "home") else 0.0), axis=1)
    
    # Away travel: non-zero only for away games
    df['away_travel'] = df.apply(lambda row: row['travel_dist'] if row['home_away'] != "home" else 0, axis=1)
    
    # Additional numeric columns to aggregate, if available
    extra_cols = [
        'FGA_2', 'FGM_2', 'FGA_3', 'FGM_3', 'FTA', 'FTM',
        'AST', 'BLK', 'STL', 'TOV', 'DREB', 'OREB',
        'F_tech', 'F_personal', 'largest_lead',
        'OT_length_min_tot', 'rest_days', 'attendance',
        'tz_dif_H_E', 'prev_game_dist'
    ]
    
    agg_dict = {
        'win': 'sum',
        'loss': 'sum',
        'home_win': 'sum',
        'adjusted_win': 'sum',
        'margin': 'mean',
        'team_score': 'mean',
        'opponent_team_score': 'mean'
    }
    for col in extra_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'
    
    # Aggregate by team
    team_stats = df.groupby('team').agg(agg_dict).reset_index()
    team_stats.rename(columns={
        'win': 'wins',
        'loss': 'losses',
        'home_win': 'home_wins',
        'margin': 'avg_margin',
        'team_score': 'avg_points_for',
        'opponent_team_score': 'avg_points_against'
    }, inplace=True)
    
    # Compute average away travel per team
    away_travel = (df[df['home_away'] != "home"]
                   .groupby('team')['travel_dist']
                   .mean()
                   .reset_index()
                   .rename(columns={'travel_dist': 'avg_away_travel'}))
    team_stats = pd.merge(team_stats, away_travel, on='team', how='left')
    team_stats['avg_away_travel'] = team_stats['avg_away_travel'].fillna(0)
    
    # Composite target: scoring differential plus a weighted win percentage deviation
    team_stats['games'] = team_stats['wins'] + team_stats['losses']
    team_stats['win_percentage'] = team_stats.apply(lambda row: row['adjusted_win'] / row['games'] if row['games'] > 0 else 0, axis=1)
    team_stats['composite_strength'] = (team_stats['avg_points_for'] - team_stats['avg_points_against']) + \
                                       10 * (team_stats['win_percentage'] - 0.5)
    
    # Merge in team-region mapping
    regions_df = pd.read_csv(regions_csv_path)
    print("Columns in team regions CSV:", regions_df.columns.tolist())
    team_stats = pd.merge(team_stats, regions_df, on='team', how='left')
    missing_regions = team_stats[team_stats['region'].isna()]
    if not missing_regions.empty:
        print("Warning: The following teams have no region assigned:")
        print(missing_regions['team'].tolist())
    
    return team_stats

#############################################
# 2. PAIRWISE RANKING MODEL (FOR RANKING TEAMS)
#############################################

class TeamStrengthModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_team_ranking_model(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Trains a network using pairwise ranking loss.
    For a pair (i, j), if team i's composite_strength > team j's,
    we enforce f(x_i) - f(x_j) >= margin.
    """
    base_features = ['avg_margin', 'avg_points_for', 'avg_points_against', 'avg_away_travel']
    extra_features = [col for col in [
        'FGA_2', 'FGM_2', 'FGA_3', 'FGM_3', 'FTA', 'FTM', 'AST',
        'BLK', 'STL', 'TOV', 'DREB', 'OREB', 'F_tech', 'F_personal',
        'largest_lead', 'OT_length_min_tot', 'rest_days', 'attendance',
        'tz_dif_H_E', 'prev_game_dist'
    ] if col in team_stats.columns]
    feature_cols = base_features + extra_features
    target_col = 'composite_strength'
    
    team_stats[feature_cols] = team_stats[feature_cols].fillna(0)
    X = team_stats[feature_cols].values
    y = team_stats[target_col].values
    
    # Standardize features for stability
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    model = TeamStrengthModel(input_dim=len(feature_cols))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    margin = 1.0  # desired margin for ranking
    
    epochs = 100
    num_pairs = 200  # number of random pairs per epoch
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_samples = X_train_tensor.shape[0]
        for _ in range(num_pairs):
            i, j = np.random.choice(num_samples, 2, replace=False)
            xi = X_train_tensor[i].unsqueeze(0)
            xj = X_train_tensor[j].unsqueeze(0)
            yi_val = y_train_tensor[i].item()
            yj_val = y_train_tensor[j].item()
            if yi_val == yj_val:
                continue
            label = 1 if yi_val > yj_val else -1
            fi = model(xi)
            fj = model(xj)
            loss = torch.relu(margin - label * (fi - fj))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / num_pairs
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Pairwise Loss = {avg_loss:.4f}")
    
    model.eval()
    with torch.no_grad():
        preds_val = model(torch.tensor(X_val, dtype=torch.float32)).squeeze().numpy()
    spearman_corr, _ = spearmanr(y_val, preds_val)
    print(f"Validation Spearman Correlation: {spearman_corr:.4f}")
    
    # Generate predictions for all teams
    with torch.no_grad():
        full_preds = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
    team_stats['nn_strength'] = full_preds
    team_stats.attrs['scaler'] = scaler
    team_stats.attrs['ranking_model'] = model
    return team_stats

#############################################
# 3. WIN PROBABILITY NEURAL NETWORK MODEL (UPDATED)
#############################################

class WinProbabilityNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

def train_win_probability_model(team_stats: pd.DataFrame) -> (WinProbabilityNN, StandardScaler):
    """
    Creates synthetic matchup data from team_stats using the learned nn_strength values.
    For each unique pair of teams, the feature vector is [strength_A, strength_B] and the label is 1 if
    team A's strength > team B's strength, else 0.
    Trains a neural network on these features and validates using log loss, Brier score, and calibration.
    """
    matchups = []
    teams = team_stats['team'].values
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            strength_A = team_stats.loc[team_stats['team'] == teams[i], 'nn_strength'].values[0]
            strength_B = team_stats.loc[team_stats['team'] == teams[j], 'nn_strength'].values[0]
            label = 1 if strength_A > strength_B else 0
            matchups.append([strength_A, strength_B, label])
    
    matchup_df = pd.DataFrame(matchups, columns=['strength_A', 'strength_B', 'label'])
    X = matchup_df[['strength_A', 'strength_B']].values
    y = matchup_df['label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    model = WinProbabilityNN(input_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = loss_fn(val_outputs, y_val_tensor)
                predicted = (val_outputs >= 0.5).float()
                accuracy = (predicted == y_val_tensor).float().mean().item()
            print(f"WinProb Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {accuracy:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_val_tensor).squeeze().numpy()
        from sklearn.metrics import log_loss, brier_score_loss
        final_log_loss = log_loss(y_val, final_outputs)
        final_brier = brier_score_loss(y_val, final_outputs)
    print(f"Final Log Loss: {final_log_loss:.4f}, Final Brier Score: {final_brier:.4f}")
    
    # Plot calibration
    plot_probability_calibration(final_outputs, y_val)
    
    return model, scaler

def plot_probability_calibration(pred_probs, true_labels):
    """
    Plots calibration of predicted win probabilities.
    """
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(pred_probs, bins) - 1
    actual_win_rates = [np.mean(true_labels[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan 
                        for i in range(len(bins) - 1)]
    
    plt.figure(figsize=(6,6))
    plt.plot(bins[:-1], actual_win_rates, marker='o', linestyle='', label='Actual win rate')
    plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
    plt.xlabel('Predicted Win Probability')
    plt.ylabel('Actual Win Rate')
    plt.legend()
    plt.title('Win Probability Calibration')
    plt.show()


def predict_win_probability_nn(team_A: str, team_B: str, team_stats: pd.DataFrame, win_model: WinProbabilityNN, scaler: StandardScaler) -> float:
    """
    Predicts the probability of Team A beating Team B using the win probability neural network.
    """
    try:
        strength_A = team_stats.loc[team_stats['team'] == team_A, 'nn_strength'].values[0]
        strength_B = team_stats.loc[team_stats['team'] == team_B, 'nn_strength'].values[0]
    except IndexError:
        raise ValueError("One or both team names not found in the aggregated data.")
    
    X = np.array([[strength_A, strength_B]])
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    win_prob = win_model(X_tensor).item()
    return win_prob

#############################################
# 4. RANKING & PREDICTING MATCHUP WIN PROBABILITIES
#############################################

def rank_teams(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Ranks teams based on their learned nn_strength (composite strength prediction)."""
    team_stats_sorted = team_stats.sort_values(by=['nn_strength'], ascending=False).reset_index(drop=True)
    return team_stats_sorted

def get_top_64_teams(team_stats_sorted: pd.DataFrame) -> pd.DataFrame:
    """Selects the top 64 teams overall."""
    return team_stats_sorted.head(64)

def get_top_16_per_region(team_stats_sorted: pd.DataFrame) -> pd.DataFrame:
    """Selects the top 16 teams for each region."""
    top_16_list = []
    for region in team_stats_sorted['region'].unique():
        region_teams = team_stats_sorted[team_stats_sorted['region'] == region]
        top_16 = region_teams.head(16)
        top_16_list.append(top_16)
    return pd.concat(top_16_list, ignore_index=True)

def predict_games(prediction_csv_path: str, team_stats: pd.DataFrame, win_model: WinProbabilityNN, scaler: StandardScaler) -> pd.DataFrame:
    """
    Loads the list of games to predict, calculates win probabilities using the win probability NN,
    and saves the results.
    """
    predictions_df = pd.read_csv(prediction_csv_path)
    predictions_df['win_probability_home'] = predictions_df.apply(
        lambda row: predict_win_probability_nn(row['team_home'], row['team_away'], team_stats, win_model, scaler), axis=1
    )
    predictions_df['win_probability_away'] = 1 - predictions_df['win_probability_home']
    predictions_df.to_csv("east_regional_predictions_nn.csv", index=False)
    print("Predictions saved to east_regional_predictions_nn.csv")
    return predictions_df

#############################################
# 5. MAIN SCRIPT
#############################################

def main():
    # File paths for competition data
    game_csv_path = "data/games_2022.csv"
    regions_csv_path = "data/Team Region Groups.csv"
    prediction_csv_path = "data/East Regional Games to predict.csv"
    
    # Step 1: Aggregate team-level statistics
    team_stats = load_and_aggregate_data(game_csv_path, regions_csv_path)
    
    # Step 2: Train the ranking model
    team_stats = train_team_ranking_model(team_stats)
    
    # Step 3: Train the win probability model using synthetic matchup data
    win_model, scaler = train_win_probability_model(team_stats)
    
    # Step 4: Predict games from the provided matchup file using the win probability NN
    predictions_df = predict_games(prediction_csv_path, team_stats, win_model, scaler)
    
    print("\n=== Sample Predictions ===")
    print(predictions_df[['game_id', 'team_home', 'team_away', 'win_probability_home', 'win_probability_away']].head())

if __name__ == "__main__":
    main()
