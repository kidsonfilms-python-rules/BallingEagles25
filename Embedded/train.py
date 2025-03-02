import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from scipy.stats import spearmanr
from math import exp
import matplotlib.pyplot as plt
from alive_progress import alive_bar

# Toggle advanced logging (set to False to disable advanced logging)
ADVANCED_LOGGING = True

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

#############################################
# 1. LOAD & AGGREGATE TEAM-LEVEL DATA
#############################################

def load_and_aggregate_data(game_csv_path: str, regions_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(game_csv_path)
    print("Columns in game data CSV:", df.columns.tolist())
    
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

    df['win'] = (df['team_score'] > df['opponent_team_score']).astype(int)
    df['loss'] = (df['team_score'] < df['opponent_team_score']).astype(int)
    df['home_win'] = ((df['home_away'] == "home") & (df['win'] == 1)).astype(int)
    df['margin'] = df['team_score'] - df['opponent_team_score']
    
    # Adjusted win: 1.0 for away wins, 0.5 for home wins
    df['adjusted_win'] = df.apply(
        lambda row: 1.0 if (row['win'] == 1 and row['home_away'] != "home")
        else (0.5 if (row['win'] == 1 and row['home_away'] == "home") else 0.0),
        axis=1
    )
    
    df['away_travel'] = df.apply(lambda row: row['travel_dist'] if row['home_away'] != "home" else 0, axis=1)

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
    
    team_stats = df.groupby('team').agg(agg_dict).reset_index()
    team_stats.rename(columns={
        'win': 'wins',
        'loss': 'losses',
        'home_win': 'home_wins',
        'margin': 'avg_margin',
        'team_score': 'avg_points_for',
        'opponent_team_score': 'avg_points_against'
    }, inplace=True)
    
    away_travel = (
        df[df['home_away'] != "home"]
        .groupby('team')['travel_dist']
        .mean()
        .reset_index()
        .rename(columns={'travel_dist': 'avg_away_travel'})
    )
    team_stats = pd.merge(team_stats, away_travel, on='team', how='left')
    team_stats['avg_away_travel'] = team_stats['avg_away_travel'].fillna(0)
    
    team_stats['games'] = team_stats['wins'] + team_stats['losses']
    team_stats['win_percentage'] = team_stats.apply(
        lambda row: row['adjusted_win'] / row['games'] if row['games'] > 0 else 0, axis=1
    )
    team_stats['composite_strength'] = (
        (team_stats['avg_points_for'] - team_stats['avg_points_against'])
        + 10 * (team_stats['win_percentage'] - 0.5)
    )
    
    regions_df = pd.read_csv(regions_csv_path)
    print("Columns in team regions CSV:", regions_df.columns.tolist())
    team_stats = pd.merge(team_stats, regions_df, on='team', how='left')
    missing_regions = team_stats[team_stats['region'].isna()]
    if not missing_regions.empty:
        print("Warning: The following teams have no region assigned:")
        print(missing_regions['team'].tolist())
    
    return team_stats

#############################################
# 2. PAIRWISE RANKING MODEL
#############################################

class TeamStrengthModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_team_ranking_model(team_stats: pd.DataFrame) -> pd.DataFrame:
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
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    model = TeamStrengthModel(input_dim=len(feature_cols))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7)
    
    margin = 1.0
    epochs = 120
    num_pairs = 400
    
    with alive_bar(epochs, title="Training Ranking Model") as bar:
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
            
            scheduler.step()
            avg_loss = epoch_loss / num_pairs
            if (epoch+1) % 10 == 0:
                print(f"on {epoch}: Epoch {epoch+1}/{epochs}, Average Pairwise Loss = {avg_loss:.4f}")
            bar()
    
    model.eval()
    with torch.no_grad():
        preds_val = model(torch.tensor(X_val, dtype=torch.float32)).squeeze().numpy()
    spearman_corr, _ = spearmanr(y_val, preds_val)
    print(f"Validation Spearman Correlation: {spearman_corr:.4f}")
    
    with torch.no_grad():
        full_preds = model(torch.tensor(X, dtype=torch.float32)).squeeze().numpy()
    team_stats['nn_strength'] = full_preds
    team_stats.attrs['scaler'] = scaler
    team_stats.attrs['ranking_model'] = model
    return team_stats

#############################################
# 3. WIN PROBABILITY NEURAL NETWORK MODEL
#############################################

class WinProbabilityNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

def train_win_probability_model(team_stats: pd.DataFrame) -> (WinProbabilityNN, StandardScaler):
    """
    Creates synthetic matchup data from team_stats using the learned nn_strength values,
    trains a win probability neural network, and logs advanced calibration info.
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

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    model = WinProbabilityNN(input_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7)
    loss_fn = nn.BCELoss()

    epochs = 200

    with alive_bar(epochs, title="Training Win Probability Model") as bar:
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = loss_fn(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (epoch+1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor).squeeze().numpy()
                    # Clamp predictions to avoid log(0)
                    val_outputs = np.clip(val_outputs, 1e-6, 1-1e-6)
                    predicted = (val_outputs >= 0.5).astype(float)
                    accuracy = (predicted == y_val_tensor.numpy().squeeze()).mean()
                    
                    y_val_np = y_val_tensor.numpy().squeeze()
                    current_log_loss = log_loss(y_val_np, val_outputs)
                print(f"on {epoch}: WinProb Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val LogLoss: {current_log_loss:.4f}, Val Acc: {accuracy:.4f}")
            bar()

    model.eval()
    with torch.no_grad():
        final_outputs = model(X_val_tensor).squeeze().numpy()
        final_outputs = np.clip(final_outputs, 1e-6, 1-1e-6)
        final_log_loss = log_loss(y_val, final_outputs)
        final_brier = brier_score_loss(y_val, final_outputs)
    print(f"Final Log Loss: {final_log_loss:.4f}, Final Brier Score: {final_brier:.4f}")

    # Advanced calibration logging using Platt scaling if enabled
    if ADVANCED_LOGGING:
        platt = LogisticRegression()
        final_outputs_2d = final_outputs.reshape(-1, 1)
        platt.fit(final_outputs_2d, y_val)
        platt_preds = platt.predict_proba(final_outputs_2d)[:, 1]
        platt_logloss = log_loss(y_val, platt_preds)
        platt_brier = brier_score_loss(y_val, platt_preds)
        print(f"Platt Scaling => Log Loss: {platt_logloss:.4f}, Brier: {platt_brier:.4f}")
        
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(platt_preds, bins) - 1
        print("\nCalibration Bins (bin_index, bin_start, bin_end, count, avg_pred_prob, actual_win_rate):")
        for i in range(len(bins) - 1):
            mask = (bin_indices == i)
            count = np.sum(mask)
            if count > 0:
                avg_pred = np.mean(platt_preds[mask])
                actual = np.mean(y_val[mask])
                print((i, bins[i], bins[i+1], count, f"{avg_pred:.5f}", f"{actual:.5f}"))
            else:
                print((i, bins[i], bins[i+1], 0, "nan", "nan"))

    return model, scaler

def predict_win_probability_nn(team_A: str, team_B: str, team_stats: pd.DataFrame,
                               win_model: WinProbabilityNN, scaler: StandardScaler) -> float:
    try:
        strength_A = team_stats.loc[team_stats['team'] == team_A, 'nn_strength'].values[0]
        strength_B = team_stats.loc[team_stats['team'] == team_B, 'nn_strength'].values[0]
    except IndexError:
        raise ValueError("One or both team names not found in the aggregated data.")
    
    X = np.array([[strength_A, strength_B]])
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    prob = win_model(X_tensor).item()
    prob = max(min(prob, 1-1e-6), 1e-6)
    return prob

#############################################
# 4. RANKING & PREDICTING MATCHUP WIN PROBABILITIES
#############################################

def rank_teams(team_stats: pd.DataFrame) -> pd.DataFrame:
    team_stats_sorted = team_stats.sort_values(by=['nn_strength'], ascending=False).reset_index(drop=True)
    return team_stats_sorted

def get_top_64_teams(team_stats_sorted: pd.DataFrame) -> pd.DataFrame:
    return team_stats_sorted.head(64)

def get_top_16_per_region(team_stats_sorted: pd.DataFrame) -> pd.DataFrame:
    top_16_list = []
    for region in team_stats_sorted['region'].unique():
        region_teams = team_stats_sorted[team_stats_sorted['region'] == region]
        top_16 = region_teams.head(16)
        top_16_list.append(top_16)
    return pd.concat(top_16_list, ignore_index=True)

def predict_games(prediction_csv_path: str, team_stats: pd.DataFrame,
                  win_model: WinProbabilityNN, scaler: StandardScaler) -> pd.DataFrame:
    predictions_df = pd.read_csv(prediction_csv_path)
    predictions_df['win_probability_home'] = predictions_df.apply(
        lambda row: predict_win_probability_nn(row['team_home'], row['team_away'],
                                               team_stats, win_model, scaler),
        axis=1
    )
    predictions_df['win_probability_away'] = 1 - predictions_df['win_probability_home']
    
    output_path = "output/east_regional_predictions_nn.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return predictions_df

#############################################
# 5. MAIN SCRIPT
#############################################

def main():
    game_csv_path = "data/games_2022.csv"
    regions_csv_path = "data/Team Region Groups.csv"
    prediction_csv_path = "data/East Regional Games to predict.csv"
    
    # 1. Load & aggregate team-level data
    team_stats = load_and_aggregate_data(game_csv_path, regions_csv_path)
    
    # 2. Train the ranking model
    team_stats = train_team_ranking_model(team_stats)
    
    # 3. Train the win probability model
    win_model, scaler = train_win_probability_model(team_stats)
    
    # 4. Predict games and output predictions to CSV in output folder
    predictions_df = predict_games(prediction_csv_path, team_stats, win_model, scaler)
    
    # Additionally, output team rankings
    team_stats_sorted = rank_teams(team_stats)
    rankings_output = "output/team_rankings.csv"
    team_stats_sorted.to_csv(rankings_output, index=False)
    print(f"Team rankings saved to {rankings_output}")
    
    # Also output top 64 and top 16 per region if needed
    top64_output = "output/top_64_teams.csv"
    get_top_64_teams(team_stats_sorted).to_csv(top64_output, index=False)
    print(f"Top 64 teams saved to {top64_output}")
    
    top16_output = "output/top_16_per_region.csv"
    get_top_16_per_region(team_stats_sorted).to_csv(top16_output, index=False)
    print(f"Top 16 teams per region saved to {top16_output}")
    
    print("\n=== Sample Predictions ===")
    print(predictions_df[['game_id', 'team_home', 'team_away', 
                          'win_probability_home', 'win_probability_away']].head())

if __name__ == "__main__":
    main()
