import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

print("Opening Dataset...")
df = pd.read_csv('data/games_2022.csv')
print(f"Found {len(df)} datapoints!")

print("[PHASE 1] Processing Data...")

# Drop unnecessary column
df.drop("home_away", axis=1, inplace=True)

# Add opponent_team column
df.insert(loc=2, column='opponent_team', value="")

# Convert date format and binary column
df['game_date'] = df['game_date'].str.replace("-", "").astype(int)
df['notD1_incomplete'] = df['notD1_incomplete'].astype(int)

# Fill opponent_team for each game pairing
for i in range(0, len(df), 2):
    df.at[i, "opponent_team"] = df.loc[i+1, 'team']
    df.at[i+1, "opponent_team"] = df.loc[i, 'team']

df.to_csv('data/dump.csv')

print("[PHASE 2] One-Hot Encoding Teams...")
df = pd.get_dummies(df, columns=['team', 'opponent_team'])

print("[PHASE 3] Normalizing Data Before Splitting...")

# Define input features (exclude scores)
input_features = ['rest_days', 'travel_dist', 'home_away_NS'] + [col for col in df.columns if col.startswith(('team_', 'opponent_team_'))]

# Extract inputs and outputs separately
input_df = df[input_features]
output_df = pd.DataFrame({
    "WL": df["team_score"] > df["opponent_team_score"],
    "T": df["team_score"] == df["opponent_team_score"]
})

# Normalize the input data
scaler = MinMaxScaler()
input_df = pd.DataFrame(scaler.fit_transform(input_df), columns=input_df.columns)

print("[PHASE 4] Splitting Train and Test Data...")

# Split by unique game_id to maintain consistency
unique_ids = df['game_id'].unique()
num_samples = int((0.1 * len(df)) / 2)
sampled_ids = np.random.choice(unique_ids, size=num_samples, replace=False)

# Create train and test splits
test_indices = df['game_id'].isin(sampled_ids)
input_test_df = input_df[test_indices]
input_train_df = input_df[~test_indices]
output_test_df = output_df[test_indices]
output_train_df = output_df[~test_indices]

input_train_df.drop("team_score", axis=1, inplace=True)
input_train_df.drop("opponent_team_score", axis=1, inplace=True)
input_test_df.drop("team_score", axis=1, inplace=True)
input_test_df.drop("opponent_team_score", axis=1, inplace=True)

# Fill NaN values
input_train_df.fillna(0, inplace=True)
input_test_df.fillna(0, inplace=True)

print("[PHASE 5] Writing Data Files...")
input_train_df.to_csv('data/processed/train_input_games_2022.csv', index=False)
input_test_df.to_csv('data/processed/test_input_games_2022.csv', index=False)
output_train_df.to_csv('data/processed/train_output_games_2022.csv', index=False)
output_test_df.to_csv('data/processed/test_output_games_2022.csv', index=False)

print("Preprocessing Complete!")
