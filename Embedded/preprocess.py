import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def copy_missing_columns(df_to_update, df_source):
    """
    Copies missing columns from df_source to df_to_update, maintaining the column order of df_source.
    """
    source_cols = df_source.columns.tolist()
    target_cols = df_to_update.columns.tolist()

    missing_cols = [col for col in source_cols if col not in target_cols]

    for col in missing_cols:
        df_to_update[col] = pd.Series(dtype=df_source[col].dtype)

    # Reorder columns to match df_source
    df_updated = df_to_update[source_cols]

    return df_updated

print("Opening Dataset...")
df = pd.read_csv('data/games_2022.csv')
print("Found " + str(len(df)) + " datapoints!")

print("[PHASE 1] Processing Data...")
# Drop unnecessary columns
df.drop("home_away", axis=1, inplace=True)

# Insert opponent_team column
df.insert(loc=2, column='opponent_team', value="")

# Convert game_date to integer format
df['game_date'] = df['game_date'].str.replace("-", "").astype(int)

# Convert notD1_incomplete to integer
df['notD1_incomplete'] = df['notD1_incomplete'].astype(int)

# Fill in opponent_team column
for i in range(0, len(df), 2):
    df.at[i, "opponent_team"] = df.loc[i + 1, 'team']
    df.at[i + 1, "opponent_team"] = df.loc[i, 'team']

print("[PHASE 2] Encoding Team Names...")
# Encode team names as numerical IDs (zero-indexed)
label_encoder = LabelEncoder()
df['team_id'] = label_encoder.fit_transform(df['team'])
df['opponent_team_id'] = label_encoder.transform(df['opponent_team'])

print("[PHASE 3] Splitting Train and Test data...")
# Split data into train and test sets
unique_ids = df['game_id'].unique()
num_samples = int((0.1 * len(df)) / 2)
sampled_ids = np.random.choice(unique_ids, size=num_samples, replace=False)

# Create test DataFrame
test_df = df[df['game_id'].isin(sampled_ids)]

# Create train DataFrame
train_df = df.drop(test_df.index)

print("[PHASE 4] Normalizing Data...")
# Normalize numerical features
scaler = MinMaxScaler()

# Normalize input features
input_test_df = pd.DataFrame({
    'rest_days': test_df['rest_days'].tolist(),
    'travel_dist': test_df['travel_dist'].tolist(),
    'home_away_NS': test_df['home_away_NS'].tolist()
})
input_train_df = pd.DataFrame({
    'rest_days': train_df['rest_days'].tolist(),
    'travel_dist': train_df['travel_dist'].tolist(),
    'home_away_NS': train_df['home_away_NS'].tolist()
})

input_test_df_norm = pd.DataFrame(scaler.fit_transform(input_test_df), columns=input_test_df.columns)
input_train_df_norm = pd.DataFrame(scaler.fit_transform(input_train_df), columns=input_train_df.columns)

# Add team and opponent_team columns
input_test_df_norm.insert(0, 'team1_id', test_df['team_id'].tolist())
input_test_df_norm.insert(1, 'team2_id', test_df['opponent_team_id'].tolist())
input_train_df_norm.insert(0, 'team1_id', train_df['team_id'].tolist())
input_train_df_norm.insert(1, 'team2_id', train_df['opponent_team_id'].tolist())

# Create output DataFrames
output_test_df = pd.DataFrame({
    'WL': (test_df['team_score'] > test_df['opponent_team_score']).astype(int),
    # 'T': (test_df['team_score'] == test_df['opponent_team_score']).astype(int)
})
output_train_df = pd.DataFrame({
    'WL': (train_df['team_score'] > train_df['opponent_team_score']).astype(int),
    # 'T': (train_df['team_score'] == train_df['opponent_team_score']).astype(int)
})

input_test_df_norm.fillna(0, inplace=True)
input_train_df_norm.fillna(0, inplace=True)

print("[PHASE 5] Writing Data Files...")
# Save processed data to CSV files
input_test_df_norm.to_csv('data/processed/test_input_games_2022.csv', index=False)
input_train_df_norm.to_csv('data/processed/train_input_games_2022.csv', index=False)
output_test_df.to_csv('data/processed/test_output_games_2022.csv', index=False)
output_train_df.to_csv('data/processed/train_output_games_2022.csv', index=False)

# Save the label encoder for inference
import pickle
with open('data/processed/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Data preprocessing completed and files saved!")