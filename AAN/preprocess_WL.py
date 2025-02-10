import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def copy_missing_columns(df_to_update, df_source):
    """
    Copies missing columns from df_source to df_to_update, maintaining the column order of df_source.

    Args:
        df_to_update (pd.DataFrame): The DataFrame to which missing columns will be added.
        df_source (pd.DataFrame): The DataFrame from which missing columns will be copied.

    Returns:
        pd.DataFrame: The updated DataFrame with missing columns added.
    """
    source_cols = df_source.columns.tolist()
    target_cols = df_to_update.columns.tolist()

    missing_cols = [col for col in source_cols if col not in target_cols]

    for col in missing_cols:
      df_to_update[col] = pd.Series(dtype=df_source[col].dtype)

    # Reorder columns to match df_source
    all_cols = target_cols + missing_cols
    df_updated = df_to_update[source_cols]
      
    return df_updated

print("Opening Dataset...")
df = pd.read_csv('data/games_2022.csv')
print("Found " + str(len(df)) + " datapoints!")

print("[PHASE 1] Processing Data...")
df.drop("home_away", axis=1, inplace=True)
df.insert(loc=2, column='opponent_team', value="")
df['game_date'] = df['game_date'].str.replace("-","").astype(int)
df['notD1_incomplete'] = df['notD1_incomplete'].astype(int)

for i in range(0, len(df), 2):
    # Get the current two rows
    two_rows = df[i:i+2]
    
    # Process the two rows
    df.at[i, "opponent_team"] = df.loc[i+1, 'team']
    df.at[i+1, "opponent_team"] = df.loc[i, 'team']
    # Add your logic here to work with 'two_rows'

df.to_csv('data/dump.csv')

print("[PHASE 2] Splitting Train and Test data...")

unique_ids = df['game_id'].unique()
num_samples = int((0.1*len(df))/2)
sampled_ids = np.random.choice(unique_ids, size=num_samples, replace=False)

# Randomly sample 10% of the DataFrame
# test_df = df.sample(frac=0.1, random_state=69)
test_df = df[df['game_id'].isin(sampled_ids)]

# Get the indices of the sampled rows
test_indices = test_df.index

# Remove the sampled rows from the original DataFrame
train_df = df.drop(test_indices)

print("[PHASE 3] Normalizing Data...")

input_test_df = pd.DataFrame({'rest_days': test_df['rest_days'], 'travel_dist': test_df['travel_dist'], "home_away_NS": test_df["home_away_NS"]})
input_train_df = pd.DataFrame({'rest_days': train_df['rest_days'], 'travel_dist': train_df['travel_dist'], "home_away_NS": train_df["home_away_NS"]})

scaler = MinMaxScaler()

input_test_df_norm = pd.DataFrame(scaler.fit_transform(input_test_df), columns=input_test_df.columns)
input_train_df_norm = pd.DataFrame(scaler.fit_transform(input_train_df), columns=input_train_df.columns)
test_names_df = pd.DataFrame({'team': test_df["team"].tolist(), 'opponent_team': test_df["opponent_team"].tolist()})
train_names_df = pd.DataFrame({'team': train_df["team"].tolist(), 'opponent_team': train_df["opponent_team"].tolist()})

input_test_df_norm.insert(1, 'team', test_names_df['team'])
input_train_df_norm.insert(1, 'team', train_names_df['team'])
input_test_df_norm.insert(2, 'opponent_team', test_names_df['opponent_team'])
input_train_df_norm.insert(2, 'opponent_team', train_names_df['opponent_team'])

input_test_df_norm = pd.get_dummies(input_test_df_norm, columns=['team', 'opponent_team'])
input_train_df_norm = pd.get_dummies(input_train_df_norm, columns=['team', 'opponent_team'])

input_test_df_norm = copy_missing_columns(input_test_df_norm.copy(), input_train_df_norm)
input_train_df_norm = copy_missing_columns(input_train_df_norm.copy(), input_test_df_norm)

output_test_df = pd.DataFrame({"WL": test_df["team_score"] >= test_df["opponent_team_score"]})
output_train_df = pd.DataFrame({"WL": train_df["team_score"] >= train_df["opponent_team_score"]})

input_test_df_norm.fillna(0, inplace=True)
input_train_df_norm.fillna(0, inplace=True)

# output_train_df = train_df.drop('game_date', 'rest_days', 'travel_distance', 'home_away_NS', 'team', 'opponent_team', 'notD1_incomplete', 'tz_dif_H_E', 'prev_game_dist')

print("[PHASE 4] Writing Data Files...")
print(input_test_df_norm)
print(input_train_df_norm)
input_test_df_norm.to_csv('data/processed/test_input_games_2022.csv', index=False)
input_train_df_norm.to_csv('data/processed/train_input_games_2022.csv', index=False)
output_test_df.to_csv('data/processed/test_output_games_2022.csv', index=False)
output_train_df.to_csv('data/processed/train_output_games_2022.csv', index=False)