import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

print("Opening Dataset...")
df = pd.read_csv('data/games_2022.csv')

print("[PHASE 1] Processing Data...")
onehot_encoder = OneHotEncoder(sparse=False)
encoded_team = onehot_encoder.fit_transform(df['team'].to_numpy().reshape(-1,1))
df.drop('team', axis=1, inplace=True)
df.insert(1, 'team', encoded_team.tolist())
df.drop("game_id", axis=1, inplace=True)
df.drop("home_away", axis=1, inplace=True)
df['game_date'] = df['game_date'].str.replace("-","").astype(int)
df['notD1_incomplete'] = df['notD1_incomplete'].astype(int)
df.to_csv('data/dump.csv')

print("[PHASE 2] Splitting Train and Test data...")
# Randomly sample 10% of the DataFrame
test_df = df.sample(frac=0.1, random_state=69)

# Get the indices of the sampled rows
test_indices = test_df.index

# Remove the sampled rows from the original DataFrame
train_df = df.drop(test_indices)

print("[PHASE 3] Normalizing Data...")

input_test_df = pd.DataFrame({'game_date': test_df['game_date'],'rest_days': test_df['rest_days'], 'travel_dist': test_df['travel_dist'], "home_away_NS": test_df["home_away_NS"]})
input_train_df = pd.DataFrame({'game_date': train_df['game_date'],'rest_days': train_df['rest_days'], 'travel_dist': train_df['travel_dist'], "home_away_NS": train_df["home_away_NS"]})

scaler = MinMaxScaler()

input_test_df_norm = pd.DataFrame(scaler.fit_transform(input_test_df), columns=input_test_df.columns)
input_train_df_norm = pd.DataFrame(scaler.fit_transform(input_train_df), columns=input_train_df.columns)

input_test_df_norm.insert(1, 'team', test_df['team'])
input_train_df_norm.insert(1, 'team', train_df['team'])

print("[PHASE 4] Writing Data Files...")
input_test_df_norm.to_csv('data/processed/test_input_games_2022.csv', index=False)
input_train_df_norm.to_csv('data/processed/train_input_games_2022.csv', index=False)