import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# load dataset into pandas dataframe
df = pd.read_csv('data/ufc.csv')

# remove title fights to help corner bias
df = df[df['title_fight'] != 1].copy()

# select relevant columns
cols = [
    # fighter names
    'r_name', 'b_name',
    # striking accuracy
    'r_sig_str_acc', 'b_sig_str_acc',
    # striking defense
    'r_str_def', 'b_str_def',
    # takedown accuracy
    'r_td_acc', 'b_td_acc',
    # takedown defense
    'r_td_def', 'b_td_def',
    # submission attempts
    'r_sub_att', 'b_sub_att',
    # control time
    'r_ctrl', 'b_ctrl',
    # reach
    'r_reach', 'b_reach',
    # height
    'r_height', 'b_height',
    # winner name
    'winner',
    # wins and losses
    'r_wins', 'b_wins', 'r_losses', 'b_losses'
]

#new -> str defense, td defense, height, win% diff

df = df[cols].copy()

# create column 'winner_corner' = 1 if Red won, 0 if Blue won
df['winner_corner'] = np.where(df['winner'] == df['r_name'], 1, 0)


# fill missing numeric values with 0
numeric_cols = [
    'r_sig_str_acc', 'b_sig_str_acc',
    'r_str_def', 'b_str_def',
    'r_td_acc', 'b_td_acc',
    'r_td_def', 'b_td_def',
    'r_sub_att', 'b_sub_att',
    'r_ctrl', 'b_ctrl',
    'r_reach', 'b_reach',
    'r_height', 'b_height',
    'r_wins', 'b_wins', 'r_losses', 'b_losses'
]

df[numeric_cols] = df[numeric_cols].fillna(0)

# calculate differences between red and blue corner for each feature
df['sig_str_diff'] = df['r_sig_str_acc'] - df['b_sig_str_acc']
df['str_def_diff'] = df['r_str_def'] - df['b_str_def']
df['td_diff'] = df['r_td_acc'] - df['b_td_acc']
df['td_def_diff'] = df['r_td_def'] - df['b_td_def']
df['sub_diff'] = df['r_sub_att'] - df['b_sub_att']
df['ctrl_diff'] = df['r_ctrl'] - df['b_ctrl']
df['reach_diff'] = df['r_reach'] - df['b_reach']
df['height_diff'] = df['r_height'] - df['b_height']
df['wins_perc_diff'] = (df['r_wins'] / (df['r_wins'] + df['r_losses'] + 1e-5)) - (df['b_wins'] / (df['b_wins'] + df['b_losses'] + 1e-5))

# features for the model

# model inputs 
X = df[['sig_str_diff', 'str_def_diff', 'td_diff', 'td_def_diff', 'sub_diff', 'ctrl_diff', 'reach_diff', 'height_diff', 'wins_perc_diff']]

# model outputs (1 for red win, 0 for blue win)
y = df['winner_corner']

# split into training and testing sets
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67) 

#scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#make predictions
predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print("Model accuracy:", round(accuracy, 3))
print("Confusion matrix:\n", cm)

#save the model and scaler
joblib.dump(model, 'ufc_model.pkl')
joblib.dump(scaler, 'scaler.pkl')