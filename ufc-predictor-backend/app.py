import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ufc_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
CSV_PATH = os.path.join(BASE_DIR, "data", "ufc.csv")

# Load model, scaler and data
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(CSV_PATH)

# normalize fighter name columns (keep your existing behavior)
name_cols = ['r_name', 'b_name', 'winner']
for col in name_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()


def get_fighter_stats(fighter_name, corner):
    # return fighter stats as a dictionary
    # fighter_name: string of fighter's full name
    # corner: 'r' for red, 'b' for blue
    fighter_row = df[df[f'{corner}_name'] == fighter_name].iloc[0]

    stats = {
        # striking accuracy
        'sig_str': fighter_row[f'{corner}_sig_str_acc'],
        # striking defense
        'str_def': fighter_row[f'{corner}_str_def'],
        # takedown accuracy
        'td': fighter_row[f'{corner}_td_acc'],
        # takedown defense
        'td_def': fighter_row[f'{corner}_td_def'],
        # submission attempts
        'sub': fighter_row[f'{corner}_sub_att'],
        # control time
        'ctrl': fighter_row[f'{corner}_ctrl'],
        # reach
        'reach': fighter_row[f'{corner}_reach'],
        # height
        'height': fighter_row[f'{corner}_height'],
        # wins
        'wins': fighter_row[f'{corner}_wins'],
        # losses
        'losses': fighter_row[f'{corner}_losses']
    }

    # replace NaN with 0 
    for key in stats:
        if pd.isna(stats[key]):
            stats[key] = 0

    return stats

def predict_fight(red_name, blue_name):
    try:
        red_stats = get_fighter_stats(red_name, 'r')
        blue_stats = get_fighter_stats(blue_name, 'b')
    except IndexError:
        return NameError ;

    # calculate differences
    features = np.array([
        red_stats['sig_str'] - blue_stats['sig_str'],
        red_stats['str_def'] - blue_stats['str_def'],
        red_stats['td'] - blue_stats['td'],
        red_stats['td_def'] - blue_stats['td_def'],
        red_stats['sub'] - blue_stats['sub'],
        red_stats['ctrl'] - blue_stats['ctrl'],
        red_stats['reach'] - blue_stats['reach'],
        red_stats['height'] - blue_stats['height'],
        (red_stats['wins'] / (red_stats['wins'] + red_stats['losses'] + 1e-5)) -
        (blue_stats['wins'] / (blue_stats['wins'] + blue_stats['losses'] + 1e-5))
    ]).reshape(1, -1)

    # scale features 
    feature_names = ['sig_str_diff', 'str_def_diff', 'td_diff', 'td_def_diff', 'sub_diff', 'ctrl_diff', 'reach_diff', 'height_diff', 'wins_perc_diff']
    features_df = pd.DataFrame(features, columns=feature_names)
    features_scaled = scaler.transform(features_df)

    # make prediction and probabilities
    prediction = model.predict(features_scaled)[0]
    prob_red = float(model.predict_proba(features_scaled)[0][1])
    prob_blue = float(model.predict_proba(features_scaled)[0][0])

    if prediction == 1:
        confidence = round(prob_red * 100, 2)
        return f"{red_name.title()} will likely win ({confidence}% confidence)"
    else:
        confidence = round(prob_blue * 100, 2)
        return f"{blue_name.title()} will likely win ({confidence}% confidence)"

# Flask wrapper

app = Flask(__name__)
CORS(app) 

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Expects JSON: { "red": "fighter name", "blue": "fighter name" }
    Returns JSON: { "message": "..." }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Send JSON with 'red' and 'blue' fields."}), 400

    red = data.get("red", "").lower().strip()
    blue = data.get("blue", "").lower().strip()

    if not red or not blue:
        return jsonify({"error": "Both 'red' and 'blue' must be provided."}), 400

    result_string = predict_fight(red, blue)

    return jsonify({"message": result_string}), 200

@app.route("/api/fighters", methods=["GET"])
def api_fighters():
    """
    Return a JSON list of fighters (from r_name and b_name) to help frontend autocomplete.
    """
    names = set()
    if 'r_name' in df.columns:
        names.update(df['r_name'].dropna().astype(str).str.strip().tolist())
    if 'b_name' in df.columns:
        names.update(df['b_name'].dropna().astype(str).str.strip().tolist())

    # return as lowercased 
    return jsonify({"fighters": sorted([n.lower() for n in names])})

# run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=5000, debug=False)
