import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model/ufc_xgb_model.pkl")
ELO_PATH     = os.path.join(BASE_DIR, "model/ufc_fighter_elo.pkl")
SCALERS_PATH = os.path.join(BASE_DIR, "model/ufc_feature_scalers.pkl")
CSV_PATH     = os.path.join(BASE_DIR, "data/UFC.csv")

# Load model, elo, and the training-derived scalers/defaults needed to
# reconstruct the composite offense/defense scores at inference time.
model       = joblib.load(MODEL_PATH)
fighter_elo = joblib.load(ELO_PATH)
scaler_data = joblib.load(SCALERS_PATH)
SCALERS     = scaler_data["scalers"]
DEFAULTS    = scaler_data["defaults"]
PRIME_AGE   = DEFAULTS["prime_age"]

df = pd.read_csv(CSV_PATH).copy()

# normalize fighter name columns
for col in ['r_name', 'b_name', 'winner']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

df["date"]    = pd.to_datetime(df["date"])
df["r_dob_x"] = pd.to_datetime(df["r_dob_x"], errors="coerce", format="mixed")
df["b_dob_y"] = pd.to_datetime(df["b_dob_y"], errors="coerce", format="mixed")

df["r_age"] = (df["date"] - df["r_dob_x"]).dt.days / 365.25
df["b_age"] = (df["date"] - df["b_dob_y"]).dt.days / 365.25


def zscore(value, key):
    """Applies the same train-only mean/std used when the model was fit."""
    s = SCALERS[key]
    return (value - s["mean"]) / s["std"]


def get_fighter_stats(fighter_name):
    # search both corners in the historical data — corner assignment in the raw
    # dataset is per-fight, not per-fighter, so a fighter may only ever appear
    # as one corner historically even though they can be predicted as either.
    matches = df[(df['r_name'] == fighter_name) | (df['b_name'] == fighter_name)]
    if matches.empty:
        raise IndexError

    row = matches.sort_values('date').iloc[-1]

    # figure out which side this fighter was on in their most recent fight
    found_corner = 'r' if row['r_name'] == fighter_name else 'b'

    stats = {
        # striking
        'splm':    row.get(f'{found_corner}_splm_x',        0),
        'sapm':    row.get(f'{found_corner}_sapm_x',        0),
        'str_acc': row.get(f'{found_corner}_str_acc_x',     0),
        'str_def': row.get(f'{found_corner}_str_def_x',     0),
        # grappling
        'td_avg':  row.get(f'{found_corner}_td_avg_x',      0),
        'td_acc':  row.get(f'{found_corner}_td_avg_acc_x',  0),
        'td_def':  row.get(f'{found_corner}_td_def_x',      0),
        'sub_avg': row.get(f'{found_corner}_sub_avg_x',     0),
        # physical — NOTE: weight only ever exists as r_weight_x in the
        # source data, so always pull that column regardless of found_corner
        'reach':   row.get(f'{found_corner}_reach',         0),
        'height':  row.get(f'{found_corner}_height',        0),
        'weight':  row.get('r_weight_x',                    0),
        'age':     row.get(f'{found_corner}_age',           0),
        # record
        'wins':    row.get(f'{found_corner}_wins',          0),
        'losses':  row.get(f'{found_corner}_losses',        0),
        # momentum / experience
        'last_3_wins':       row.get(f'{found_corner}_last_3_wins',           0),
        'last_6_wins':       row.get(f'{found_corner}_last_6_wins',           0),
        'ufc_fights_before': row.get(f'{found_corner}_ufc_fights_before',     0),
        'days_since':        row.get(f'{found_corner}_days_since_last_fight', 365),
        'quality_last_3':    row.get(f'{found_corner}_quality_last_3',        0),
        'quality_last_6':    row.get(f'{found_corner}_quality_last_6',        0),
        # finish quality (Elo-weighted) — 0 prior weighted finishes is a
        # true statement for a fighter with no finish history, so 0 is the
        # correct default here (unlike chin/finish/decision rate below)
        'weighted_ko_win':   row.get(f'{found_corner}_total_weighted_ko_win',  0),
        'weighted_sub_win':  row.get(f'{found_corner}_total_weighted_sub_win', 0),
        'weighted_ko_loss':  row.get(f'{found_corner}_total_weighted_ko_loss', 0),
        'weighted_sub_loss': row.get(f'{found_corner}_total_weighted_sub_loss',0),
        # chin / finish tendency — default to the TRAIN-ONLY mean, not 0,
        # since 0 would falsely claim "never been finished / never finishes"
        'chin_metric':   row.get(f'{found_corner}_chin_metric',   DEFAULTS["chin_metric"]),
        'finish_rate':   row.get(f'{found_corner}_finish_rate',   DEFAULTS["finish_rate"]),
        'decision_rate': row.get(f'{found_corner}_decision_rate', DEFAULTS["decision_rate"]),
    }

    # replace NaN with the same defaults used above (row.get only catches
    # a missing COLUMN, not a present-but-NaN cell)
    nan_fallbacks = {
        'chin_metric': DEFAULTS["chin_metric"],
        'finish_rate': DEFAULTS["finish_rate"],
        'decision_rate': DEFAULTS["decision_rate"],
    }
    for key in stats:
        if pd.isna(stats[key]):
            stats[key] = nan_fallbacks.get(key, 0)

    return stats


def composite_scores(stats):
    """
    Mirrors the notebook's offense/defense composite score formulas exactly,
    using the persisted train-only z-score constants.
    """
    z_splm     = zscore(stats['splm'],     "splm")
    z_str_acc  = zscore(stats['str_acc'],  "str_acc")
    z_str_def  = zscore(stats['str_def'],  "str_def")
    z_sapm     = zscore(stats['sapm'],     "sapm")
    z_td_avg   = zscore(stats['td_avg'],   "td_avg")
    z_td_acc   = zscore(stats['td_acc'],   "td_acc")
    z_td_def   = zscore(stats['td_def'],   "td_def")
    z_sub_avg  = zscore(stats['sub_avg'],  "sub_avg")
    z_ko_win_q  = zscore(stats['weighted_ko_win'],   "ko_win_q")
    z_sub_win_q = zscore(stats['weighted_sub_win'],  "sub_win_q")
    z_ko_loss_q = zscore(stats['weighted_ko_loss'],  "ko_loss_q")
    z_sub_loss_q= zscore(stats['weighted_sub_loss'], "sub_loss_q")
    z_chin      = zscore(stats['chin_metric'], "chin")

    striking_offense  = z_splm + z_str_acc + z_ko_win_q
    striking_defense  = z_str_def - z_sapm - z_chin - z_ko_loss_q
    grappling_offense = z_td_avg + z_td_acc + z_sub_avg + z_sub_win_q
    grappling_defense = z_td_def - z_sub_loss_q

    return {
        "striking_offense": striking_offense,
        "striking_defense": striking_defense,
        "grappling_offense": grappling_offense,
        "grappling_defense": grappling_defense,
    }


def predict_fight(red_name, blue_name):
    try:
        red  = get_fighter_stats(red_name)
        blue = get_fighter_stats(blue_name)
    except IndexError:
        return {"error": "Fighter Not Found. Check for proper spelling."}, 404

    if red_name == blue_name:
        return {"error": "Please enter two different fighters."}, 400

    # elo lookup — use frozen training elo, default 1500 if unseen
    r_elo = fighter_elo.get(red_name,  1500)
    b_elo = fighter_elo.get(blue_name, 1500)

    r_win_rate = red['wins']  / (red['wins']  + red['losses']  + 1e-6)
    b_win_rate = blue['wins'] / (blue['wins'] + blue['losses'] + 1e-6)

    r_composites = composite_scores(red)
    b_composites = composite_scores(blue)

    r_age_from_prime = abs(red['age']  - PRIME_AGE)
    b_age_from_prime = abs(blue['age'] - PRIME_AGE)
    age_diff = red['age'] - blue['age']

    features = pd.DataFrame([{
        # physical
        "age_diff":            age_diff,
        "reach_diff":          red['reach']  - blue['reach'],
        "height_diff":         red['height'] - blue['height'],
        "weight_class":        red['weight'],
        "age_from_prime_diff": r_age_from_prime - b_age_from_prime,
        "age_diff_sq":         age_diff ** 2,

        # striking
        "str_volume_diff":  red['splm']    - blue['splm'],
        "str_acc_diff":     red['str_acc'] - blue['str_acc'],
        "str_def_diff":     red['str_def'] - blue['str_def'],
        "sapm_diff":        red['sapm']    - blue['sapm'],

        # grappling
        "td_volume_diff":   red['td_avg']  - blue['td_avg'],
        "td_acc_diff":      red['td_acc']  - blue['td_acc'],
        "td_def_diff":      red['td_def']  - blue['td_def'],
        "sub_diff":         red['sub_avg'] - blue['sub_avg'],

        # experience
        "fights_diff":        red['ufc_fights_before'] - blue['ufc_fights_before'],
        "rest_diff":          red['days_since']         - blue['days_since'],
        "finish_rate_diff":   red['finish_rate']   - blue['finish_rate'],
        "decision_rate_diff": red['decision_rate'] - blue['decision_rate'],

        # momentum
        "momentum_3_diff":  (red['last_3_wins'] / 3) - (blue['last_3_wins'] / 3),
        "momentum_6_diff":  (red['last_6_wins'] / 6) - (blue['last_6_wins'] / 6),

        # quality
        "quality_3_diff":   (red['quality_last_3'] / 3) - (blue['quality_last_3'] / 3),
        "quality_6_diff":   (red['quality_last_6'] / 6) - (blue['quality_last_6'] / 6),

        # win rate
        "win_rate_diff":    r_win_rate - b_win_rate,

        # elo
        "elo_diff":         r_elo - b_elo,
        "total_elo":        r_elo * b_elo,

        # chin
        "chin_diff": red['chin_metric'] - blue['chin_metric'],

        # offense/defense composites
        "striking_offense_diff":  r_composites["striking_offense"] - b_composites["striking_offense"],
        "striking_defense_diff":  r_composites["striking_defense"] - b_composites["striking_defense"],
        "grappling_offense_diff": r_composites["grappling_offense"] - b_composites["grappling_offense"],
        "grappling_defense_diff": r_composites["grappling_defense"] - b_composites["grappling_defense"],

        # matchup interactions (offense vs opponent's defense)
        "striking_edge_diff": (
            (r_composites["striking_offense"] - b_composites["striking_defense"])
            - (b_composites["striking_offense"] - r_composites["striking_defense"])
        ),
        "grappling_edge_diff": (
            (r_composites["grappling_offense"] - b_composites["grappling_defense"])
            - (b_composites["grappling_offense"] - r_composites["grappling_defense"])
        ),

        # multiplicative interactions
        "striking_offense_interaction":  r_composites["striking_offense"] * b_composites["striking_offense"],
        "grappling_offense_interaction": r_composites["grappling_offense"] * b_composites["grappling_offense"],
    }])

    # match the exact feature order/names the model was trained on
    features = features[model.get_booster().feature_names]

    # make prediction and probabilities
    prediction = model.predict(features)[0]
    prob_red   = float(model.predict_proba(features)[0][1])
    prob_blue  = float(model.predict_proba(features)[0][0])

    if prediction == 1:
        confidence = round(prob_red * 100, 2)
        return f"{red_name.title()} will likely win ({confidence}% confidence)"
    else:
        confidence = round(prob_blue * 100, 2)
        return f"{blue_name.title()} will likely win ({confidence}% confidence)"


# Flask wrapper

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "https://aidans-ufc-predictor.vercel.app",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175"
]}})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Expects JSON: { "red": "fighter name", "blue": "fighter name" }
    Returns JSON: { "message": "..." }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Send JSON with 'red' and 'blue' fields."}), 400

    red  = data.get("red",  "").lower().strip()
    blue = data.get("blue", "").lower().strip()

    if not red or not blue:
        return jsonify({"error": "Both 'red' and 'blue' must be provided."}), 400

    result = predict_fight(red, blue)

    # predict_fight returns either a string or a (dict, status) tuple on error
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]

    return jsonify({"message": result}), 200

@app.route("/api/fighters", methods=["GET"])
def api_fighters():
    """
    Return a JSON list of fighters to help frontend autocomplete.
    """
    names = set()
    if 'r_name' in df.columns:
        names.update(df['r_name'].dropna().astype(str).str.strip().tolist())
    if 'b_name' in df.columns:
        names.update(df['b_name'].dropna().astype(str).str.strip().tolist())

    return jsonify({"fighters": sorted([n.lower() for n in names])})


# run the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=5000, debug=False)