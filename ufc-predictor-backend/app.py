import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import shap

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "model/ufc_xgb_model.pkl")
ELO_PATH     = os.path.join(BASE_DIR, "model/ufc_fighter_elo.pkl")
SCALERS_PATH = os.path.join(BASE_DIR, "model/ufc_feature_scalers.pkl")
CSV_PATH     = os.path.join(BASE_DIR, "data/UFC.csv")

# ============================================================
# LOAD MODEL, ELO, SCALERS
# ============================================================

model       = joblib.load(MODEL_PATH)
fighter_elo = joblib.load(ELO_PATH)
scaler_data = joblib.load(SCALERS_PATH)
SCALERS     = scaler_data["scalers"]
DEFAULTS    = scaler_data["defaults"]
PRIME_AGE   = DEFAULTS["prime_age"]

# Build the SHAP explainer ONCE at startup — it's tied to the model,
# not the request, and rebuilding it per-request is pure waste.
explainer = shap.TreeExplainer(model)

df = pd.read_csv(CSV_PATH).copy()

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
    matches = df[(df['r_name'] == fighter_name) | (df['b_name'] == fighter_name)]
    if matches.empty:
        raise IndexError

    row = matches.sort_values('date').iloc[-1]
    found_corner = 'r' if row['r_name'] == fighter_name else 'b'

    stats = {
        'splm':    row.get(f'{found_corner}_splm_x',        0),
        'sapm':    row.get(f'{found_corner}_sapm_x',        0),
        'str_acc': row.get(f'{found_corner}_str_acc_x',     0),
        'str_def': row.get(f'{found_corner}_str_def_x',     0),
        'td_avg':  row.get(f'{found_corner}_td_avg_x',      0),
        'td_acc':  row.get(f'{found_corner}_td_avg_acc_x',  0),
        'td_def':  row.get(f'{found_corner}_td_def_x',      0),
        'sub_avg': row.get(f'{found_corner}_sub_avg_x',     0),
        'reach':   row.get(f'{found_corner}_reach',         0),
        'height':  row.get(f'{found_corner}_height',        0),
        'weight':  row.get('r_weight_x',                    0),
        'age':     row.get(f'{found_corner}_age',           0),
        'wins':    row.get(f'{found_corner}_wins',          0),
        'losses':  row.get(f'{found_corner}_losses',        0),
        'last_3_wins':       row.get(f'{found_corner}_last_3_wins',           0),
        'last_6_wins':       row.get(f'{found_corner}_last_6_wins',           0),
        'ufc_fights_before': row.get(f'{found_corner}_ufc_fights_before',     0),
        'days_since':        row.get(f'{found_corner}_days_since_last_fight', 365),
        'quality_last_3':    row.get(f'{found_corner}_quality_last_3',        0),
        'quality_last_6':    row.get(f'{found_corner}_quality_last_6',        0),
        'weighted_ko_win':   row.get(f'{found_corner}_total_weighted_ko_win',  0),
        'weighted_sub_win':  row.get(f'{found_corner}_total_weighted_sub_win', 0),
        'weighted_ko_loss':  row.get(f'{found_corner}_total_weighted_ko_loss', 0),
        'weighted_sub_loss': row.get(f'{found_corner}_total_weighted_sub_loss',0),
        'chin_metric':   row.get(f'{found_corner}_chin_metric',   DEFAULTS["chin_metric"]),
        'finish_rate':   row.get(f'{found_corner}_finish_rate',   DEFAULTS["finish_rate"]),
        'decision_rate': row.get(f'{found_corner}_decision_rate', DEFAULTS["decision_rate"]),
    }

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


# ============================================================
# SHAP EXPLANATION LAYER
# ============================================================
#
# Features are grouped into fight-relevant concepts so the output reads
# as "wrestling advantage" rather than a dump of raw diff columns.
# ============================================================

FEATURE_GROUPS = {
    "wrestling":           ["td_volume_diff", "td_acc_diff", "td_def_diff", "sub_diff"],
    "striking":            ["str_volume_diff", "str_acc_diff", "str_def_diff", "sapm_diff"],
    "physical":            ["reach_diff", "height_diff", "age_from_prime_diff", "age_diff_sq"],
    "experience":          ["fights_diff"],
    "recent_form":         ["momentum_3_diff", "momentum_6_diff"],
    "win_quality":         ["quality_3_diff", "quality_6_diff"],
    "overall_record":      ["win_rate_diff"],
    "rest":                ["rest_diff"],
    "elo":                 ["elo_diff", "total_elo"],
    "finishing_tendency":  ["finish_rate_diff", "decision_rate_diff", "chin_diff"],
    "striking_matchup":    ["striking_offense_diff", "striking_defense_diff",
                             "striking_edge_diff", "striking_offense_interaction"],
    "grappling_matchup":   ["grappling_offense_diff", "grappling_defense_diff",
                             "grappling_edge_diff", "grappling_offense_interaction"],
}

GROUP_LABELS = {
    "wrestling":          "grappling level",
    "striking":           "striking dominance",
    "physical":           "physical attributes",
    "experience":         "UFC experience",
    "recent_form":        "recent form",
    "win_quality":        "quality of opposition",
    "overall_record":     "overall career record",
    "rest":               "recent activity",
    "elo":                "level of competition",
    "finishing_tendency": "finishing ability and durability",
    "striking_matchup":   "specific striking matchup",
    "grappling_matchup":  "specific grappling matchup",
}

# Primary bar: a group needs at least this share of total |SHAP| to
# qualify on its own merit.
PRIMARY_THRESHOLD = 0.08

# If a side (advantages or concerns) has fewer than MIN_PER_SIDE factors
# after the primary threshold, backfill with the next-strongest factors
# on that side down to this floor — so a real side isn't empty just
# because everything individually sat a bit under 8%.
MIN_PER_SIDE = 2
BACKFILL_FLOOR = 0.02


def _strength_label(importance):
    if importance >= 0.30:
        return "Game-braker"
    elif importance >= 0.18:
        return "Very strong factor"
    elif importance >= 0.10:
        return "Strong factor"
    elif importance >= 0.05:
        return "Moderate factor"
    else:
        return "Minor factor"


def explain_prediction(features_df, predicted_is_red):
    """
    Returns a dict: {"advantages": [...], "concerns": [...]}
    Each entry: {group, label, importance (0-1), strength}

    The count on each side is DYNAMIC — every group clearing
    PRIMARY_THRESHOLD is included (no artificial cap), and if a side
    ends up with fewer than MIN_PER_SIDE, it's backfilled from
    remaining factors down to BACKFILL_FLOOR so a real signal isn't
    hidden just because it landed slightly under the main cutoff.
    """
    shap_values = explainer.shap_values(features_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values).flatten()

    rows = pd.DataFrame({"feature": features_df.columns, "shap": shap_values})
    abs_total = rows["shap"].abs().sum()

    all_factors = []
    for group, feats in FEATURE_GROUPS.items():
        subset = rows[rows["feature"].isin(feats)]
        if subset.empty or abs_total == 0:
            continue

        group_shap = subset["shap"].sum()
        importance = subset["shap"].abs().sum() / abs_total
        favours_predicted = (group_shap > 0) if predicted_is_red else (group_shap < 0)

        all_factors.append({
            "group": group,
            "label": GROUP_LABELS[group],
            "importance": float(importance),
            "favours_predicted": bool(favours_predicted),
        })

    def select_side(side_factors):
        side_factors = sorted(side_factors, key=lambda f: f["importance"], reverse=True)

        selected = [f for f in side_factors if f["importance"] >= PRIMARY_THRESHOLD]

        if len(selected) < MIN_PER_SIDE:
            remaining = [f for f in side_factors if f not in selected]
            for f in remaining:
                if len(selected) >= MIN_PER_SIDE:
                    break
                if f["importance"] >= BACKFILL_FLOOR:
                    selected.append(f)
            selected = sorted(selected, key=lambda f: f["importance"], reverse=True)

        for f in selected:
            f["importance"] = round(f["importance"], 3)
            f["strength"] = _strength_label(f["importance"])

        return selected

    advantages = select_side([f for f in all_factors if f["favours_predicted"]])
    concerns   = select_side([f for f in all_factors if not f["favours_predicted"]])

    return {"advantages": advantages, "concerns": concerns}


# ============================================================
# PREDICTION
# ============================================================

def predict_fight(red_name, blue_name):
    try:
        red  = get_fighter_stats(red_name)
        blue = get_fighter_stats(blue_name)
    except IndexError:
        return {"error": "Fighter Not Found. Check for proper spelling."}, 404

    if red_name == blue_name:
        return {"error": "Please enter two different fighters."}, 400

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
        "reach_diff":          red['reach']  - blue['reach'],
        "height_diff":         red['height'] - blue['height'],
        "weight_class":        red['weight'],
        "age_from_prime_diff": r_age_from_prime - b_age_from_prime,
        "age_diff_sq":         age_diff ** 2,

        "str_volume_diff":  red['splm']    - blue['splm'],
        "str_acc_diff":     red['str_acc'] - blue['str_acc'],
        "str_def_diff":     red['str_def'] - blue['str_def'],
        "sapm_diff":        red['sapm']    - blue['sapm'],

        "td_volume_diff":   red['td_avg']  - blue['td_avg'],
        "td_acc_diff":      red['td_acc']  - blue['td_acc'],
        "td_def_diff":      red['td_def']  - blue['td_def'],
        "sub_diff":         red['sub_avg'] - blue['sub_avg'],

        "fights_diff":        red['ufc_fights_before'] - blue['ufc_fights_before'],
        "rest_diff":          red['days_since']         - blue['days_since'],
        "finish_rate_diff":   red['finish_rate']   - blue['finish_rate'],
        "decision_rate_diff": red['decision_rate'] - blue['decision_rate'],

        "momentum_3_diff":  (red['last_3_wins'] / 3) - (blue['last_3_wins'] / 3),
        "momentum_6_diff":  (red['last_6_wins'] / 6) - (blue['last_6_wins'] / 6),

        "quality_3_diff":   (red['quality_last_3'] / 3) - (blue['quality_last_3'] / 3),
        "quality_6_diff":   (red['quality_last_6'] / 6) - (blue['quality_last_6'] / 6),

        "win_rate_diff":    r_win_rate - b_win_rate,

        "elo_diff":         r_elo - b_elo,
        "total_elo":        r_elo * b_elo,

        "chin_diff": red['chin_metric'] - blue['chin_metric'],

        "striking_offense_diff":  r_composites["striking_offense"] - b_composites["striking_offense"],
        "striking_defense_diff":  r_composites["striking_defense"] - b_composites["striking_defense"],
        "grappling_offense_diff": r_composites["grappling_offense"] - b_composites["grappling_offense"],
        "grappling_defense_diff": r_composites["grappling_defense"] - b_composites["grappling_defense"],

        "striking_edge_diff": (
            (r_composites["striking_offense"] - b_composites["striking_defense"])
            - (b_composites["striking_offense"] - r_composites["striking_defense"])
        ),
        "grappling_edge_diff": (
            (r_composites["grappling_offense"] - b_composites["grappling_defense"])
            - (b_composites["grappling_offense"] - r_composites["grappling_defense"])
        ),

        "striking_offense_interaction":  r_composites["striking_offense"] * b_composites["striking_offense"],
        "grappling_offense_interaction": r_composites["grappling_offense"] * b_composites["grappling_offense"],
    }])

    features = features[model.get_booster().feature_names]

    prediction = model.predict(features)[0]
    prob_red   = float(model.predict_proba(features)[0][1])
    prob_blue  = float(model.predict_proba(features)[0][0])

    predicted_is_red = (prediction == 1)
    winner_name = red_name if predicted_is_red else blue_name
    confidence = round((prob_red if predicted_is_red else prob_blue) * 100, 2)

    explanation = explain_prediction(features, predicted_is_red)

    return {
        "message": f"{winner_name.title()} will likely win ({confidence}% confidence)",
        "winner": winner_name.title(),
        "loser": (blue_name if predicted_is_red else red_name).title(),
        "confidence": confidence,
        "predicted_corner": "red" if predicted_is_red else "blue",
        "factors": explanation,
    }


# ============================================================
# FLASK APP
# ============================================================

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
    Returns JSON: {
        "message": "...",
        "winner": "...",
        "loser": "...",
        "confidence": 73.2,
        "predicted_corner": "red" | "blue",
        "factors": {
            "advantages": [{group, label, importance, strength}, ...],
            "concerns":   [{group, label, importance, strength}, ...]
        }
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Send JSON with 'red' and 'blue' fields."}), 400

    red  = data.get("red",  "").lower().strip()
    blue = data.get("blue", "").lower().strip()

    if not red or not blue:
        return jsonify({"error": "Both 'red' and 'blue' must be provided."}), 400

    result = predict_fight(red, blue)

    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]

    return jsonify(result), 200


@app.route("/api/fighters", methods=["GET"])
def api_fighters():
    names = set()
    if 'r_name' in df.columns:
        names.update(df['r_name'].dropna().astype(str).str.strip().tolist())
    if 'b_name' in df.columns:
        names.update(df['b_name'].dropna().astype(str).str.strip().tolist())

    return jsonify({"fighters": sorted([n.lower() for n in names])})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)