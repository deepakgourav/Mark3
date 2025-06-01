from flask import Flask, request, jsonify, render_template
import json
import os
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ✅ Format hand inputs like '555' => '5-5-5'
def fix_hand(hand_str):
    valid_cards = {'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'}
    hand_str = str(hand_str).replace('-', '').upper()
    cards = []
    i = 0
    while i < len(hand_str):
        if hand_str[i] == '1' and i + 1 < len(hand_str) and hand_str[i + 1] == '0':
            cards.append('10')
            i += 2
        else:
            cards.append(hand_str[i])
            i += 1
    return '-'.join(card for card in cards if card in valid_cards)

app = Flask(__name__)
DATA_DIR = 'data'
HISTORICAL_FILE = os.path.join(DATA_DIR, 'game_data.json')
PREDICT_FEED_FILE = os.path.join(DATA_DIR, 'feed_data.json')

# Ensure data directory and files exist
os.makedirs(DATA_DIR, exist_ok=True)
for file in [HISTORICAL_FILE, PREDICT_FEED_FILE]:
    if not os.path.exists(file):
        with open(file, 'w') as f:
            json.dump([], f, indent=2)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Data is not a list")
            return data
    except Exception as e:
        logger.error(f"Load error from {file_path}: {e}")
        return []

def save_data(file_path, data):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Save error to {file_path}: {e}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add_game', methods=['POST'])
def add_game():
    try:
        data = request.get_json()
        player_hand = fix_hand(data.get('player_hand'))
        banker_hand = fix_hand(data.get('banker_hand'))
        outcome = data.get('outcome')

        if not all([player_hand, banker_hand, outcome]):
            return jsonify({'error': 'Missing fields'}), 400

        games = load_data(HISTORICAL_FILE)
        games.append({
            'player_hand': player_hand,
            'banker_hand': banker_hand,
            'outcome': outcome
        })

        if not save_data(HISTORICAL_FILE, games):
            return jsonify({'error': 'Save failed'}), 500

        return jsonify({'success': True}), 201

    except Exception as e:
        logger.error(f"Add game error: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/add_outcome', methods=['POST'])
def add_outcome():
    try:
        data = request.get_json()
        outcome = data.get('outcome')

        if outcome not in ['Player', 'Banker', 'Tie']:
            return jsonify({'error': 'Invalid outcome'}), 400

        feed = load_data(PREDICT_FEED_FILE)
        feed.append({'outcome': outcome})

        if not save_data(PREDICT_FEED_FILE, feed):
            return jsonify({'error': 'Failed to save outcome'}), 500

        return jsonify({'success': True, 'total_rounds': len(feed)}), 201

    except Exception as e:
        logger.error(f"Add outcome error: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/predict', methods=['GET'])
def predict():
    try:
        games = load_data(HISTORICAL_FILE)
        feed = load_data(PREDICT_FEED_FILE)
        combined = games + feed

        if len(combined) < 20:
            return jsonify({'error': 'Need at least 20 rounds (historical + feed)'}), 400

        outcome_map = {"Player": 0, "Banker": 1, "Tie": 2}
        reverse_map = {0: "Player", 1: "Banker", 2: "Tie"}
        X, y = [], []

        for i in range(len(combined) - 10):
            sequence = combined[i:i+10]
            features = [outcome_map.get(g['outcome'], -1) for g in sequence]
            if -1 in features:
                continue
            streak = 1
            for j in range(1, 10):
                if sequence[j]['outcome'] == sequence[j-1]['outcome']:
                    streak += 1
                else:
                    streak = 1
            features.append(streak)
            label = outcome_map.get(combined[i + 10]['outcome'], -1)
            if label == -1:
                continue
            X.append(features)
            y.append(label)

        if not X or not y:
            return jsonify({'error': 'Not enough clean data to train'}), 500

        model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
                ('xgb', XGBClassifier(eval_metric='mlogloss')),
                ('lr', LogisticRegression(max_iter=1000))
            ],
            voting='soft'
        )
        model.fit(np.array(X), np.array(y))

        if len(feed) < 10:
            return jsonify({'error': 'Need at least 10 recent outcomes for prediction'}), 400

        recent = feed[-10:]
        test_features = [outcome_map.get(g['outcome'], -1) for g in recent]
        if -1 in test_features:
            return jsonify({'error': 'Invalid outcome in feed'}), 400

        streak = 1
        for j in range(1, 10):
            if recent[j]['outcome'] == recent[j-1]['outcome']:
                streak += 1
            else:
                streak = 1
        test_features.append(streak)

        input_array = np.array([test_features])
        prediction = int(model.predict(input_array)[0])
        proba = model.predict_proba(input_array)[0]
        confidence = {reverse_map[i]: f"{p*100:.1f}%" for i, p in enumerate(proba)}

        return jsonify({
            "prediction": reverse_map[prediction],
            "confidence": confidence,
            "feed_length": len(feed)
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
