import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import os # For file path checking

# --- THIS MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Tennis Match Predictor", layout="wide", initial_sidebar_state="expanded")
# --- END OF FIRST STREAMLIT COMMAND ---

# --- 0. Configuration and Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get directory of this app.py script
MODEL_PATH = os.path.join(BASE_DIR, 'tennis_predictor_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'tennis_data_scaler.pkl')
TRAINING_COLS_PATH = os.path.join(BASE_DIR, 'training_columns.json')
NUMERICAL_COLS_PATH = os.path.join(BASE_DIR, 'numerical_features_list.json')
CONFIG_PATH = os.path.join(BASE_DIR, 'model_config.json')
HISTORICAL_DATA_PATH = os.path.join(BASE_DIR, 'all_matches_data.csv')

# --- 1. Load Trained Model and Supporting Objects ---
@st.cache_resource # Cache loading of model and scaler for performance
def load_model_objects():
    loaded_objects = {"model": None, "scaler": None, "training_cols": None, 
                      "numerical_cols": None, "has_elo": False}
    all_files_present = True
    
    files_to_check = {
        "model": MODEL_PATH, "scaler": SCALER_PATH, "training_cols": TRAINING_COLS_PATH,
        "numerical_cols": NUMERICAL_COLS_PATH, "config": CONFIG_PATH
    }

    for name, path in files_to_check.items():
        if not os.path.exists(path):
            st.error(f"Critical file not found: {os.path.basename(path)}. Please ensure it's in the directory: {BASE_DIR}")
            all_files_present = False
            
    if not all_files_present:
        st.error("Application cannot start due to missing model/configuration files.")
        return loaded_objects 

    try:
        with open(MODEL_PATH, 'rb') as f:
            loaded_objects["model"] = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            loaded_objects["scaler"] = pickle.load(f)
        with open(TRAINING_COLS_PATH, 'r') as f:
            loaded_objects["training_cols"] = json.load(f)
        with open(NUMERICAL_COLS_PATH, 'r') as f:
            loaded_objects["numerical_cols"] = json.load(f)
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        loaded_objects["has_elo"] = config.get('has_elo', False)
        
        st.success("Model and supporting objects loaded successfully!")
        return loaded_objects
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model objects: {e}")
        # Return dictionary with None values to prevent NameErrors later
        return {"model": None, "scaler": None, "training_cols": None, "numerical_cols": None, "has_elo": False}

MODEL_OBJECTS = load_model_objects()
model = MODEL_OBJECTS["model"]
scaler = MODEL_OBJECTS["scaler"]
TRAINING_COLUMNS = MODEL_OBJECTS["training_cols"]
NUMERICAL_FEATURES_TO_SCALE = MODEL_OBJECTS["numerical_cols"]
HAS_ELO_FEATURES = MODEL_OBJECTS["has_elo"]


# --- 2. Data Loading and Preparation for "Live" Stats ---
@st.cache_data # Cache the loaded historical data
def load_historical_data(data_path):
    if not os.path.exists(data_path):
        st.error(f"Historical data file not found: {os.path.basename(data_path)} at {data_path}. "
                 "Please save your combined df_matches from the notebook to this file named 'all_matches_data.csv' in the app directory.")
        return pd.DataFrame()

    try:
        st.info(f"Loading historical data from {os.path.basename(data_path)}... This might take a moment.")
        df = pd.read_csv(data_path, low_memory=False)
        
        if 'tourney_date' not in df.columns:
            st.error("'tourney_date' column missing from historical data. Cannot proceed.")
            return pd.DataFrame()
        try:
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
        except: 
            df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
        df.dropna(subset=['tourney_date'], inplace=True)
        df.sort_values(by='tourney_date', inplace=True)

        for col_prefix in ['winner', 'loser']:
            id_col = f'{col_prefix}_id'
            if id_col in df.columns:
                df[id_col] = pd.to_numeric(df[id_col], errors='coerce').fillna(-1).astype(int)
            else:
                st.warning(f"Column {id_col} not found in historical data.")
        
        df.dropna(subset=['winner_id', 'loser_id'], inplace=True)
        df = df[(df['winner_id'] != -1) & (df['loser_id'] != -1)] # Filter out placeholder IDs

        st.success("Historical data loaded and preprocessed.")
        return df
    except Exception as e:
        st.error(f"Error loading or preprocessing historical data: {e}")
        return pd.DataFrame()

df_historical_matches = load_historical_data(HISTORICAL_DATA_PATH)


# --- 3. Helper Functions to Get Player Stats (SIMPLIFIED - PLACEHOLDER LOGIC) ---
def get_player_latest_stats(player_id_int, prediction_date, df_history_full, surface_context):
    default_stats = {'id': player_id_int, 'elo': 1500 if HAS_ELO_FEATURES else np.nan, 'rank': 1000, 
                     'rank_points': 0, 'ht': 180, 'age': 25, 'hand': 'U'}
    if TRAINING_COLUMNS: # Ensure TRAINING_COLUMNS is not None
        for col_template in TRAINING_COLUMNS:
            if col_template.startswith('p1_roll_'): 
                actual_col_name = col_template.replace('p1_', '')
                default_stats[actual_col_name] = (0.5 if 'pct' in actual_col_name else (0.05 if 'rate' in actual_col_name else 0))
            elif col_template.startswith('p1_matches_last_') or col_template.startswith('p1_minutes_last_'):
                 default_stats[col_template.replace('p1_', '')] = 0
    
    if df_history_full.empty:
        st.warning(f"Historical data is empty. Using default stats for Player ID {player_id_int}.")
        return default_stats

    player_matches_all_time = df_history_full[
        ((df_history_full['winner_id'] == player_id_int) | (df_history_full['loser_id'] == player_id_int)) &
        (df_history_full['tourney_date'] < prediction_date)
    ].sort_values(by='tourney_date', ascending=False)

    if player_matches_all_time.empty:
        return default_stats

    latest_match_played = player_matches_all_time.iloc[0]
    is_winner_latest = latest_match_played['winner_id'] == player_id_int
    
    stats = default_stats.copy() # Start with defaults, then override
    stats.update({
        'id': player_id_int,
        'elo': latest_match_played[f"{'winner' if is_winner_latest else 'loser'}_elo"] if HAS_ELO_FEATURES and f"{'winner'if is_winner_latest else 'loser'}_elo" in latest_match_played else (1500 if HAS_ELO_FEATURES else np.nan),
        'rank': latest_match_played[f"{'winner' if is_winner_latest else 'loser'}_rank"] if f"{'winner'if is_winner_latest else 'loser'}_rank" in latest_match_played else 1000,
        'rank_points': latest_match_played[f"{'winner' if is_winner_latest else 'loser'}_rank_points"] if f"{'winner'if is_winner_latest else 'loser'}_rank_points" in latest_match_played else 0,
        'ht': latest_match_played[f"{'winner' if is_winner_latest else 'loser'}_ht"] if f"{'winner'if is_winner_latest else 'loser'}_ht" in latest_match_played else 180,
        'age': latest_match_played[f"{'winner' if is_winner_latest else 'loser'}_age"] if f"{'winner'if is_winner_latest else 'loser'}_age" in latest_match_played else 25, # Stale age
        'hand': latest_match_played[f"{'winner' if is_winner_latest else 'loser'}_hand"] if f"{'winner'if is_winner_latest else 'loser'}_hand" in latest_match_played else 'U',
    })
    
    # Simplified Rolling Stats (Placeholder - real app needs robust calculation)
    last_10_matches = player_matches_all_time.head(10)
    if not last_10_matches.empty:
        won_count_10 = sum(last_10_matches['winner_id'] == player_id_int)
        stats['roll_overall_win_pct_10'] = won_count_10 / len(last_10_matches) if len(last_10_matches) > 0 else 0.5
    
    last_5_surface_matches = player_matches_all_time[player_matches_all_time['surface'] == surface_context].head(5)
    if not last_5_surface_matches.empty:
        won_count_surface_5 = sum(last_5_surface_matches['winner_id'] == player_id_int)
        stats['roll_surface_win_pct_5'] = won_count_surface_5 / len(last_5_surface_matches) if len(last_5_surface_matches) > 0 else 0.5
        
    # Fill NaNs in the collected stats again with robust defaults
    for k, v in stats.items():
        if pd.isna(v):
            if 'elo' in k and HAS_ELO_FEATURES: stats[k] = 1500
            elif 'elo' in k and not HAS_ELO_FEATURES: stats[k] = 0 # Or some other placeholder if elo not used
            elif 'rank' in k and 'points' not in k : stats[k] = 1000
            elif 'points' in k : stats[k] = 0
            elif 'ht' in k : stats[k] = 180
            elif 'age' in k : stats[k] = 25
            elif 'hand' in k : stats[k] = 'U'
            elif 'pct' in k : stats[k] = 0.5
            elif 'rate' in k : stats[k] = 0.05
            else: stats[k] = 0
    return stats

def get_h2h_stats(player1_id_int, player2_id_int, prediction_date, df_history_full):
    if df_history_full.empty:
        return {'p1_h2h_wins': 0, 'p2_h2h_wins': 0}
        
    h2h_matches = df_history_full[
        (((df_history_full['winner_id'] == player1_id_int) & (df_history_full['loser_id'] == player2_id_int)) |
         ((df_history_full['winner_id'] == player2_id_int) & (df_history_full['loser_id'] == player1_id_int))) &
        (df_history_full['tourney_date'] < prediction_date)
    ]
    p1_wins = len(h2h_matches[h2h_matches['winner_id'] == player1_id_int])
    p2_wins = len(h2h_matches[h2h_matches['winner_id'] == player2_id_int])
    return {'p1_h2h_wins': p1_wins, 'p2_h2h_wins': p2_wins}

# --- 4. Prediction Function ---
def make_prediction_for_app(p1_live_stats, p2_live_stats, match_context_live, h2h_live_stats):
    if model is None or scaler is None or TRAINING_COLUMNS is None or NUMERICAL_FEATURES_TO_SCALE is None:
        st.error("Model or supporting objects not loaded. Cannot make prediction.")
        return None, None

    feature_vector = {}

    for stat_key, live_val in p1_live_stats.items():
        if stat_key != 'id': feature_vector[f'p1_{stat_key}'] = live_val
    for stat_key, live_val in p2_live_stats.items():
        if stat_key != 'id': feature_vector[f'p2_{stat_key}'] = live_val
    
    feature_vector['p1_h2h_wins'] = h2h_live_stats.get('p1_h2h_wins', 0)
    feature_vector['p2_h2h_wins'] = h2h_live_stats.get('p2_h2h_wins', 0)
    
    for diff_col_name in TRAINING_COLUMNS:
        if diff_col_name.endswith('_diff'):
            base_feature_name = diff_col_name.replace('_diff', '')
            p1_col_name = f'p1_{base_feature_name}'
            p2_col_name = f'p2_{base_feature_name}'
            val1 = pd.to_numeric(feature_vector.get(p1_col_name), errors='coerce')
            val2 = pd.to_numeric(feature_vector.get(p2_col_name), errors='coerce')
            if pd.notna(val1) and pd.notna(val2):
                feature_vector[diff_col_name] = val1 - val2
            else:
                feature_vector[diff_col_name] = np.nan 

    for col in TRAINING_COLUMNS:
        is_dummy_or_interaction = False
        for prefix in ['surface_', 'tourney_level_', 'best_of_', 'round_', 'p1_hand_', 'p2_hand_']:
            if col.startswith(prefix): is_dummy_or_interaction = True; break
        if col in ['p1_lefty_vs_p2_righty', 'p1_righty_vs_p2_lefty', 'both_lefty', 'both_righty']:
            is_dummy_or_interaction = True
        if is_dummy_or_interaction and col not in feature_vector:
            feature_vector[col] = 0

    if 'surface' in match_context_live and f"surface_{match_context_live['surface']}" in TRAINING_COLUMNS:
        feature_vector[f"surface_{match_context_live['surface']}"] = 1
    if 'tourney_level' in match_context_live and f"tourney_level_{match_context_live['tourney_level']}" in TRAINING_COLUMNS:
        feature_vector[f"tourney_level_{match_context_live['tourney_level']}"] = 1
    if 'best_of' in match_context_live and f"best_of_{match_context_live['best_of']}" in TRAINING_COLUMNS:
        feature_vector[f"best_of_{match_context_live['best_of']}"] = 1
    if 'round' in match_context_live and f"round_{match_context_live['round']}" in TRAINING_COLUMNS:
        feature_vector[f"round_{match_context_live['round']}"] = 1
        
    p1_hand_val = p1_live_stats.get('hand', 'U')
    p2_hand_val = p2_live_stats.get('hand', 'U')
    if f'p1_hand_{p1_hand_val}' in TRAINING_COLUMNS: feature_vector[f'p1_hand_{p1_hand_val}'] = 1
    if f'p2_hand_{p2_hand_val}' in TRAINING_COLUMNS: feature_vector[f'p2_hand_{p2_hand_val}'] = 1
    
    # Ensure these specific interaction features are explicitly int 0 or 1
    for interaction_col in ['p1_lefty_vs_p2_righty', 'p1_righty_vs_p2_lefty', 'both_lefty', 'both_righty']:
        if interaction_col in TRAINING_COLUMNS: # Only if this feature was trained on
            is_interaction_true = False
            if interaction_col == 'p1_lefty_vs_p2_righty' and p1_hand_val == 'L' and p2_hand_val == 'R': is_interaction_true = True
            if interaction_col == 'p1_righty_vs_p2_lefty' and p1_hand_val == 'R' and p2_hand_val == 'L': is_interaction_true = True
            if interaction_col == 'both_lefty' and p1_hand_val == 'L' and p2_hand_val == 'L': is_interaction_true = True
            if interaction_col == 'both_righty' and p1_hand_val == 'R' and p2_hand_val == 'R': is_interaction_true = True
            feature_vector[interaction_col] = 1 if is_interaction_true else 0
        elif interaction_col not in feature_vector: # if it was in training_cols but not explicitly set above
             feature_vector[interaction_col] = 0


    input_df_row = pd.Series(feature_vector, dtype=object).reindex(TRAINING_COLUMNS)
    input_df = pd.DataFrame([input_df_row], columns=TRAINING_COLUMNS)

    for col in input_df.columns:
        # Attempt to convert to numeric first; 'ignore' leaves non-convertible as is (object)
        input_df[col] = pd.to_numeric(input_df[col], errors='ignore')
        
        if input_df[col].isnull().any(): # If NaNs exist after numeric conversion attempt
            if pd.api.types.is_numeric_dtype(input_df[col].dtype):
                input_df[col].fillna(0, inplace=True) # Fill numeric NaNs with 0
            else: # If still object (e.g., all NaNs or unconvertible strings)
                input_df[col].fillna(0, inplace=True) # Fill object NaNs also with 0
        
        # Now, explicitly cast known binary/dummy columns to int after NaNs are handled
        # This is to fix the 'object' dtype issue for these specific columns
        is_binary_col = False
        for prefix in ['surface_', 'tourney_level_', 'best_of_', 'round_', 'p1_hand_', 'p2_hand_']:
            if col.startswith(prefix): is_binary_col = True; break
        if col in ['p1_lefty_vs_p2_righty', 'p1_righty_vs_p2_lefty', 'both_lefty', 'both_righty']:
            is_binary_col = True
        
        if is_binary_col:
            try:
                # Ensure all values are suitable for int conversion (e.g. no floats like 0.0, 1.0 if they occurred)
                input_df[col] = input_df[col].round().astype(int) 
            except ValueError as e_cast:
                st.warning(f"Could not convert column '{col}' to int (values: {input_df[col].unique()[:5]}). Error: {e_cast}. Setting to 0.")
                input_df[col] = 0 # Fallback if conversion fails

    input_df_scaled = input_df.copy()
    if NUMERICAL_FEATURES_TO_SCALE is None:
        st.error("List of numerical features to scale not loaded. Cannot proceed.")
        return None, None
        
    cols_to_scale_present = [col for col in NUMERICAL_FEATURES_TO_SCALE if col in input_df_scaled.columns]
    if cols_to_scale_present:
        try:
            for col_to_scale in cols_to_scale_present:
                if not pd.api.types.is_numeric_dtype(input_df_scaled[col_to_scale].dtype):
                    st.error(f"Critical: Column '{col_to_scale}' for scaling has non-numeric dtype {input_df_scaled[col_to_scale].dtype}. Prediction may fail.")
                    # Attempt last-ditch conversion or raise error
                    input_df_scaled[col_to_scale] = pd.to_numeric(input_df_scaled[col_to_scale], errors='coerce').fillna(0)

            input_df_scaled[cols_to_scale_present] = scaler.transform(input_df_scaled[cols_to_scale_present])
        except Exception as e:
            st.error(f"Error during scaling input for prediction: {e}")
            return None, None
    
    try:
        pred_proba = model.predict_proba(input_df_scaled)[0]
        return pred_proba[1], pred_proba[0] 
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        return None, None

# --- 5. Streamlit UI ---
st.title("🎾 Ultimate Tennis Match Predictor")

if model is None or df_historical_matches.empty or TRAINING_COLUMNS is None:
    st.error("Application initialization failed. Essential components (model, historical data, or training columns) are missing.")
else:
    st.sidebar.header("Match Input")
    
    player_name_options = ["Enter Player ID Manually"]
    display_name_to_id_map = {}
    if not df_historical_matches.empty:
        ids_names1 = df_historical_matches[['winner_id', 'winner_name']].copy().rename(columns={'winner_id':'id', 'winner_name':'name'})
        ids_names2 = df_historical_matches[['loser_id', 'loser_name']].copy().rename(columns={'loser_id':'id', 'loser_name':'name'})
        if not ids_names1.empty or not ids_names2.empty:
            unique_players_df = pd.concat([ids_names1, ids_names2]).drop_duplicates(subset=['id']).dropna(subset=['id','name'])
            unique_players_df['id'] = unique_players_df['id'].astype(int)
            unique_players_df = unique_players_df[unique_players_df['id'] != -1]
            unique_players_df.sort_values('name', inplace=True)
            unique_players_df['display_name'] = unique_players_df['name'] + " (ID: " + unique_players_df['id'].astype(str) + ")"
            player_name_options.extend(unique_players_df['display_name'].tolist())
            display_name_to_id_map = pd.Series(unique_players_df['id'].values, index=unique_players_df['display_name']).to_dict()

    st.sidebar.subheader("Players")
    player1_display_selected = st.sidebar.selectbox("Player 1:", player_name_options, index=0, key="p1_name_select")
    player2_display_selected = st.sidebar.selectbox("Player 2:", player_name_options, index=0, key="p2_name_select")

    player1_id_manual_input = st.sidebar.text_input("Player 1 ID (if 'Enter Manually'):", value="104745") 
    player2_id_manual_input = st.sidebar.text_input("Player 2 ID (if 'Enter Manually'):", value="206421") 

    st.sidebar.subheader("Match Context")
    surface = st.sidebar.selectbox("Surface:", ["Hard", "Clay", "Grass", "Carpet", "Unknown"], index=0)
    tourney_level_options = {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP Tour 250/500", 
                             "F": "Tour Finals/Olympics", "D": "Davis Cup", "C": "Challengers", "Unknown": "Unknown"}
    tourney_level_display = st.sidebar.selectbox("Tournament Level:", list(tourney_level_options.values()), index=2)
    tourney_level = [k for k, v in tourney_level_options.items() if v == tourney_level_display][0]
    
    best_of_val = st.sidebar.selectbox("Best of (sets):", [3, 5], format_func=lambda x: f"{x} sets", index=0)
    
    round_options = {"F": "Final", "SF": "Semi-Final", "QF": "Quarter-Final", "R16": "Round of 16", 
                     "R32": "Round of 32", "R64": "Round of 64", "R128": "Round of 128", 
                     "RR": "Round Robin", "BR": "Bronze Medal", "Q1": "Qualifying R1", "Q2":"Qualifying R2", "Q3": "Qualifying R3", "Unknown": "Unknown"}
    round_display = st.sidebar.selectbox("Round:", list(round_options.values()), index=0)
    round_val = [k for k, v in round_options.items() if v == round_display][0]
    
    prediction_date = datetime.now() 

    if st.sidebar.button("🔮 Predict Match", use_container_width=True):
        p1_id_to_use, p2_id_to_use = None, None
        p1_name_for_display, p2_name_for_display = "Player 1", "Player 2"

        if player1_display_selected != "Enter Player ID Manually":
            p1_id_to_use = display_name_to_id_map.get(player1_display_selected)
            p1_name_for_display = player1_display_selected.split(" (ID:")[0] if p1_id_to_use else "Player 1 (Invalid Selection)"
        elif player1_id_manual_input:
            try: p1_id_to_use = int(player1_id_manual_input); p1_name_for_display = f"ID {p1_id_to_use}"
            except ValueError: st.error("Player 1 ID (manual) is invalid."); 
        
        if player2_display_selected != "Enter Player ID Manually":
            p2_id_to_use = display_name_to_id_map.get(player2_display_selected)
            p2_name_for_display = player2_display_selected.split(" (ID:")[0] if p2_id_to_use else "Player 2 (Invalid Selection)"
        elif player2_id_manual_input:
            try: p2_id_to_use = int(player2_id_manual_input); p2_name_for_display = f"ID {p2_id_to_use}"
            except ValueError: st.error("Player 2 ID (manual) is invalid."); 

        if p1_id_to_use is None or p2_id_to_use is None:
            st.error("Please provide valid Player 1 and Player 2 IDs.")
        elif p1_id_to_use == p2_id_to_use:
            st.error("Player 1 and Player 2 cannot be the same.")
        else:
            with st.spinner(f"Analyzing match: {p1_name_for_display} vs {p2_name_for_display}..."):
                p1_stats_live = get_player_latest_stats(p1_id_to_use, prediction_date, df_historical_matches, surface)
                p2_stats_live = get_player_latest_stats(p2_id_to_use, prediction_date, df_historical_matches, surface)
                h2h_live = get_h2h_stats(p1_id_to_use, p2_id_to_use, prediction_date, df_historical_matches)

                match_context_dict_live = {'surface': surface, 'tourney_level': tourney_level, 
                                           'best_of': int(best_of_val), 'round': round_val, # Use best_of_val
                                           'minutes': 120} 

                prob_p1, prob_p2 = make_prediction_for_app(p1_stats_live, p2_stats_live, 
                                                           match_context_dict_live, h2h_live)

                if prob_p1 is not None and prob_p2 is not None:
                    st.subheader("Match Prediction")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label=f"{p1_name_for_display} Win Probability", value=f"{prob_p1*100:.1f}%")
                    with col2:
                        st.metric(label=f"{p2_name_for_display} Win Probability", value=f"{prob_p2*100:.1f}%")

                    winner_name = p1_name_for_display if prob_p1 > prob_p2 else (p2_name_for_display if prob_p2 > prob_p1 else "Toss-up")
                    if abs(prob_p1 - prob_p2) < 1e-6 : # Effectively equal
                         st.info("**Prediction: Toss-up (50/50)!**")
                    else:
                         st.success(f"**Predicted Winner: {winner_name}**")
                    
                    with st.expander("View Input Stats (Simplified & Illustrative)"):
                        st.markdown(f"**{p1_name_for_display} (P1) Stats Used:**")
                        st.json({k: v for k,v in p1_stats_live.items() if k != 'id' and not (isinstance(v, float) and pd.isna(v))})
                        st.markdown(f"**{p2_name_for_display} (P2) Stats Used:**")
                        st.json({k: v for k,v in p2_stats_live.items() if k != 'id' and not (isinstance(v, float) and pd.isna(v))})
                        st.markdown(f"**H2H (P1 wins vs P2 wins prior):**")
                        st.json(h2h_live)
                        st.markdown(f"**Match Context Used:**")
                        st.json(match_context_dict_live)
                else:
                    st.error("Prediction could not be generated. Review earlier error messages in the UI or console.")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by a machine learning model. Live player stats are highly simplified for this demo.")