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
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR, 'tennis_predictor_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'tennis_data_scaler.pkl')
TRAINING_COLS_PATH = os.path.join(BASE_DIR, 'training_columns.json')
NUMERICAL_COLS_PATH = os.path.join(BASE_DIR, 'numerical_features_list.json')
CONFIG_PATH = os.path.join(BASE_DIR, 'model_config.json')

# !!! IMPORTANT: REPLACE THIS WITH THE ACTUAL PUBLIC URL TO YOUR all_matches_data.csv FILE !!!
DATA_URL = "https://drive.google.com/file/d/1440a80_iYxjMPqAzC9li4JcUsskT8NQo" 
# Example: DATA_URL = "https://your-bucket.s3.amazonaws.com/all_matches_data.csv"
# Example: DATA_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID" (ensure direct download)

# --- 1. Load Trained Model and Supporting Objects ---
@st.cache_resource 
def load_model_objects():
    loaded_objects = {"model": None, "scaler": None, "training_cols": None, 
                      "numerical_cols": None, "has_elo": False}
    all_files_present = True
    
    files_to_check_local = { # Files expected to be local with app.py
        "model": MODEL_PATH, "scaler": SCALER_PATH, "training_cols": TRAINING_COLS_PATH,
        "numerical_cols": NUMERICAL_COLS_PATH, "config": CONFIG_PATH
    }

    for name, path in files_to_check_local.items():
        if not os.path.exists(path):
            st.error(f"Critical local file not found: {os.path.basename(path)}. Please ensure it's in the app directory: {BASE_DIR}")
            all_files_present = False
            
    if not all_files_present:
        st.error("Application cannot start due to missing local model/configuration files.")
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
        
        # st.success("Model and supporting objects loaded successfully!")
        return loaded_objects
    except Exception as e:
        st.error(f"An unexpected error occurred while loading local model objects: {e}")
        return {"model": None, "scaler": None, "training_cols": None, "numerical_cols": None, "has_elo": False}

MODEL_OBJECTS = load_model_objects()
model = MODEL_OBJECTS["model"]
scaler = MODEL_OBJECTS["scaler"]
TRAINING_COLUMNS = MODEL_OBJECTS["training_cols"] # This is a list
NUMERICAL_FEATURES_TO_SCALE = MODEL_OBJECTS["numerical_cols"] # This is a list
HAS_ELO_FEATURES = MODEL_OBJECTS["has_elo"] # This is a boolean


# --- 2. Data Loading and Preparation for "Live" Stats ---
@st.cache_data # Cache the loaded historical data
def load_historical_data(data_url_param):
    if not data_url_param or not data_url_param.startswith("http") or DATA_URL == "YOUR_PUBLIC_URL_TO_ALL_MATCHES_DATA_CSV_HERE":
        st.error(f"Invalid or placeholder data URL: '{data_url_param}'. "
                 "Please update DATA_URL in app.py with a valid public URL to your all_matches_data.csv file.")
        return pd.DataFrame()

    try:
        st.info(f"Loading historical data from URL... This might take a moment.")
        df = pd.read_csv(data_url_param, low_memory=False) 
        
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
        
        df.dropna(subset=['winner_id', 'loser_id'], inplace=True) # Should be done after fillna+astype
        df = df[(df['winner_id'] != -1) & (df['loser_id'] != -1)] 

        # st.success("Historical data loaded and preprocessed from URL.")
        return df
    except Exception as e:
        st.error(f"Error loading or preprocessing historical data from URL: {e}")
        return pd.DataFrame()

df_historical_matches = load_historical_data(DATA_URL)


# --- 3. Helper Functions to Get Player Stats ---
def count_in_time_window_app(s_dates, window_str): 
    if s_dates.empty or s_dates.nunique() == 0: 
        return np.nan 
    df_temp_fatigue = pd.DataFrame({'date': s_dates.values, 'val': 1}, index=s_dates.index)
    df_temp_fatigue = df_temp_fatigue.reset_index().rename(columns={'index':'original_idx'}).set_index('date')
    df_temp_fatigue['rolled_count'] = df_temp_fatigue['val'].rolling(window=window_str, min_periods=1).count().shift(1)
    result_series = df_temp_fatigue.reset_index().set_index('original_idx')['rolled_count'].reindex(s_dates.index)
    if result_series.empty or result_series.isnull().all():
        return np.nan 
    return result_series.iloc[-1]

# @st.cache_data(ttl=900) # Consider caching for performance
def calculate_live_rolling_stats(player_id_int, prediction_date, surface_context, df_history_full):
    default_rolling_stats = {}
    if TRAINING_COLUMNS: 
        for col_template in TRAINING_COLUMNS:
            is_p1_roll_stat = col_template.startswith('p1_roll_') or \
                              col_template.startswith('p1_matches_last_') or \
                              col_template.startswith('p1_minutes_last_')
            if is_p1_roll_stat:
                actual_col_name = col_template.replace('p1_', '')
                default_rolling_stats[actual_col_name] = (0.5 if 'pct' in actual_col_name else 
                                                         (0.05 if 'rate' in actual_col_name else 0))
    if df_history_full.empty:
        return default_rolling_stats

    p_matches_winner = df_history_full[
        (df_history_full['winner_id'] == player_id_int) & 
        (df_history_full['tourney_date'] < prediction_date)
    ].copy()
    p_matches_winner['won_match'] = 1
    p_matches_winner['player_id'] = p_matches_winner['winner_id']
    
    p_matches_loser = df_history_full[
        (df_history_full['loser_id'] == player_id_int) & 
        (df_history_full['tourney_date'] < prediction_date)
    ].copy()
    p_matches_loser['won_match'] = 0
    p_matches_loser['player_id'] = p_matches_loser['loser_id']

    serve_return_cols_map_winner_app = {
        'w_ace': 'aces', 'w_df': 'dfs', 'w_svpt': 'svpt', 'w_1stIn': 'first_in', 
        'w_1stWon': 'first_won', 'w_2ndWon': 'second_won', 'w_SvGms': 'sv_gms', 
        'w_bpSaved': 'bp_saved', 'w_bpFaced': 'bp_faced'
    }
    serve_return_cols_map_loser_app = {
        'l_ace': 'aces', 'l_df': 'dfs', 'l_svpt': 'svpt', 'l_1stIn': 'first_in',
        'l_1stWon': 'first_won', 'l_2ndWon': 'second_won', 'l_SvGms': 'sv_gms',
        'l_bpSaved': 'bp_saved', 'l_bpFaced': 'bp_faced'
    }
    
    cols_to_keep_winner = ['tourney_date', 'surface', 'minutes', 'player_id', 'won_match', 'loser_id']
    rename_map_winner = {'loser_id': 'opponent_id'}
    for orig_col, new_name in serve_return_cols_map_winner_app.items():
        if orig_col in p_matches_winner.columns:
            cols_to_keep_winner.append(orig_col); rename_map_winner[orig_col] = new_name
    
    cols_to_keep_loser = ['tourney_date', 'surface', 'minutes', 'player_id', 'won_match', 'winner_id']
    rename_map_loser = {'winner_id': 'opponent_id'}
    for orig_col, new_name in serve_return_cols_map_loser_app.items():
        if orig_col in p_matches_loser.columns:
            cols_to_keep_loser.append(orig_col); rename_map_loser[orig_col] = new_name

    p_matches_winner = p_matches_winner[[col for col in cols_to_keep_winner if col in p_matches_winner.columns]].copy() # Use .copy()
    p_matches_loser = p_matches_loser[[col for col in cols_to_keep_loser if col in p_matches_loser.columns]].copy() # Use .copy()

    p_matches_winner.rename(columns=rename_map_winner, inplace=True)
    p_matches_loser.rename(columns=rename_map_loser, inplace=True)

    player_timeline_live = pd.concat([p_matches_winner, p_matches_loser], ignore_index=True)
    if player_timeline_live.empty:
        return default_rolling_stats

    player_timeline_live.sort_values(by=['tourney_date', 'minutes'], inplace=True)
    player_timeline_live.drop_duplicates(subset=['tourney_date', 'opponent_id'], keep='last', inplace=True)
    player_timeline_live.reset_index(drop=True, inplace=True)

    stat_cols_to_impute = ['minutes', 'aces', 'dfs', 'svpt', 'first_in', 'first_won', 'second_won', 
                           'sv_gms', 'bp_saved', 'bp_faced'] 
    for col in stat_cols_to_impute:
        if col in player_timeline_live.columns:
            player_timeline_live[col] = pd.to_numeric(player_timeline_live[col], errors='coerce').fillna(0)
        else: player_timeline_live[col] = 0

    live_rolling_stats = default_rolling_stats.copy() 

    if not player_timeline_live.empty:
        live_rolling_stats['roll_overall_win_pct_10'] = player_timeline_live['won_match'].rolling(window=10, min_periods=3).mean().shift(1).iloc[-1] if len(player_timeline_live) >= 1 else 0.5
        live_rolling_stats['roll_overall_win_pct_20'] = player_timeline_live['won_match'].rolling(window=20, min_periods=5).mean().shift(1).iloc[-1] if len(player_timeline_live) >= 1 else 0.5
        
        surface_matches = player_timeline_live[player_timeline_live['surface'] == surface_context]
        live_rolling_stats['roll_surface_win_pct_5'] = surface_matches['won_match'].rolling(window=5, min_periods=2).mean().shift(1).iloc[-1] if len(surface_matches) >=1 else 0.5
        live_rolling_stats['roll_surface_win_pct_10'] = surface_matches['won_match'].rolling(window=10, min_periods=3).mean().shift(1).iloc[-1] if len(surface_matches) >= 1 else 0.5

        live_rolling_stats['matches_last_7d'] = count_in_time_window_app(player_timeline_live['tourney_date'], '7D')
        live_rolling_stats['matches_last_30d'] = count_in_time_window_app(player_timeline_live['tourney_date'], '30D')
        
        if 'minutes' in player_timeline_live.columns:
            live_rolling_stats['minutes_last_5_matches'] = player_timeline_live['minutes'].rolling(window=5, min_periods=1).sum().shift(1).iloc[-1] if len(player_timeline_live) >=1 else 0
            live_rolling_stats['minutes_last_10_matches'] = player_timeline_live['minutes'].rolling(window=10, min_periods=1).sum().shift(1).iloc[-1] if len(player_timeline_live) >=1 else 0

        if 'aces' in player_timeline_live.columns and 'svpt' in player_timeline_live.columns:
            player_timeline_live['ace_rate_match'] = player_timeline_live['aces'] / (player_timeline_live['svpt'] + 1e-6)
            live_rolling_stats['roll_ace_rate_5'] = player_timeline_live['ace_rate_match'].rolling(window=5, min_periods=2).mean().shift(1).iloc[-1] if len(player_timeline_live) >=1 else 0.05
        
        if 'dfs' in player_timeline_live.columns and 'svpt' in player_timeline_live.columns:
            player_timeline_live['df_rate_match'] = player_timeline_live['dfs'] / (player_timeline_live['svpt'] + 1e-6)
            live_rolling_stats['roll_df_rate_5'] = player_timeline_live['df_rate_match'].rolling(window=5, min_periods=2).mean().shift(1).iloc[-1] if len(player_timeline_live) >=1 else 0.05

        if 'first_in' in player_timeline_live.columns and 'svpt' in player_timeline_live.columns:
            valid_svpt_mask = player_timeline_live['svpt'] > 0
            player_timeline_live.loc[valid_svpt_mask, 'first_serve_in_pct_match'] = \
                player_timeline_live.loc[valid_svpt_mask, 'first_in'] / player_timeline_live.loc[valid_svpt_mask, 'svpt']
            player_timeline_live['first_serve_in_pct_match'].fillna(0.60, inplace=True)
            live_rolling_stats['roll_first_serve_in_pct_5'] = player_timeline_live['first_serve_in_pct_match'].rolling(window=5, min_periods=2).mean().shift(1).iloc[-1] if len(player_timeline_live) >=1 else 0.60

        if 'first_won' in player_timeline_live.columns and 'first_in' in player_timeline_live.columns:
            valid_first_in_mask = player_timeline_live['first_in'] > 0
            player_timeline_live.loc[valid_first_in_mask, 'first_serve_won_pct_match'] = \
                player_timeline_live.loc[valid_first_in_mask, 'first_won'] / player_timeline_live.loc[valid_first_in_mask, 'first_in']
            player_timeline_live['first_serve_won_pct_match'].fillna(0.70, inplace=True)
            live_rolling_stats['roll_first_serve_won_pct_5'] = player_timeline_live['first_serve_won_pct_match'].rolling(window=5, min_periods=2).mean().shift(1).iloc[-1] if len(player_timeline_live) >=1 else 0.70
            
        if 'second_won' in player_timeline_live.columns and 'svpt' in player_timeline_live.columns and 'first_in' in player_timeline_live.columns:
            player_timeline_live['second_serve_pts_played'] = (player_timeline_live['svpt'] - player_timeline_live['first_in']).clip(lower=0)
            valid_second_serve_mask = player_timeline_live['second_serve_pts_played'] > 0
            player_timeline_live.loc[valid_second_serve_mask, 'second_serve_won_pct_match'] = \
                player_timeline_live.loc[valid_second_serve_mask, 'second_won'] / player_timeline_live.loc[valid_second_serve_mask, 'second_serve_pts_played']
            player_timeline_live['second_serve_won_pct_match'].fillna(0.50, inplace=True)
            live_rolling_stats['roll_second_serve_won_pct_5'] = player_timeline_live['second_serve_won_pct_match'].rolling(window=5, min_periods=2).mean().shift(1).iloc[-1] if len(player_timeline_live) >=1 else 0.50
    
    for key in default_rolling_stats.keys():
        if key not in live_rolling_stats or pd.isna(live_rolling_stats.get(key)):
            live_rolling_stats[key] = default_rolling_stats[key]
            
    return live_rolling_stats

def get_player_latest_stats(player_id_int, prediction_date, df_history_full, surface_context):
    base_stats = {'id': player_id_int, 'elo': 1500 if HAS_ELO_FEATURES else np.nan, 'rank': 1000, 
                  'rank_points': 0, 'ht': 180, 'age': 25, 'hand': 'U'}

    live_rolling_stats = calculate_live_rolling_stats(player_id_int, prediction_date, surface_context, df_history_full)
    base_stats.update(live_rolling_stats) 

    if df_history_full.empty:
        return base_stats

    player_matches_for_base_stats = df_history_full[
        ((df_history_full['winner_id'] == player_id_int) | (df_history_full['loser_id'] == player_id_int)) &
        (df_history_full['tourney_date'] < prediction_date) 
    ].sort_values(by='tourney_date', ascending=False)

    if not player_matches_for_base_stats.empty:
        latest_match_played = player_matches_for_base_stats.iloc[0]
        is_winner_latest = latest_match_played['winner_id'] == player_id_int
        
        if HAS_ELO_FEATURES and f"{'winner'if is_winner_latest else 'loser'}_elo" in latest_match_played and pd.notna(latest_match_played[f"{'winner'if is_winner_latest else 'loser'}_elo"]):
            base_stats['elo'] = latest_match_played[f"{'winner'if is_winner_latest else 'loser'}_elo"]
        
        for stat_suffix in ['rank', 'rank_points', 'ht', 'age', 'hand']:
            col_name = f"{'winner' if is_winner_latest else 'loser'}_{stat_suffix}"
            if col_name in latest_match_played and pd.notna(latest_match_played[col_name]):
                base_stats[stat_suffix] = latest_match_played[col_name]
    
    final_player_stats = base_stats.copy()
    for k, v in final_player_stats.items():
        if pd.isna(v):
            if 'elo' in k and HAS_ELO_FEATURES: final_player_stats[k] = 1500
            elif 'elo' in k and not HAS_ELO_FEATURES: final_player_stats[k] = 0 
            elif 'rank' in k and 'points' not in k : final_player_stats[k] = 1000
            elif 'points' in k : final_player_stats[k] = 0
            elif 'ht' in k : final_player_stats[k] = 180
            elif 'age' in k : final_player_stats[k] = 25
            elif 'hand' in k : final_player_stats[k] = 'U'
            elif 'pct' in k : final_player_stats[k] = 0.5 
            elif 'rate' in k : final_player_stats[k] = 0.05 
            else: final_player_stats[k] = 0
            
    return final_player_stats

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
    
    for diff_col_name in TRAINING_COLUMNS: # Ensure TRAINING_COLUMNS is not None
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
        is_dummy_or_interaction = any(col.startswith(prefix) for prefix in ['surface_', 'tourney_level_', 'best_of_', 'round_', 'p1_hand_', 'p2_hand_']) or \
                                  col in ['p1_lefty_vs_p2_righty', 'p1_righty_vs_p2_lefty', 'both_lefty', 'both_righty']
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
    
    for interaction_col in ['p1_lefty_vs_p2_righty', 'p1_righty_vs_p2_lefty', 'both_lefty', 'both_righty']:
        if interaction_col in TRAINING_COLUMNS:
            is_interaction_true = False
            if interaction_col == 'p1_lefty_vs_p2_righty' and p1_hand_val == 'L' and p2_hand_val == 'R': is_interaction_true = True
            if interaction_col == 'p1_righty_vs_p2_lefty' and p1_hand_val == 'R' and p2_hand_val == 'L': is_interaction_true = True
            if interaction_col == 'both_lefty' and p1_hand_val == 'L' and p2_hand_val == 'L': is_interaction_true = True
            if interaction_col == 'both_righty' and p1_hand_val == 'R' and p2_hand_val == 'R': is_interaction_true = True
            feature_vector[interaction_col] = 1 if is_interaction_true else 0
        elif interaction_col not in feature_vector: 
             feature_vector[interaction_col] = 0

    input_df_row = pd.Series(feature_vector, dtype=object).reindex(TRAINING_COLUMNS)
    input_df = pd.DataFrame([input_df_row], columns=TRAINING_COLUMNS)

    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='ignore')
        if input_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(input_df[col].dtype):
                input_df[col].fillna(0, inplace=True)
            else: input_df[col].fillna(0, inplace=True)
        
        is_binary_col = any(col.startswith(prefix) for prefix in ['surface_', 'tourney_level_', 'best_of_', 'round_', 'p1_hand_', 'p2_hand_']) or \
                        col in ['p1_lefty_vs_p2_righty', 'p1_righty_vs_p2_lefty', 'both_lefty', 'both_righty']
        
        if is_binary_col:
            try: input_df[col] = input_df[col].round().astype(int) 
            except (ValueError, TypeError):
                input_df[col] = 0

    input_df_scaled = input_df.copy()
    if NUMERICAL_FEATURES_TO_SCALE is None: # Check if it was loaded
        st.error("List of numerical features to scale was not loaded. Cannot proceed.")
        return None, None
        
    cols_to_scale_present = [col for col in NUMERICAL_FEATURES_TO_SCALE if col in input_df_scaled.columns]
    if cols_to_scale_present:
        try:
            for col_to_scale in cols_to_scale_present:
                if not pd.api.types.is_numeric_dtype(input_df_scaled[col_to_scale].dtype):
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
        # For debugging:
        # st.write("Data types of features passed to model for prediction:")
        # st.dataframe(input_df_scaled.dtypes.rename("dtype").to_frame().T)
        # st.write("First row of scaled data passed to model for prediction:")
        # st.dataframe(input_df_scaled.head(1))
        return None, None

# --- 5. Streamlit UI ---
st.title("🎾 Ultimate Tennis Match Predictor")

if model is None or df_historical_matches.empty or TRAINING_COLUMNS is None:
    st.error("Application initialization failed. Essential components are missing.")
else:
    st.sidebar.header("Match Input")
    
    player_name_options = ["Enter Player ID Manually"]
    display_name_to_id_map = {}
    if not df_historical_matches.empty:
        ids_names1 = df_historical_matches[['winner_id', 'winner_name']].copy().rename(columns={'winner_id':'id', 'winner_name':'name'})
        ids_names2 = df_historical_matches[['loser_id', 'loser_name']].copy().rename(columns={'loser_id':'id', 'loser_name':'name'})
        if not ids_names1.empty or not ids_names2.empty:
            unique_players_df = pd.concat([ids_names1, ids_names2]).drop_duplicates(subset=['id']).dropna(subset=['id','name'])
            unique_players_df['id'] = unique_players_df['id'].astype(int) # Ensure ID is int
            unique_players_df = unique_players_df[unique_players_df['id'] != -1] # Exclude placeholder IDs
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
    surface_options = ["Hard", "Clay", "Grass", "Carpet", "Unknown"]
    surface = st.sidebar.selectbox("Surface:", surface_options, index=0)
    
    tourney_level_map = {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP Tour 250/500", 
                         "F": "Tour Finals/Olympics", "D": "Davis Cup", "C": "Challengers", "Unknown": "Unknown"}
    tourney_level_display = st.sidebar.selectbox("Tournament Level:", list(tourney_level_map.values()), index=2)
    tourney_level = [k for k, v in tourney_level_map.items() if v == tourney_level_display][0]
    
    best_of_val = st.sidebar.selectbox("Best of (sets):", [3, 5], format_func=lambda x: f"{x} sets", index=0)
    
    round_map = {"F": "Final", "SF": "Semi-Final", "QF": "Quarter-Final", "R16": "Round of 16", 
                 "R32": "Round of 32", "R64": "Round of 64", "R128": "Round of 128", 
                 "RR": "Round Robin", "BR": "Bronze Medal", "Q1": "Qualifying R1", 
                 "Q2":"Qualifying R2", "Q3": "Qualifying R3", "Unknown": "Unknown"}
    round_display = st.sidebar.selectbox("Round:", list(round_map.values()), index=0)
    round_val = [k for k, v in round_map.items() if v == round_display][0]
    
    prediction_date = datetime.now() 

    if st.sidebar.button("🔮 Predict Match", use_container_width=True):
        p1_id_to_use, p2_id_to_use = None, None
        p1_name_for_display, p2_name_for_display = "Player 1", "Player 2"

        if player1_display_selected != "Enter Player ID Manually":
            p1_id_to_use = display_name_to_id_map.get(player1_display_selected)
            p1_name_for_display = player1_display_selected.split(" (ID:")[0] if p1_id_to_use is not None else "P1 (Invalid Sel.)"
        elif player1_id_manual_input:
            try: p1_id_to_use = int(player1_id_manual_input); p1_name_for_display = f"ID {p1_id_to_use}"
            except ValueError: st.error("Player 1 ID (manual) is invalid."); 
        
        if player2_display_selected != "Enter Player ID Manually":
            p2_id_to_use = display_name_to_id_map.get(player2_display_selected)
            p2_name_for_display = player2_display_selected.split(" (ID:")[0] if p2_id_to_use is not None else "P2 (Invalid Sel.)"
        elif player2_id_manual_input:
            try: p2_id_to_use = int(player2_id_manual_input); p2_name_for_display = f"ID {p2_id_to_use}"
            except ValueError: st.error("Player 2 ID (manual) is invalid."); 

        if p1_id_to_use is None or p2_id_to_use is None:
            st.error("Please provide valid Player 1 and Player 2 identifiers.")
        elif p1_id_to_use == p2_id_to_use:
            st.error("Player 1 and Player 2 cannot be the same.")
        else:
            with st.spinner(f"Analyzing match: {p1_name_for_display} vs {p2_name_for_display}..."):
                p1_stats_live = get_player_latest_stats(p1_id_to_use, prediction_date, df_historical_matches, surface)
                p2_stats_live = get_player_latest_stats(p2_id_to_use, prediction_date, df_historical_matches, surface)
                h2h_live = get_h2h_stats(p1_id_to_use, p2_id_to_use, prediction_date, df_historical_matches)

                match_context_dict_live = {'surface': surface, 'tourney_level': tourney_level, 
                                           'best_of': int(best_of_val), 'round': round_val, 
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
                    if abs(prob_p1 - prob_p2) < 1e-6 : 
                         st.info("**Prediction: Toss-up (50/50)!**")
                    else:
                         st.success(f"**Predicted Winner: {winner_name}**")
                    
                    with st.expander("View Input Stats (Simplified & Illustrative)"):
                        st.markdown(f"**{p1_name_for_display} (P1) Stats Used:**")
                        st.json({k: (f"{v:.3f}" if isinstance(v, float) else v) for k,v in p1_stats_live.items() if k != 'id' and not (isinstance(v, float) and pd.isna(v))})
                        st.markdown(f"**{p2_name_for_display} (P2) Stats Used:**")
                        st.json({k: (f"{v:.3f}" if isinstance(v, float) else v) for k,v in p2_stats_live.items() if k != 'id' and not (isinstance(v, float) and pd.isna(v))})
                        st.markdown(f"**H2H (P1 wins vs P2 wins prior):**")
                        st.json(h2h_live)
                        st.markdown(f"**Match Context Used:**")
                        st.json(match_context_dict_live)
                else:
                    st.error("Prediction could not be generated. Review earlier error messages in the UI or console.")

st.sidebar.markdown("---")
st.sidebar.caption("Akrams Pengar Maskin")