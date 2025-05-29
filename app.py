import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
import os 
import gdown # For Google Drive downloads
# from io import StringIO # Not strictly needed if gdown writes to file then pd reads file

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

# !!! IMPORTANT: REPLACE THIS WITH YOUR ACTUAL GOOGLE DRIVE FILE ID !!!
GOOGLE_DRIVE_FILE_ID = "1440a80_iYxjMPqAzC9li4JcUsskT8NQo" # Example ID from your logs

# --- 1. Load Trained Model and Supporting Objects ---
@st.cache_resource 
def load_model_objects():
    # st.write("DEBUG: Attempting to load model objects...") # Usually too early for st.write
    loaded_objects = {"model": None, "scaler": None, "training_cols": None, 
                      "numerical_cols": None, "has_elo": False}
    all_files_present = True
    files_to_check_local = {
        "model": MODEL_PATH, "scaler": SCALER_PATH, "training_cols": TRAINING_COLS_PATH,
        "numerical_cols": NUMERICAL_COLS_PATH, "config": CONFIG_PATH
    }
    for name, path in files_to_check_local.items():
        if not os.path.exists(path):
            # st.error(f"Critical file not found: {os.path.basename(path)}. App directory: {BASE_DIR}") # Can't use st.error before app runs
            print(f"ERROR (load_model_objects): Critical file not found: {os.path.basename(path)} at {path}")
            all_files_present = False
    if not all_files_present:
        print("ERROR (load_model_objects): Application cannot start due to missing model/configuration files.")
        return loaded_objects 
    try:
        with open(MODEL_PATH, 'rb') as f: loaded_objects["model"] = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: loaded_objects["scaler"] = pickle.load(f)
        with open(TRAINING_COLS_PATH, 'r') as f: loaded_objects["training_cols"] = json.load(f)
        with open(NUMERICAL_COLS_PATH, 'r') as f: loaded_objects["numerical_cols"] = json.load(f)
        with open(CONFIG_PATH, 'r') as f: config = json.load(f)
        loaded_objects["has_elo"] = config.get('has_elo', False)
        print("INFO (load_model_objects): Model and supporting objects loaded successfully.")
        return loaded_objects
    except Exception as e:
        print(f"ERROR (load_model_objects): An unexpected error occurred while loading model objects: {e}")
        return {"model": None, "scaler": None, "training_cols": None, "numerical_cols": None, "has_elo": False}

MODEL_OBJECTS = load_model_objects()
model = MODEL_OBJECTS["model"]
scaler = MODEL_OBJECTS["scaler"]
TRAINING_COLUMNS = MODEL_OBJECTS["training_cols"]
NUMERICAL_FEATURES_TO_SCALE = MODEL_OBJECTS["numerical_cols"]
HAS_ELO_FEATURES = MODEL_OBJECTS["has_elo"]

# --- 2. Data Loading from Google Drive ---
@st.cache_data(show_spinner="Downloading historical match data (this may take a while)...")
def load_historical_data(file_id_param): # Renamed param to avoid conflict with global
    if not file_id_param or file_id_param == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        st.error("Google Drive File ID for historical data is not configured in app.py.")
        return pd.DataFrame()

    output_filename = "temp_downloaded_all_matches_data.csv" 
    df = pd.DataFrame() 

    try:
        # st.write(f"DEBUG (load_historical_data): Attempting gdown with File ID: {file_id_param}")
        gdown.download(id=file_id_param, output=output_filename, quiet=True, fuzzy=True)
        
        if not os.path.exists(output_filename) or os.path.getsize(output_filename) < 1000:
            st.error(f"`gdown` completed but output file '{output_filename}' is missing or too small.")
            return pd.DataFrame()
        
        df = pd.read_csv(output_filename, low_memory=False)
        # st.write(f"DEBUG (load_historical_data): Loaded from temp file. Shape: {df.shape}")
        
        if 'tourney_date' not in df.columns:
            st.error("'tourney_date' column missing from historical data. Check CSV integrity.")
            st.write("DEBUG (load_historical_data): Columns found:", df.columns.tolist())
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
        
        df.dropna(subset=['winner_id', 'loser_id'], inplace=True)
        df = df[(df['winner_id'] != -1) & (df['loser_id'] != -1)].copy() # Use .copy() to avoid SettingWithCopyWarning

        return df
    except Exception as e:
        st.error(f"Error during gdown download or pandas parsing in load_historical_data: {e}")
        return pd.DataFrame()
    finally: 
        if os.path.exists(output_filename):
            try: os.remove(output_filename)
            except Exception as e_remove: st.warning(f"Could not remove temp file '{output_filename}': {e_remove}")
            
df_historical_matches = load_historical_data(GOOGLE_DRIVE_FILE_ID)

# --- 3. Helper Functions to Get Player Stats ---
def count_in_time_window_app(s_dates, window_str): 
    if s_dates.empty or s_dates.nunique() == 0: return np.nan 
    df_temp_fatigue = pd.DataFrame({'date': s_dates.values, 'val': 1}, index=s_dates.index)
    df_temp_fatigue = df_temp_fatigue.reset_index().rename(columns={'index':'original_idx'}).set_index('date')
    df_temp_fatigue['rolled_count'] = df_temp_fatigue['val'].rolling(window=window_str, min_periods=1).count().shift(1)
    result_series = df_temp_fatigue.reset_index().set_index('original_idx')['rolled_count'].reindex(s_dates.index)
    if result_series.empty or result_series.isnull().all(): return np.nan 
    return result_series.iloc[-1]

# @st.cache_data(ttl=900, show_spinner="Calculating rolling stats for player...") # Cache this if it becomes a bottleneck
def calculate_live_rolling_stats(player_id_int, prediction_date, surface_context, df_history_full_param): # Renamed param
    # Prepare a dictionary for default rolling stats based on TRAINING_COLUMNS
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
    if df_history_full_param.empty:
        # st.write(f"DEBUG (calculate_live_rolling_stats for {player_id_int}): df_history_full_param is empty, returning defaults.")
        return default_rolling_stats

    # Filter matches for the specific player BEFORE prediction_date
    p_matches_winner = df_history_full_param[(df_history_full_param['winner_id'] == player_id_int) & (df_history_full_param['tourney_date'] < prediction_date)].copy()
    if not p_matches_winner.empty:
        p_matches_winner['won_match'] = 1
        p_matches_winner['player_id'] = p_matches_winner['winner_id']
    
    p_matches_loser = df_history_full_param[(df_history_full_param['loser_id'] == player_id_int) & (df_history_full_param['tourney_date'] < prediction_date)].copy()
    if not p_matches_loser.empty:
        p_matches_loser['won_match'] = 0
        p_matches_loser['player_id'] = p_matches_loser['loser_id']

    serve_return_cols_map_winner_app = {'w_ace': 'aces', 'w_df': 'dfs', 'w_svpt': 'svpt', 'w_1stIn': 'first_in', 'w_1stWon': 'first_won', 'w_2ndWon': 'second_won', 'w_SvGms': 'sv_gms', 'w_bpSaved': 'bp_saved', 'w_bpFaced': 'bp_faced'}
    serve_return_cols_map_loser_app = {'l_ace': 'aces', 'l_df': 'dfs', 'l_svpt': 'svpt', 'l_1stIn': 'first_in', 'l_1stWon': 'first_won', 'l_2ndWon': 'second_won', 'l_SvGms': 'sv_gms', 'l_bpSaved': 'bp_saved', 'l_bpFaced': 'bp_faced'}
    
    cols_to_keep_winner = ['tourney_date', 'surface', 'minutes', 'player_id', 'won_match', 'loser_id']
    rename_map_winner = {'loser_id': 'opponent_id'}
    for orig_col, new_name in serve_return_cols_map_winner_app.items():
        if orig_col in p_matches_winner.columns: cols_to_keep_winner.append(orig_col); rename_map_winner[orig_col] = new_name
    
    cols_to_keep_loser = ['tourney_date', 'surface', 'minutes', 'player_id', 'won_match', 'winner_id']
    rename_map_loser = {'winner_id': 'opponent_id'}
    for orig_col, new_name in serve_return_cols_map_loser_app.items():
        if orig_col in p_matches_loser.columns: cols_to_keep_loser.append(orig_col); rename_map_loser[orig_col] = new_name

    # Ensure DataFrames are not empty before selecting columns
    if not p_matches_winner.empty:
        p_matches_winner = p_matches_winner[[col for col in cols_to_keep_winner if col in p_matches_winner.columns]].copy()
        p_matches_winner.rename(columns=rename_map_winner, inplace=True)
    if not p_matches_loser.empty:
        p_matches_loser = p_matches_loser[[col for col in cols_to_keep_loser if col in p_matches_loser.columns]].copy()
        p_matches_loser.rename(columns=rename_map_loser, inplace=True)

    if p_matches_winner.empty and p_matches_loser.empty:
        # st.write(f"DEBUG (calculate_live_rolling_stats for {player_id_int}): No historical matches found before prediction date. Returning defaults.")
        return default_rolling_stats
        
    player_timeline_live = pd.concat([p_matches_winner, p_matches_loser], ignore_index=True)
    if player_timeline_live.empty: return default_rolling_stats

    player_timeline_live.sort_values(by=['tourney_date', 'minutes'], inplace=True) # Sort by date, then by minutes if multiple matches on a day
    # opponent_id might be float if one of the DFs was empty and concat introduced NaNs, ensure int
    if 'opponent_id' in player_timeline_live.columns:
        player_timeline_live['opponent_id'] = pd.to_numeric(player_timeline_live['opponent_id'], errors='coerce').fillna(-1).astype(int)
    player_timeline_live.drop_duplicates(subset=['tourney_date', 'opponent_id'], keep='last', inplace=True)
    player_timeline_live.reset_index(drop=True, inplace=True)

    stat_cols_to_impute = ['minutes', 'aces', 'dfs', 'svpt', 'first_in', 'first_won', 'second_won', 'sv_gms', 'bp_saved', 'bp_faced'] 
    for col in stat_cols_to_impute:
        if col in player_timeline_live.columns: player_timeline_live[col] = pd.to_numeric(player_timeline_live[col], errors='coerce').fillna(0)
        else: player_timeline_live[col] = 0 # Ensure column exists

    live_rolling_stats = default_rolling_stats.copy() 
    if not player_timeline_live.empty: # Check again after potential drops
        # Win Percentages
        for stat, window, min_p in [('roll_overall_win_pct_10', 10, 3), ('roll_overall_win_pct_20', 20, 5)]:
            if len(player_timeline_live) >= min_p : # Check if enough data for min_periods
                live_rolling_stats[stat] = player_timeline_live['won_match'].rolling(window=window, min_periods=min_p).mean().shift(1).iloc[-1]
            else: live_rolling_stats[stat] = 0.5 # Fallback to default
        
        surface_matches = player_timeline_live[player_timeline_live['surface'] == surface_context]
        for stat, window, min_p in [('roll_surface_win_pct_5', 5, 2), ('roll_surface_win_pct_10', 10, 3)]:
            if len(surface_matches) >= min_p:
                live_rolling_stats[stat] = surface_matches['won_match'].rolling(window=window, min_periods=min_p).mean().shift(1).iloc[-1]
            else: live_rolling_stats[stat] = 0.5

        # Fatigue
        if 'tourney_date' in player_timeline_live.columns and not player_timeline_live.empty:
            live_rolling_stats['matches_last_7d'] = count_in_time_window_app(player_timeline_live['tourney_date'], '7D')
            live_rolling_stats['matches_last_30d'] = count_in_time_window_app(player_timeline_live['tourney_date'], '30D')
        
        if 'minutes' in player_timeline_live.columns and not player_timeline_live.empty:
            for stat, window in [('minutes_last_5_matches', 5), ('minutes_last_10_matches', 10)]:
                if len(player_timeline_live) >=1: live_rolling_stats[stat] = player_timeline_live['minutes'].rolling(window=window, min_periods=1).sum().shift(1).iloc[-1]
                else: live_rolling_stats[stat] = 0
        
        # Serve/Return % stats
        for base_col, denom_col, rate_name, default_val in [
            ('aces', 'svpt', 'roll_ace_rate_5', 0.05), ('dfs', 'svpt', 'roll_df_rate_5', 0.05),
            ('first_in', 'svpt', 'roll_first_serve_in_pct_5', 0.60), ('first_won', 'first_in', 'roll_first_serve_won_pct_5', 0.70)]:
            if base_col in player_timeline_live.columns and denom_col in player_timeline_live.columns and not player_timeline_live.empty:
                # Create per-match rate only if denominator is valid
                temp_rate_col = f'{base_col}_rate_match_temp' # Avoid permanent change
                player_timeline_live[temp_rate_col] = np.nan # Initialize
                valid_denom_mask = player_timeline_live[denom_col] > 0
                player_timeline_live.loc[valid_denom_mask, temp_rate_col] = \
                    player_timeline_live.loc[valid_denom_mask, base_col] / player_timeline_live.loc[valid_denom_mask, denom_col]
                player_timeline_live[temp_rate_col].fillna(default_val, inplace=True) # Fill NaNs from division by zero or missing data
                if len(player_timeline_live) >= 1:
                     live_rolling_stats[rate_name] = player_timeline_live[temp_rate_col].rolling(window=5, min_periods=2).mean().shift(1).iloc[-1]
                else: live_rolling_stats[rate_name] = default_val
            else: live_rolling_stats[rate_name] = default_val
        
        if 'second_won' in player_timeline_live.columns and 'svpt' in player_timeline_live.columns and \
           'first_in' in player_timeline_live.columns and not player_timeline_live.empty:
            player_timeline_live['second_serve_pts_played'] = (player_timeline_live['svpt'] - player_timeline_live['first_in']).clip(lower=0)
            temp_ssp_rate_col = 'second_serve_won_pct_match_temp'
            player_timeline_live[temp_ssp_rate_col] = np.nan
            valid_ssp_mask = player_timeline_live['second_serve_pts_played'] > 0
            player_timeline_live.loc[valid_ssp_mask, temp_ssp_rate_col] = \
                player_timeline_live.loc[valid_ssp_mask, 'second_won'] / player_timeline_live.loc[valid_ssp_mask, 'second_serve_pts_played']
            player_timeline_live[temp_ssp_rate_col].fillna(0.50, inplace=True)
            if len(player_timeline_live) >= 1:
                live_rolling_stats['roll_second_serve_won_pct_5'] = player_timeline_live[temp_ssp_rate_col].rolling(window=5, min_periods=2).mean().shift(1).iloc[-1]
            else: live_rolling_stats['roll_second_serve_won_pct_5'] = 0.50
        else: live_rolling_stats['roll_second_serve_won_pct_5'] = 0.50
    
    # Final fill for any rolling stat that might still be NaN (e.g. if iloc[-1] on empty series)
    for key in default_rolling_stats.keys():
        if key not in live_rolling_stats or pd.isna(live_rolling_stats.get(key)):
            live_rolling_stats[key] = default_rolling_stats[key]
    return live_rolling_stats

def get_player_latest_stats(player_id_int, prediction_date, df_history_full, surface_context):
    base_stats = {'id': player_id_int, 'elo': 1500 if HAS_ELO_FEATURES else np.nan, 'rank': 1000, 
                  'rank_points': 0, 'ht': 180, 'age': 25, 'hand': 'U'}
    live_rolling_stats = calculate_live_rolling_stats(player_id_int, prediction_date, surface_context, df_history_full)
    base_stats.update(live_rolling_stats) 

    if df_history_full.empty: return base_stats

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
            elif 'age' in k : final_player_stats[k] = 25 # This should be calculated from birth year for live app
            elif 'hand' in k : final_player_stats[k] = 'U'
            elif 'pct' in k : final_player_stats[k] = 0.5 
            elif 'rate' in k : final_player_stats[k] = 0.05 
            else: final_player_stats[k] = 0
    return final_player_stats

def get_h2h_stats(player1_id_int, player2_id_int, prediction_date, df_history_full):
    if df_history_full.empty: return {'p1_h2h_wins': 0, 'p2_h2h_wins': 0}
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
            p1_col_name = f'p1_{base_feature_name}'; p2_col_name = f'p2_{base_feature_name}'
            val1 = pd.to_numeric(feature_vector.get(p1_col_name), errors='coerce')
            val2 = pd.to_numeric(feature_vector.get(p2_col_name), errors='coerce')
            feature_vector[diff_col_name] = val1 - val2 if pd.notna(val1) and pd.notna(val2) else np.nan 

    for col in TRAINING_COLUMNS:
        is_dummy_or_interaction = any(col.startswith(prefix) for prefix in ['surface_', 'tourney_level_', 'best_of_', 'round_', 'p1_hand_', 'p2_hand_']) or \
                                  col in ['p1_lefty_vs_p2_righty', 'p1_righty_vs_p2_lefty', 'both_lefty', 'both_righty']
        if is_dummy_or_interaction and col not in feature_vector: feature_vector[col] = 0

    for context_key, prefix in [('surface','surface_'), ('tourney_level','tourney_level_'), ('best_of','best_of_'), ('round','round_')]:
        # Ensure context_key exists in match_context_live before using it
        if context_key in match_context_live and f"{prefix}{match_context_live[context_key]}" in TRAINING_COLUMNS:
            feature_vector[f"{prefix}{match_context_live[context_key]}"] = 1
            
    p1_hand_val = p1_live_stats.get('hand', 'U'); p2_hand_val = p2_live_stats.get('hand', 'U')
    if f'p1_hand_{p1_hand_val}' in TRAINING_COLUMNS: feature_vector[f'p1_hand_{p1_hand_val}'] = 1
    if f'p2_hand_{p2_hand_val}' in TRAINING_COLUMNS: feature_vector[f'p2_hand_{p2_hand_val}'] = 1
    
    for interaction_col in ['p1_lefty_vs_p2_righty', 'p1_righty_vs_p2_lefty', 'both_lefty', 'both_righty']:
        if interaction_col in TRAINING_COLUMNS: # Only create if it was a feature during training
            is_true = (interaction_col == 'p1_lefty_vs_p2_righty' and p1_hand_val == 'L' and p2_hand_val == 'R') or \
                      (interaction_col == 'p1_righty_vs_p2_lefty' and p1_hand_val == 'R' and p2_hand_val == 'L') or \
                      (interaction_col == 'both_lefty' and p1_hand_val == 'L' and p2_hand_val == 'L') or \
                      (interaction_col == 'both_righty' and p1_hand_val == 'R' and p2_hand_val == 'R')
            feature_vector[interaction_col] = 1 if is_true else 0
        # If not in TRAINING_COLUMNS, it won't be added to input_df anyway by reindex, so no need for else here.

    input_df_row = pd.Series(feature_vector, dtype=object).reindex(TRAINING_COLUMNS) # Ensure correct order & all columns
    input_df = pd.DataFrame([input_df_row], columns=TRAINING_COLUMNS)

    for col in input_df.columns:
        input_df[col] = pd.to_numeric(input_df[col], errors='ignore')
        if input_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(input_df[col].dtype): input_df[col].fillna(0, inplace=True)
            else: input_df[col].fillna(0, inplace=True)
        
        is_binary_col = any(col.startswith(prefix) for prefix in ['surface_', 'tourney_level_', 'best_of_', 'round_', 'p1_hand_', 'p2_hand_']) or \
                        col in ['p1_lefty_vs_p2_righty', 'p1_righty_vs_p2_lefty', 'both_lefty', 'both_righty']
        
        if is_binary_col:
            try: input_df[col] = input_df[col].round().astype(int) 
            except (ValueError, TypeError): input_df[col] = 0 # Fallback

    input_df_scaled = input_df.copy()
    if NUMERICAL_FEATURES_TO_SCALE is None:
        st.error("List of numerical features to scale not loaded."); return None, None
        
    cols_to_scale_present = [col for col in NUMERICAL_FEATURES_TO_SCALE if col in input_df_scaled.columns]
    if cols_to_scale_present:
        try:
            for col_to_scale in cols_to_scale_present: # Ensure numeric before scaling
                if not pd.api.types.is_numeric_dtype(input_df_scaled[col_to_scale].dtype):
                    input_df_scaled[col_to_scale] = pd.to_numeric(input_df_scaled[col_to_scale], errors='coerce').fillna(0)
            input_df_scaled[cols_to_scale_present] = scaler.transform(input_df_scaled[cols_to_scale_present])
        except Exception as e: st.error(f"Error during scaling input: {e}"); return None, None
    
    try:
        pred_proba = model.predict_proba(input_df_scaled)[0]
        return pred_proba[1], pred_proba[0] 
    except Exception as e: st.error(f"Model prediction error: {e}"); return None, None

# --- 5. Streamlit UI ---
st.title("🎾 Ultimate Tennis Match Predictor")

if model is None or df_historical_matches.empty or TRAINING_COLUMNS is None:
    st.error("App initialization failed: Critical components missing. Check file paths and ensure model objects and historical data are correctly loaded.")
    st.markdown(f"- Model Loaded: `{'Yes' if model else 'No'}`")
    st.markdown(f"- Historical Data Loaded: `{'Yes' if not df_historical_matches.empty else 'No'}` (Shape: {df_historical_matches.shape if not df_historical_matches.empty else 'N/A'})")
    st.markdown(f"- Training Columns Loaded: `{'Yes' if TRAINING_COLUMNS else 'No'}` (Count: {len(TRAINING_COLUMNS) if TRAINING_COLUMNS else 'N/A'})")
    st.markdown(f"- Numerical Features List Loaded: `{'Yes'if NUMERICAL_FEATURES_TO_SCALE else 'No'}`")
    st.markdown(f"- Config (has_elo) Loaded: `{'Yes' if 'has_elo' in MODEL_OBJECTS else 'No'}` (Value: {HAS_ELO_FEATURES})")
    if GOOGLE_DRIVE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
        st.warning("Placeholder GOOGLE_DRIVE_FILE_ID detected. Update it in app.py.")

else: # App can proceed
    st.sidebar.header("Match Input")
    
    player_name_options = ["Enter Player ID Manually"]; display_name_to_id_map = {}
    if not df_historical_matches.empty:
        ids_n1 = df_historical_matches[['winner_id','winner_name']].copy().rename(columns={'winner_id':'id','winner_name':'name'})
        ids_n2 = df_historical_matches[['loser_id','loser_name']].copy().rename(columns={'loser_id':'id','loser_name':'name'})
        if not ids_n1.empty or not ids_n2.empty:
            unique_df = pd.concat([ids_n1,ids_n2]).drop_duplicates('id').dropna(subset=['id','name'])
            if not unique_df.empty and 'id' in unique_df.columns:
                unique_df['id'] = unique_df['id'].astype(int)
                unique_df = unique_df[unique_df['id']!=-1].sort_values('name')
                unique_df['display_name'] = unique_df['name'] + " (ID: " + unique_df['id'].astype(str) + ")"
                player_name_options.extend(unique_df['display_name'].tolist())
                display_name_to_id_map = pd.Series(unique_df['id'].values, index=unique_df['display_name']).to_dict()

    st.sidebar.subheader("Players")
    p1_sel = st.sidebar.selectbox("Player 1:", player_name_options, index=0, key="p1s")
    p2_sel = st.sidebar.selectbox("Player 2:", player_name_options, index=0, key="p2s")
    p1_id_manual = st.sidebar.text_input("Player 1 ID (if 'Enter Manually'):", value="104745") # Djokovic Example
    p2_id_manual = st.sidebar.text_input("Player 2 ID (if 'Enter Manually'):", value="206421") # Alcaraz Example

    st.sidebar.subheader("Match Context")
    surface_ops = ["Hard", "Clay", "Grass", "Carpet", "Unknown"]
    surface_val = st.sidebar.selectbox("Surface:", surface_ops, index=0)
    
    # Using a simpler map for display for level and round, ensure keys match what's in your CSVs for tourney_level/round
    level_map = {"G": "Grand Slam", "M": "Masters 1000", "A": "ATP Tour", 
                 "F": "Tour Finals/Olympics", "D": "Davis Cup", "C": "Challengers", "Unknown": "Unknown"}
    level_disp_options = list(level_map.values())
    level_disp_default_idx = 2 if "ATP Tour" in level_disp_options else 0
    level_disp = st.sidebar.selectbox("Tournament Level:", level_disp_options, index=level_disp_default_idx)
    level_val = [k for k,v in level_map.items() if v == level_disp][0] if level_disp in level_map.values() else "A" # Default to "A" if mapping fails
    
    best_of_val = st.sidebar.selectbox("Best of (sets):", [3, 5], format_func=lambda x: f"{x} sets", index=0)
    
    round_map = {"F": "Final", "SF": "Semi-Final", "QF": "Quarter-Final", 
                 "R16": "Round of 16", "R32": "Round of 32", "R64": "Round of 64", "R128": "Round of 128", 
                 "RR": "Round Robin", "BR": "Bronze Medal", 
                 "Q1":"Qual. R1", "Q2":"Qual. R2", "Q3":"Qual. R3", # Added Q for qualifiers
                 "Unknown": "Unknown"}
    round_disp_options = list(round_map.values())
    round_disp_default_idx = 0 if "Final" in round_disp_options else 0
    round_disp = st.sidebar.selectbox("Round:", round_disp_options, index=round_disp_default_idx)
    round_val = [k for k,v in round_map.items() if v == round_disp][0] if round_disp in round_map.values() else "F" # Default to "F"

    pred_date = datetime.now() # Predict for "today"

    if st.sidebar.button("🔮 Predict Match Outcome", use_container_width=True):
        p1_id_to_use, p2_id_to_use = None, None
        p1_name_for_display, p2_name_for_display = "Player 1", "Player 2"

        if p1_sel != "Enter Player ID Manually":
            p1_id_to_use = display_name_to_id_map.get(p1_sel)
            p1_name_for_display = p1_sel.split(" (ID:")[0] if p1_id_to_use is not None else "P1 (Invalid Sel.)"
        elif p1_id_manual:
            try: p1_id_to_use = int(p1_id_manual); p1_name_for_display = f"ID {p1_id_to_use}"
            except ValueError: st.error("Player 1 ID (manual) is invalid."); 
        
        if p2_sel != "Enter Player ID Manually":
            p2_id_to_use = display_name_to_id_map.get(p2_sel)
            p2_name_for_display = p2_sel.split(" (ID:")[0] if p2_id_to_use is not None else "P2 (Invalid Sel.)"
        elif p2_id_manual:
            try: p2_id_to_use = int(p2_id_manual); p2_name_for_display = f"ID {p2_id_to_use}"
            except ValueError: st.error("Player 2 ID (manual) is invalid."); 

        if p1_id_to_use is None or p2_id_to_use is None:
            st.error("Please provide valid Player 1 and Player 2 identifiers.")
        elif p1_id_to_use == p2_id_to_use:
            st.error("Player 1 and Player 2 cannot be the same.")
        else:
            with st.spinner(f"Analyzing match: {p1_name_for_display} vs {p2_name_for_display}..."):
                p1_stats_live = get_player_latest_stats(p1_id_to_use, pred_date, df_historical_matches, surface_val)
                p2_stats_live = get_player_latest_stats(p2_id_to_use, pred_date, df_historical_matches, surface_val)
                h2h_live = get_h2h_stats(p1_id_to_use, p2_id_to_use, pred_date, df_historical_matches)

                match_context_dict_live = {'surface': surface_val, 'tourney_level': level_val, 
                                           'best_of': int(best_of_val), 'round': round_val, 
                                           'minutes': 120} # Placeholder for avg match duration

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
                        # Function to format stats for JSON view, handling NaNs and floats
                        def format_stats_for_json(stats_dict):
                            formatted = {}
                            for k,v in stats_dict.items():
                                if k == 'id': continue
                                if isinstance(v, float):
                                    if pd.isna(v): formatted[k] = None # or "N/A"
                                    else: formatted[k] = f"{v:.3f}"
                                else:
                                    formatted[k] = v
                            return formatted

                        st.markdown(f"**{p1_name_for_display} (P1) Stats Used:**")
                        st.json(format_stats_for_json(p1_stats_live))
                        st.markdown(f"**{p2_name_for_display} (P2) Stats Used:**")
                        st.json(format_stats_for_json(p2_stats_live))
                        st.markdown(f"**H2H (P1 wins vs P2 wins prior):**")
                        st.json(h2h_live)
                        st.markdown(f"**Match Context Used:**")
                        st.json(match_context_dict_live)
                else:
                    st.error("Prediction could not be generated. Review earlier error messages in the UI or console.")

st.sidebar.markdown("---")
st.sidebar.caption("Powered by a Machine Learning model. Live player statistics are simplified for this demo.")