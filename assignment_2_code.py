# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Feature Stocks:
# MSFT, GOOGL, AMZN, ORCL:
# https://investor.nvidia.com/news/press-release-details/2023/NVIDIA-Introduces-Generative-AI-Foundry-Service-on-Microsoft-Azure-for-Enterprises-and-Startups-Worldwide/default.aspx
#
# SNOW, NOW, SAP, ADBE:
# https://www.businesswire.com/news/home/20240318504019/en/Snowflake-Teams-with-NVIDIA-to-Deliver-Full-Stack-AI-Platform-for-Customers-to-Transform-Their-Industries
#
# DELL, HPE, SMCI:
# https://www.dell.com/en-us/dt/corporate/newsroom/announcements/detailpage.press-releases~usa~2024~05~20240520-dell-technologies-expands-dell-ai-factory-with-nvidia-to-turbocharge-ai-adoption.htm#/filter-on/Country:en-us
#
# Others (competitors or suppliers): AMD, INTC, AVGO, TSM, ASML, AMAT, KLAC, LRCX, MU
#
#
# ### Target Variable / asset:
# We will be tracking the % change in NVIDIA share price movement, and thus the aim is to long or short the NVDA shares.
# Frequency of predictions will be daily, as our goal is to take advantage of the rapid movements in AI-tech firm to predict NVDA movements at t+1.   
#
# ### Control / Benchmark:
# We need an appropriate benchmark to trace when the broad market / sector moves so that any predictive signal from peers and the model isn’t just the market movements. An appropriate index we can use as a benchmark are SOXX (semi-conductor ETF which is invested in many companies listed as feature stocks such as AVGO, NVDA, AMD, and more) for industry movements, or SPY for overall market movements.
# Implementation idea (for later): for each peer, regress its return on SOXX and use the residual as the feature (i.e., peer move beyond the sector). This isolates idiosyncratic spillovers into NVDA instead of generic semi beta.
#
# ### Currency
# Will need to double check all the feature stocks, but they should all be in some US stock exchange (NYSE or NASDAQ). So we shouldn't have to deal with any foreign exchanage rates, and deal only with USD.
#
# ### Frequency:
# As we are trying to capitalize on the rapid movements of we will choose daily
#
# ### Decision time:
# We need to define the decision time (the timestamp when we form our signal using only information available by then, and lock in the trade we’ll place). I'm just going to set the decision time at U.S. cash close (at time t, ~16:00 New York), and execution time at next open (t+1).
#
# This allows the label choices to then align naturally, predicting at close for --> next-open. This avoids look-ahead, is easy to explain, and matches data availability (daily movements)

# %%
# Step 0: Import functions
import pandas as pd
import numpy as np
import yfinance as yf
import time
import matplotlib.pyplot as plt
from pathlib import Path # Added for results directory
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings # Added to suppress convergence warnings


# Personal plot configuration
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

# %%
# Step 1: Source the adjusted close data from yahoo, and find the closed-to-closed for features, and closed-to-open for target, and merge into a cleaned DF

# --------------------
# Config
# --------------------
START = "2023-07-01"                                # Define the data start date
END   = "2025-06-30"                                # Define the data end date

TARGET   = ["NVDA"]                                 # Stock we want to predict
PEERS    = [                                        # Related stocks (potential features)
    # Hyperscalers / cloud partners
    "MSFT","GOOGL","AMZN","ORCL",
    # Enterprise software partners
    "SNOW","NOW","SAP","ADBE",
    # Semis & equipment
    "AMD","INTC","AVGO","TSM","ASML","AMAT","KLAC","LRCX","MU",
    # AI server/OEM
    "DELL","HPE","SMCI"
]
CONTROLS = ["SOXX"]                                 # Semiconductor sector ETF (control)
CTRL_COL = "SOXX"                                   # Define control column name explicitly (for using only one)

ALL_TICKERS = sorted(set(TARGET + PEERS + CONTROLS)) # Create a single, sorted list of all tickers to download

# --------------------
# Robust single-ticker fetch function
# --------------------
def fetch_one(ticker, start, end, max_tries=5, sleep_s=1.5):
    """Return (adj_close_series, open_series) indexed by trading dates, or (None, None) on failure."""
    last_err = None                                 # Variable to store the last error
    for k in range(max_tries):                      # Loop for retry attempts
        try:
            # Use yf.Ticker().history() to fetch data for one stock
            h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=False)
            if h.empty:                             # Check if any data was returned
                raise RuntimeError(f"No data for {ticker}")
            
            # Use 'Adj Close' if available (accounts for splits/dividends), else 'Close'
            adj = h["Adj Close"] if "Adj Close" in h.columns else h["Close"]
            opn = h["Open"]                         # Get the 'Open' price
            
            adj.name = ticker                       # Set the pandas Series name to the ticker
            opn.name = ticker                       # Set the pandas Series name to the ticker
            return adj, opn                         # Return the two data series
        
        except Exception as e:                      # Catch any exception during download
            last_err = e                            # Store the error
            time.sleep(sleep_s * (k + 1))           # Wait before retrying (exponential backoff)
            
    print(f"[WARN] Failed {ticker}: {last_err}")    # Print a warning if all retries fail
    return None, None                               # Return None if fetching failed

# --------------------
# Download loop and table assembly
# --------------------
adj_cols, open_cols = [], []                        # Initialize empty lists to store data series

for tkr in ALL_TICKERS:                             # Loop through every ticker in our master list
    adj, opn = fetch_one(tkr, START, END)           # Call the fetch function
    if adj is not None: adj_cols.append(adj)        # Add adjusted close series to list if successful
    if opn is not None: open_cols.append(opn)       # Add open price series to list if successful

# Combine all individual series into wide DataFrames
adj_close = pd.concat(adj_cols, axis=1).sort_index() # axis=1 joins columns; sort by date
open_px   = pd.concat(open_cols, axis=1).sort_index() # axis=1 joins columns; sort by date
print(f"[INFO] Raw shapes -> adj_close: {adj_close.shape} | open_px: {open_px.shape}")


# %%
# Step 2: Data Cleaning and Model table

## Function to drop columns from that have too much missing data for sanity
def drop_sparse(df, thresh=0.8):
    keep = df.columns[df.notna().mean() >= thresh]  # Calculate % of non-NA values, keep if >= 80%
    dropped = sorted(set(df.columns) - set(keep))   # Find which columns were dropped
    if dropped:
        print(f"[INFO] Dropping sparse tickers: {dropped}")
    return df[keep]                                 # Return the filtered DataFrame

adj_close = drop_sparse(adj_close, thresh=0.8)      # Clean the adj_close DataFrame

# Align open_px to match adj_close's rows (index) and columns exactly
open_px   = open_px.reindex(index=adj_close.index, columns=adj_close.columns)

# Define the tickers that MUST be present for the analysis
must_have = [c for c in ["NVDA","SOXX"] if c in adj_close.columns]
if len(must_have) < 2:
    print("[WARN] NVDA or SOXX missing after cleaning.")

# Keep only trading days where our 'must_have' tickers both have data
idx_ok = adj_close.dropna(subset=must_have).index if must_have else adj_close.index
adj_close = adj_close.loc[idx_ok]                   # Filter adj_close by these common dates
open_px   = open_px.loc[idx_ok]                     # Filter open_px by these common dates
print(f"[INFO] Cleaned shapes -> adj_close: {adj_close.shape} | open_px: {open_px.shape}")

# Forward fill internal NaNs. We should be careful, but I believe it's relatively safe to use ensure time series continuity for short gaps that occur in the data due to
# things like public holidays (common practice). It is an assumption we will need to highlight in the report though
adj_close = adj_close.ffill()
open_px = open_px.ffill()

# Drop initial rows if ffill couldn't fill
initial_nas = adj_close.isna().any(axis=1) | open_px.isna().any(axis=1)
if initial_nas.any():
    print(f"[INFO] Dropping {initial_nas.sum()} initial rows with NAs after ffill.")
    adj_close = adj_close.loc[~initial_nas]
    open_px = open_px.loc[~initial_nas]
    print(f"[INFO] Final shapes after initial NA drop -> adj_close: {adj_close.shape} | open_px: {open_px.shape}")

# --------------------
# Returns construction
# --------------------
# ret_cc: close(t-1) -> close(t) log-returns (standard daily return)
ret_cc = np.log(adj_close / adj_close.shift(1)).dropna() # shift(1) gets previous day's close

# ret_co: close(t) -> next open(t+1) log-returns (overnight return)
open_next = open_px.shift(-1)                       # shift(-1) gets next day's open
ret_co = np.log(open_next / adj_close)              # Calculate log return from today's close to tomorrow's open
ret_co = ret_co.dropna(how="all")                   # Drop the last row (will be all NaN due to shift)

# Label (y): Isolate NVDA's close-to-open return
nvda_co = ret_co["NVDA"].dropna()                   # Get the 'NVDA' column from ret_co
print(f"[INFO] Shapes -> ret_cc: {ret_cc.shape} | ret_co: {ret_co.shape} | nvda_co: {nvda_co.shape}")



# --------------------
# Modeling-ready table (X_y)
# --------------------
## Definitions:
### X_t = Features: Peer/control daily returns (ret_cc) known at time t
### y_t = Target:   NVDA's overnight return (nvda_co) from time t to t+1

LABEL_COL = "y_nvda_co"

# Use all columns from ret_cc EXCEPT the target (NVDA) as features
feature_cols = [c for c in ret_cc.columns if c not in TARGET]
X = ret_cc[feature_cols].copy()                     # Create the feature matrix X

# Align the target series (y) with the feature matrix (X)
y = nvda_co.reindex(X.index)                        # This aligns dates (index t)
# Join X and y, naming the target column 'y_nvda_co'
X_y = X.join(y.rename(LABEL_COL))
# Drop any rows where the target is missing (e.g., the last day)
X_y = X_y.dropna(subset=[LABEL_COL])

print("\n[INFO] Modeling table X_y:")
print(f"Rows: {X_y.shape[0]} | Features: {X_y.shape[1]-1} | Target: y_nvda_co")
print(X_y.head())

# Check data date range
print(f"\n[INFO] Data available from {X_y.index.min()} to {X_y.index.max()}")

# %%
# Step 3: Configuration for Rolling Window Backtest
FEATURE_COLS = [c for c in X_y.columns if c != LABEL_COL]   # use ALL features

# Train/Validation/Test window length (time-based using pandas DateOffset)
TRAIN_OFFSET = pd.DateOffset(months=12)    # ~1 year
VAL_OFFSET   = pd.DateOffset(months=6)     # ~6 months
TEST_OFFSET  = pd.DateOffset(months=6)     # ~6 months # Length of the final reporting period
PREDICT_STEP = pd.DateOffset(days=1)       # Predict one day at a time in the test set

# Hyperparameter grid for alpha (Ridge)
ALPHA_GRID = np.logspace(-4, 2, 13)

# Cost config (Account for trading costs for buying and selling, arbitarily set for now)
COST_BPS = 5
ONE_WAY  = COST_BPS/10000.0                # 5 basis point (or 0.05%) trading cost


# Step 3.5 - Results registry to store model outputs consistently
RESULTS_DIR = Path("reports"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Each "model_df" is an indexed-by-date DataFrame with at least columns: ["y_hat","y_real","signal","signal_prev","cost","pnl"]
results_store = {}      # dict[str, pd.DataFrame] keyed by model name
results_summary = None  # pd.DataFrame summary table

def register_results(name: str, df: pd.DataFrame):
    """Register a model's walk-forward result; persists to CSV."""
    if df is None or df.empty:
        print(f"[WARN] No results DataFrame provided for {name}. Skipping registration.")
        return
    assert {"y_real","pnl"}.issubset(df.columns), f"{name} missing required columns 'y_real', 'pnl'"
    results_store[name] = df.copy()
    df_out = df.copy()
    # Save results to CSV file
    try:
        df_out.to_csv(RESULTS_DIR / f"{name}_wf.csv", index=True)
        print(f"[INFO] Results for {name} saved to {RESULTS_DIR / f'{name}_wf.csv'}")
    except Exception as e:
        print(f"[ERROR] Failed to save results for {name} to CSV: {e}")


def ann_stats(returns: pd.Series, periods_per_year=252):
    """Calculates annualized return, volatility, and Sharpe ratio."""
    r = returns.dropna()
    if r.empty:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    mu  = r.mean()*periods_per_year
    vol = r.std(ddof=1)*np.sqrt(periods_per_year)
    shp = mu/vol if vol>1e-9 else (0 if mu==0 else np.sign(mu)*np.inf) # Handle zero vol
    return {"ann_ret": mu, "ann_vol": vol, "sharpe": shp}

def summarize_results(results_store: dict) -> pd.DataFrame:
    """Creates a summary table of performance metrics for all registered models."""
    rows = []
    for name, df in results_store.items():
        if df is None or df.empty: continue # Skip if no results

        net  = ann_stats(df["pnl"])
        # Ensure y_real exists before calculating BH stats
        bh = ann_stats(df["y_real"]) if "y_real" in df.columns else {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
        
        # Calculate turnover and exposure if signal columns exist
        turn = np.nan
        expo = pd.Series(dtype=float)
        if {"signal","signal_prev"}.issubset(df.columns):
            turn = (df["signal"] - df["signal_prev"].fillna(0)).abs().mean()
            expo = df["signal"].value_counts(normalize=True)

        rows.append({
            "model": name,
            **{f"net_{k}":v for k,v in net.items()},
            **{f"bh_{k}":v for k,v in bh.items()}, # Baseline is buy&hold of the target leg (e.g., C->O or C->C)
            "turnover_legs_per_day": float(turn),
            "pct_long": float(expo.get(1,0.0)),
            "pct_short": float(expo.get(-1,0.0)),
            # Calculate pct_flat ensuring it sums to 1 even if signal has NaNs
            "pct_flat": max(0.0, 1.0 - expo.get(1,0.0) - expo.get(-1,0.0)),
        })
        
    if not rows:
        print("[WARN] No results found in results_store to summarize.")
        return pd.DataFrame()
        
    out = pd.DataFrame(rows).set_index("model").sort_values("net_sharpe", ascending=False)
    # Save summary to CSV
    try:
        out.to_csv(RESULTS_DIR / "model_summary.csv")
        print(f"[INFO] Model summary saved to {RESULTS_DIR / 'model_summary.csv'}")
    except Exception as e:
        print(f"[ERROR] Failed to save model summary to CSV: {e}")
    return out


# %%
# Step 4: Residualization Helper Function

# Get all column names that are peers (thus not the target NVDA, or the control SOXX) from FEATURE_COLS
peers_all = [c for c in FEATURE_COLS if c not in ["NVDA", CTRL_COL]] # NVDA shouldn't be in FEATURE_COLS, but added check for safety
# Safety check to ensure all peers selected are actually in the initial DataFrame X_y
use_peers = [c for c in peers_all if c in X_y.columns]

# Calculates residuals for a single day's features (row_X) based on a historical training set (train_X)
def residualize_row(train_X: pd.DataFrame, row_X: pd.Series, ctrl_col=CTRL_COL, peers=use_peers):
    """
    Calculates residuals for row_X (features at time t) using betas fitted ONLY on train_X (data up to t-1). Refits regression for each peer for every call.
    """
    out = {}                                        # Initialize an empty dictionary to store results
    
    # Get the control (SOXX) value for the current day (time t)
    # Use .get() for safety in case row_X doesn't have the control column (though it should)
    soxx_t = row_X.get(ctrl_col, 0.0)               # row_X = feature at time t to calculate the residual
    
    # Add the raw SOXX return to the output; it is not residualized (if present)
    if ctrl_col in row_X.index:
        out[ctrl_col] = soxx_t
    
    for p in peers:                                 # Loop through each peer ticker
        # Check if the peer and control columns exist in the training data (train_X)
        if ctrl_col in train_X.columns and p in train_X.columns: # train_X = features up to time t-1 used for fitting model
            
            # Create a training subset of just this peer and the control, dropping NAs
            df_tr = train_X[[p, ctrl_col]].dropna()
            
            # Check for sufficient data (>= 50 points) and variance (std > 0) to fit a model *in this training slice*
            if len(df_tr) >= 50 and df_tr[ctrl_col].std() > 1e-8:
                
                # Fit a simple linear regression: peer_return ~ beta * soxx_return ON THE TRAINING DATA SLICE
                try:
                    lr = LinearRegression().fit(df_tr[[ctrl_col]], df_tr[p])
                    # Get the beta (slope) from the fitted model
                    beta = float(lr.coef_[0])
                    
                    # Calculate the residual for time t (using row_X values):
                    # residual = actual_peer_return_t - predicted_peer_return_t
                    peer_t = row_X.get(p, np.nan) # Get current peer value safely
                    if not pd.isna(peer_t):
                         out[p + "_res"] = peer_t - (beta * soxx_t)
                    else:
                         out[p + "_res"] = np.nan # Peer value missing at time t
                except Exception as e:
                    print(f"[WARN] LinReg fit failed for peer {p} in residualize_row: {e}")
                    out[p + "_res"] = np.nan # Could not calculate beta
            
            else:
                # Not enough data or no variance in control in train_X to fit a model, so set residual to Not-a-Number
                out[p + "_res"] = np.nan
        else:
            # Control or Peer columns were missing in train_X, so set residual to Not-a-Number
            out[p + "_res"] = np.nan
            
    # Define the expected output columns consistently
    final_cols_order = []
    if ctrl_col in X_y.columns: # Check against original X_y columns for consistency
        final_cols_order.append(ctrl_col)
    final_cols_order += [p + "_res" for p in peers] # All potential residual columns

    # Return a Series with consistent index/columns
    return pd.Series(out, index=final_cols_order)



# %% [markdown]
# ### Model 1: Ridge Regression

# %%
# Step 5: Walk-Forward Functions

# Helper function for hyper parameter tuning the alpha on validation set
# Now expects residualized features as input
def _best_alpha_by_val(X_train_res, y_train, X_val_res, y_val, alphas=ALPHA_GRID):
    """
    Trains a separate model for each alpha ON RESIDUALIZED FEATURES, and then finds the one with the lowest MSE on the validation set. After, it refits a new model
    on the combined (train + validation) residualized data using that best alpha.
    """
    
    # Initialize trackers for the best hyperparameter and its score
    best_alpha, best_mse = None, np.inf
    
    # Determine common columns present in both train and val residualized sets
    common_features = X_train_res.columns.intersection(X_val_res.columns).tolist()
    if not common_features:
        print("[WARN] No common features between residualized train and validation sets.")
        return 1.0, None # Default alpha, no model

    # --- 1. Validation Loop: Find the best alpha ---
    for a in alphas:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge",  Ridge(alpha=a, random_state=42))
        ])
        
        # Prepare training data: Use common features, join target, drop NAs
        train_fit_df = X_train_res[common_features].join(y_train).dropna()
        if train_fit_df.empty: continue
        Xtr_fit = train_fit_df[common_features]
        ytr_fit = train_fit_df[y_train.name]
        if Xtr_fit.empty: continue
             
        try: pipe.fit(Xtr_fit, ytr_fit)
        except ValueError as e:
             print(f"[WARN] Pipe fitting failed for alpha {a}: {e}")
             continue
        
        # Prepare validation data: Use common features, drop NAs, align target
        Xva_pred_ready = X_val_res[common_features].dropna()
        if Xva_pred_ready.empty: mse = np.inf
        else:
             yva_aligned = y_val.loc[Xva_pred_ready.index].dropna()
             valid_idx = Xva_pred_ready.index.intersection(yva_aligned.index)
             if valid_idx.empty: mse = np.inf
             else:
                  Xva_pred_final = Xva_pred_ready.loc[valid_idx]
                  yva_aligned_final = yva_aligned.loc[valid_idx]
                  if Xva_pred_final.empty: mse = np.inf
                  else:
                       y_pred_val = pipe.predict(Xva_pred_final)
                       mse = mean_squared_error(yva_aligned_final, y_pred_val)
        
        if mse < best_mse: best_mse, best_alpha = mse, a
    
    if best_alpha is None:
        print("[WARN] No best alpha found. Defaulting to 1.0")
        best_alpha = 1.0

    # --- 2. Refit on (Train + Val) using common features ---
    X_tv_res = pd.concat([X_train_res, X_val_res], axis=0)[common_features]
    y_tv = pd.concat([y_train, y_val], axis=0)

    final_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge",  Ridge(alpha=best_alpha, random_state=42))
    ])

    final_fit_df = X_tv_res.join(y_tv).dropna()
    if final_fit_df.empty:
         print("[ERROR] Combined Train+Val residualized data empty. Cannot fit final model.")
         return best_alpha, None
         
    Xtv_fit = final_fit_df[common_features]
    ytv_fit = final_fit_df[y_tv.name]

    try: final_pipe.fit(Xtv_fit, ytv_fit)
    except ValueError as e:
        print(f"[ERROR] Final pipe fitting failed with alpha {best_alpha}: {e}")
        return best_alpha, None

    return best_alpha, final_pipe

# Performs a "walk-forward" or "rolling window" backtest using time offsets and row-by-row residualization
def walk_forward_ridge_all_features_time_resid_row(
    X_y: pd.DataFrame,
    feature_cols=FEATURE_COLS, # Raw feature cols from X_y
    label_col=LABEL_COL,
    ctrl_col=CTRL_COL,
    peers_list=use_peers,      # Peers for residualize_row
    train_offset=TRAIN_OFFSET,
    val_offset=VAL_OFFSET,
    test_offset=TEST_OFFSET,
    predict_step=PREDICT_STEP,
    alphas=ALPHA_GRID,
    cost_one_way=ONE_WAY
):
    """
    Simulates walk-forward backtesting using time-based rolling windows. Applies residualize_row function row-by-row based on preceding training data.
    Predicts one day at a time within the final test period.
    """
    
    dates = X_y.index
    first_train_start = dates.min()
    first_predict_date = first_train_start + train_offset + val_offset
    last_date = dates.max()
    test_period_start_date = last_date - test_offset + pd.Timedelta(days=1)
    
    if first_predict_date > last_date:
        raise ValueError("Not enough data for Train + Val offset.")
        
    actual_first_predict_idx = dates.searchsorted(first_predict_date)
    if actual_first_predict_idx >= len(dates): raise ValueError("First prediction date beyond data.")
    actual_first_predict_date = dates[actual_first_predict_idx]

    test_start_idx = dates.searchsorted(test_period_start_date)
    if test_start_idx >= len(dates):
        print("[WARN] Test period start beyond data. Backtest may be shorter.")
        test_start_idx = actual_first_predict_idx

    print(f"[INFO] Backtest run | First prediction: {actual_first_predict_date.date()} | Test start: {dates[test_start_idx].date()} | Last date: {last_date.date()}")

    records = []
    prev_sig = 0
    
    # Define expected columns from residualize_row
    resid_feature_cols = []
    if ctrl_col in feature_cols: resid_feature_cols.append(ctrl_col)
    resid_feature_cols += [p + "_res" for p in peers_list]

    current_predict_idx = actual_first_predict_idx
    while current_predict_idx < len(dates):
        t = dates[current_predict_idx]
        
        # --- 1. Define Window Boundaries ---
        val_end_date = t - pd.Timedelta(days=1)
        val_start_date = val_end_date - val_offset + pd.Timedelta(days=1)
        train_end_date = val_start_date - pd.Timedelta(days=1)
        train_start_date = train_end_date - train_offset + pd.Timedelta(days=1)
        train_start_date = max(train_start_date, first_train_start)
        val_start_date = max(val_start_date, first_train_start)

        # --- 2. Slice Raw Data ---
        hist_train_raw = X_y.loc[train_start_date : train_end_date]
        hist_val_raw   = X_y.loc[val_start_date : val_end_date]
        row_t_raw      = X_y.loc[t]

        if hist_train_raw.empty or hist_val_raw.empty:
            if t >= dates[test_start_idx]: records.append({"date": t, "y_hat": np.nan, "y_real": float(row_t_raw.get(label_col, np.nan)), "alpha": np.nan, "signal": 0, "signal_prev": prev_sig, "cost": 0, "pnl": 0})
            current_predict_idx += 1; continue

        # --- 3. Residualization using the helper function ---
        # Data available *before* day t for residualizing day t's features
        train_X_for_resid_t = X_y.loc[train_start_date : val_end_date, feature_cols] # Up to t-1
        
        # Residualize test row (xt)
        xt_res_series = residualize_row(train_X_for_resid_t, row_t_raw[feature_cols], ctrl_col, peers_list)

        # Residualize validation set (Xva) - Apply row-wise. So for each row in validation, use data *before that row's date* for beta calculation
        Xva_res_list = []
        for val_idx, val_row in hist_val_raw.iterrows():
            train_X_for_resid_val = X_y.loc[train_start_date : val_idx - pd.Timedelta(days=1), feature_cols]
            if not train_X_for_resid_val.empty:
                 res_row = residualize_row(train_X_for_resid_val, val_row[feature_cols], ctrl_col, peers_list)
                 res_row.name = val_idx # Assign index
                 Xva_res_list.append(res_row)
        Xva_res = pd.DataFrame(Xva_res_list) if Xva_res_list else pd.DataFrame(columns=resid_feature_cols)


        # Residualize training set (Xtr) - Apply row-wise (computationally heavy), thus for each row in training, use data *before that row's date*
        # Potential optimization: Calculate betas once on hist_train_raw and apply. Let's stick to the definition for now, but comment on potential slowness in report.
        Xtr_res_list = []
        for train_idx, train_row in hist_train_raw.iterrows():
             # Data strictly before the current training row's date
             train_X_for_resid_train = X_y.loc[train_start_date : train_idx - pd.Timedelta(days=1), feature_cols]
             if not train_X_for_resid_train.empty:
                  res_row = residualize_row(train_X_for_resid_train, train_row[feature_cols], ctrl_col, peers_list)
                  res_row.name = train_idx
                  Xtr_res_list.append(res_row)
        Xtr_res = pd.DataFrame(Xtr_res_list) if Xtr_res_list else pd.DataFrame(columns=resid_feature_cols)

        # Get corresponding labels
        ytr = hist_train_raw[label_col]
        yva = hist_val_raw[label_col]
        y_real = float(row_t_raw[label_col])

        # --- 4. Find Best Alpha and Refit Model ---
        best_alpha, final_model = _best_alpha_by_val(Xtr_res, ytr, Xva_res, yva, alphas=alphas)

        if final_model is None:
            print(f"[WARN] Model fitting failed for {t.date()}. Skipping.")
            if t >= dates[test_start_idx]: records.append({"date": t, "y_hat": np.nan, "y_real": y_real, "alpha": best_alpha, "signal": 0, "signal_prev": prev_sig, "cost": 0, "pnl": 0})
            current_predict_idx += 1; continue

        # --- 5. Predict for Day 't' ---
        # Ensure xt_res_series is aligned with features expected by the model
        model_features = final_model.feature_names_in_
        xt_res_series_aligned = xt_res_series.reindex(model_features)
        xt_res_df = pd.DataFrame([xt_res_series_aligned.values], index=[t], columns=model_features) # Select and order

        if xt_res_df.isnull().any().any():
             y_hat = np.nan; sig = 0
        else:
             try:
                 y_hat = float(final_model.predict(xt_res_df)[0])
                 sig = 1 if y_hat > 0 else (-1 if y_hat < 0 else 0)
             except Exception as e:
                 print(f"[ERROR] Prediction failed at {t.date()}: {e}"); y_hat = np.nan; sig = 0

        # --- 6. Calculate PnL ---
        if pd.isna(y_real):
             pnl = np.nan; trade_cost = np.nan
             legs = abs(sig - prev_sig)
             trade_cost = legs * cost_one_way # Cost might still occur
        else:
             legs = abs(sig - prev_sig)
             gross_pnl = sig * y_real
             trade_cost = legs * cost_one_way
             pnl = gross_pnl - trade_cost

        # --- 7. Record Results (Only for the final test period) ---
        if t >= dates[test_start_idx]:
            records.append({
                "date": t, "y_hat": y_hat, "y_real": y_real, "alpha": best_alpha,
                "signal": sig, "signal_prev": prev_sig, "cost": trade_cost, "pnl": pnl
            })
            
        prev_sig = sig
        current_predict_idx += 1

    # --- End of Loop ---
    print("[INFO] ...Backtest complete.")
    wf = pd.DataFrame.from_records(records)
    if not wf.empty: wf = wf.set_index("date").sort_index()
    return wf

# Execute the time-based walk-forward backtest using row-by-row residualization
print("\n" + "="*50)
print("Executing Time-Based Rolling Window Backtest with Row-by-Row Residualization")
print("="*50)
wf_ridge_all_time_res_row = walk_forward_ridge_all_features_time_resid_row(X_y)
print("\n[INFO] wf_ridge_all_time_res_row (head of results):")
if not wf_ridge_all_time_res_row.empty:
    print(wf_ridge_all_time_res_row.head())
else:
    print("[INFO] No results generated.")
print("="*50)

# Register the results
register_results("Ridge_TimeRoll_ResidRow", wf_ridge_all_time_res_row)

# %%
# Step 6: Evaluation (Using the results registry functions), and plotting results

# Summarize results from the store
print("\n" + "="*40)
print("PERFORMANCE SUMMARY (All Models)")
print("="*40)
results_summary = summarize_results(results_store)
if results_summary is not None and not results_summary.empty:
    print("\nSummary Statistics Table:")
    # Display more precision in the summary table
    with pd.option_context('display.float_format', '{:,.4f}'.format):
        print(results_summary)
else:
     print("No model results available to summarize.")
print("="*40)

# --- Plotting Results ---
# Plot only models present in the results_store
if results_store:
    plt.figure(figsize=(14, 8))
    plot_count = 0
    
    # Plot strategy cumulative returns
    for name, df in results_store.items():
        if df is not None and not df.empty and 'pnl' in df.columns:
            # Add cumulative PnL column if not already added by evaluate_model_results
            if 'cum_pnl' not in df.columns:
                 df['cum_pnl'] = (1 + df['pnl'].fillna(0)).cumprod() - 1

            # Ensure data is available for plotting
            if not df['cum_pnl'].isna().all():
                 plt.plot(df.index, df['cum_pnl'], label=f"{name} (Net)")
                 plot_count += 1
            else:
                 print(f"[INFO] Skipping plot for {name} - cumulative PnL is all NaN.")

    # Add Buy & Hold NVDA Close-to-Open benchmark from a representative run
    bh_df_source = None
    bh_col_name = 'cum_bh'
    # Find a valid results df to source the benchmark y_real
    for name, df in results_store.items():
        if df is not None and not df.empty and 'y_real' in df.columns:
            bh_df_source = df.copy() # Use this df's index and y_real
            bh_df_source[bh_col_name] = (1 + bh_df_source['y_real'].fillna(0)).cumprod() - 1
            break # Use the first valid one

    if bh_df_source is not None and not bh_df_source[bh_col_name].isna().all():
         plt.plot(bh_df_source.index, bh_df_source[bh_col_name], label='BH NVDA CO', linestyle='--', color='black')
         plot_count += 1
    else:
         print("[WARN] Could not plot Buy & Hold NVDA (CO) benchmark.")

    if plot_count > 0:
        # Determine overall date range for title
        # Correctly find min/max dates across all valid DataFrame indices
        all_min_dates = [df.index.min() for df in results_store.values() if df is not None and not df.empty]
        all_max_dates = [df.index.max() for df in results_store.values() if df is not None and not df.empty]
        
        if all_min_dates and all_max_dates:
             min_date_str = min(all_min_dates).date()
             max_date_str = max(all_max_dates).date()
             plt.title(f"Cumulative PnL Comparison (Test Periods ending {max_date_str})")
        else:
             plt.title("Cumulative PnL Comparison")

        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("[INFO] No valid model results found to plot.")

    # Plot alpha chosen over time for the Ridge model run
    if "Ridge_TimeRoll_ResidRow" in results_store:
         ridge_df = results_store["Ridge_TimeRoll_ResidRow"]
         if 'alpha' in ridge_df.columns and not ridge_df['alpha'].isna().all():
              plt.figure(figsize=(14, 5))
              plt.plot(ridge_df.index, ridge_df['alpha'], marker='.', linestyle='None', label='Chosen Alpha')
              plt.yscale('log') # Use log scale for alpha
              plt.title("Ridge Alpha Chosen Over Time (Log Scale)")
              plt.xlabel("Date")
              plt.ylabel("Alpha (log scale)")
              plt.grid(True)
              plt.show()
         else:
              print("[INFO] Alpha column not available or empty for Ridge model.")

else:
    print("[INFO] results_store is empty. Nothing to plot.")

# %% [markdown]
# ### Lasso Models (also could do  Elastic net (Combines Lasso and Ridge) later if needed)

# %%
# Hyperparameter grids
ALPHA_GRID_RIDGE = np.logspace(-4, 2, 13) # Alpha for Ridge
ALPHA_GRID_LASSO = np.logspace(-5, -1, 13) # Alpha for Lasso (typically needs smaller values)
ALPHA_GRID_ENET = np.logspace(-5, -1, 13) # Alpha for ElasticNet
L1_RATIO_GRID_ENET = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99] # L1 ratio for ElasticNet


# %%
# Generic Walk-Forward Function using a specified model helper (need to test), backtest using time-based rolling windows, row-by-row residualization, and a specified tuning 
# helper.
def walk_forward_generic(
    X_y: pd.DataFrame,
    tuning_helper_func: callable, # e.g., _best_alpha_by_val_ridge
    model_name: str,              # e.g., "Lasso"
    feature_cols=FEATURE_COLS,
    label_col=LABEL_COL,
    ctrl_col=CTRL_COL,
    peers_list=use_peers,
    train_offset=TRAIN_OFFSET,
    val_offset=VAL_OFFSET,
    test_offset=TEST_OFFSET,
    predict_step=PREDICT_STEP,
    hyperparam_grid=None, # Passed to helper, format depends on helper
    cost_one_way=ONE_WAY
):
    dates = X_y.index
    first_train_start = dates.min()
    first_predict_date = first_train_start + train_offset + val_offset
    last_date = dates.max()
    test_period_start_date = last_date - test_offset + pd.Timedelta(days=1)

    if first_predict_date > last_date: raise ValueError(f"Not enough data for {model_name}.")

    actual_first_predict_idx = dates.searchsorted(first_predict_date)
    if actual_first_predict_idx >= len(dates): raise ValueError(f"First prediction date beyond data for {model_name}.")
    actual_first_predict_date = dates[actual_first_predict_idx]

    test_start_idx = dates.searchsorted(test_period_start_date)
    if test_start_idx >= len(dates):
        print(f"[WARN] {model_name}: Test period start beyond data.")
        test_start_idx = actual_first_predict_idx

    print(f"[INFO] {model_name} Backtest | First prediction: {actual_first_predict_date.date()} | Test start: {dates[test_start_idx].date()} | Last date: {last_date.date()}")

    records = []
    prev_sig = 0

    resid_feature_cols = []
    if ctrl_col in feature_cols: resid_feature_cols.append(ctrl_col)
    resid_feature_cols += [p + "_res" for p in peers_list]

    current_predict_idx = actual_first_predict_idx
    while current_predict_idx < len(dates):
        t = dates[current_predict_idx]

        val_end_date = t - pd.Timedelta(days=1)
        val_start_date = val_end_date - val_offset + pd.Timedelta(days=1)
        train_end_date = val_start_date - pd.Timedelta(days=1)
        train_start_date = train_end_date - train_offset + pd.Timedelta(days=1)
        train_start_date = max(train_start_date, first_train_start)
        val_start_date = max(val_start_date, first_train_start)

        hist_train_raw = X_y.loc[train_start_date : train_end_date]
        hist_val_raw   = X_y.loc[val_start_date : val_end_date]
        row_t_raw      = X_y.loc[t]

        if hist_train_raw.empty or hist_val_raw.empty:
            if t >= dates[test_start_idx]: records.append({"date": t, "y_hat": np.nan, "y_real": float(row_t_raw.get(label_col, np.nan)), "hyperparam": np.nan, "signal": 0, "signal_prev": prev_sig, "cost": 0, "pnl": 0})
            current_predict_idx += 1; continue

        train_X_for_resid_t = X_y.loc[train_start_date : val_end_date, feature_cols]
        xt_res_series = residualize_row(train_X_for_resid_t, row_t_raw[feature_cols], ctrl_col, peers_list)

        Xva_res_list = []
        for val_idx, val_row in hist_val_raw.iterrows():
            train_X_for_resid_val = X_y.loc[train_start_date : val_idx - pd.Timedelta(days=1), feature_cols]
            if not train_X_for_resid_val.empty: res_row = residualize_row(train_X_for_resid_val, val_row[feature_cols], ctrl_col, peers_list); res_row.name = val_idx; Xva_res_list.append(res_row)
        Xva_res = pd.DataFrame(Xva_res_list) if Xva_res_list else pd.DataFrame(columns=resid_feature_cols)

        Xtr_res_list = []
        for train_idx, train_row in hist_train_raw.iterrows():
             train_X_for_resid_train = X_y.loc[train_start_date : train_idx - pd.Timedelta(days=1), feature_cols]
             if not train_X_for_resid_train.empty: res_row = residualize_row(train_X_for_resid_train, train_row[feature_cols], ctrl_col, peers_list); res_row.name = train_idx; Xtr_res_list.append(res_row)
        Xtr_res = pd.DataFrame(Xtr_res_list) if Xtr_res_list else pd.DataFrame(columns=resid_feature_cols)

        ytr = hist_train_raw[label_col]; yva = hist_val_raw[label_col]; y_real = float(row_t_raw[label_col])

        # Call the specific tuning helper
        if "alphas" in tuning_helper_func.__code__.co_varnames:
            best_hyperparam, final_model = tuning_helper_func(Xtr_res, ytr, Xva_res, yva, alphas=hyperparam_grid)
        else: # For ENet helper that takes multiple grids
            best_hyperparam, final_model = tuning_helper_func(Xtr_res, ytr, Xva_res, yva)


        if final_model is None:
            print(f"[WARN] {model_name} model fitting failed for {t.date()}. Skipping.")
            if t >= dates[test_start_idx]: records.append({"date": t, "y_hat": np.nan, "y_real": y_real, "hyperparam": best_hyperparam, "signal": 0, "signal_prev": prev_sig, "cost": 0, "pnl": 0})
            current_predict_idx += 1; continue

        model_features = final_model.feature_names_in_
        xt_res_series_aligned = xt_res_series.reindex(model_features)
        xt_res_df = pd.DataFrame([xt_res_series_aligned.values], index=[t], columns=model_features)

        if xt_res_df.isnull().any().any(): y_hat = np.nan; sig = 0
        else:
             try: y_hat = float(final_model.predict(xt_res_df)[0]); sig = 1 if y_hat > 0 else (-1 if y_hat < 0 else 0)
             except Exception as e: print(f"[ERROR] {model_name} prediction failed at {t.date()}: {e}"); y_hat = np.nan; sig = 0

        if pd.isna(y_real): pnl = np.nan; legs = abs(sig - prev_sig); trade_cost = legs * cost_one_way
        else: legs = abs(sig - prev_sig); gross_pnl = sig * y_real; trade_cost = legs * cost_one_way; pnl = gross_pnl - trade_cost

        if t >= dates[test_start_idx]:
            records.append({"date": t, "y_hat": y_hat, "y_real": y_real, "hyperparam": best_hyperparam, "signal": sig, "signal_prev": prev_sig, "cost": trade_cost, "pnl": pnl})

        prev_sig = sig
        current_predict_idx += 1

    print(f"[INFO] ...{model_name} Backtest complete.")
    wf = pd.DataFrame.from_records(records)
    if not wf.empty: wf = wf.set_index("date").sort_index()
    return wf


# --- Helper function for LASSO hyperparameter tuning ---
def _best_alpha_by_val_lasso(X_train_res, y_train, X_val_res, y_val, alphas=ALPHA_GRID_LASSO):
    """
    Trains Lasso for each alpha ON RESIDUALIZED FEATURES, finds lowest MSE on validation,
    and refits on combined (train + validation) residualized data.
    """
    best_alpha, best_mse = None, np.inf
    common_features = X_train_res.columns.intersection(X_val_res.columns).tolist()
    if not common_features: return 1e-3, None # Default alpha, no model

    for a in alphas:
        # Increase max_iter for Lasso convergence
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lasso",  Lasso(alpha=a, random_state=42, max_iter=2000))
        ])
        train_fit_df = X_train_res[common_features].join(y_train).dropna();
        if train_fit_df.empty: continue
        Xtr_fit, ytr_fit = train_fit_df[common_features], train_fit_df[y_train.name]
        if Xtr_fit.empty: continue
        try: pipe.fit(Xtr_fit, ytr_fit)
        except ValueError as e: print(f"[WARN] Lasso pipe fitting failed for alpha {a}: {e}"); continue

        Xva_pred_ready = X_val_res[common_features].dropna()
        if Xva_pred_ready.empty: mse = np.inf
        else:
             yva_aligned = y_val.loc[Xva_pred_ready.index].dropna()
             valid_idx = Xva_pred_ready.index.intersection(yva_aligned.index)
             if valid_idx.empty: mse = np.inf
             else:
                  Xva_pred_final = Xva_pred_ready.loc[valid_idx]; yva_aligned_final = yva_aligned.loc[valid_idx]
                  if Xva_pred_final.empty: mse = np.inf
                  else: y_pred_val = pipe.predict(Xva_pred_final); mse = mean_squared_error(yva_aligned_final, y_pred_val)
        if mse < best_mse: best_mse, best_alpha = mse, a

    if best_alpha is None: best_alpha = 1e-3; print("[WARN] No best alpha found for Lasso. Defaulting.")

    X_tv_res = pd.concat([X_train_res, X_val_res], axis=0)[common_features]; y_tv = pd.concat([y_train, y_val], axis=0)
    final_pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("lasso", Lasso(alpha=best_alpha, random_state=42, max_iter=2000))])
    final_fit_df = X_tv_res.join(y_tv).dropna()
    if final_fit_df.empty: print("[ERROR] Lasso final fit data empty."); return best_alpha, None
    try: final_pipe.fit(final_fit_df[common_features], final_fit_df[y_tv.name])
    except ValueError as e: print(f"[ERROR] Lasso final pipe fitting failed: {e}"); return best_alpha, None
    return best_alpha, final_pipe


# --- Execute Lasso Backtest ---
print("\n" + "="*50); print("Executing Lasso Backtest"); print("="*50)
wf_lasso_time_res = walk_forward_generic(
    X_y,
    _best_alpha_by_val_lasso,           # Lasso-specific tuning helper
    "Lasso",                            # Model name for logging/registration
    hyperparam_grid=ALPHA_GRID_LASSO    # Pass the Lasso alpha grid
)

# Register the results (saves DF to results_store dict and to a CSV file)
# The results_store dictionary should already exist from a previous cell.
register_results("Lasso_TimeRoll_ResidRow", wf_lasso_time_res)

# Optional: Print head of Lasso results DataFrame
if wf_lasso_time_res is not None and not wf_lasso_time_res.empty:
    print("\n[INFO] wf_lasso_time_res (head of results):")
    print(wf_lasso_time_res.head())
else:
    print("[INFO] No results generated for Lasso.")
print("="*50)

# %%
# ## Step 6: Evaluation

# Summarize results from the store
print("\n" + "="*40)
print("PERFORMANCE SUMMARY (All Models)")
print("="*40)
# Ensure results_store is passed to the function
results_summary = summarize_results(results_store) # results_store should exist from previous steps
if results_summary is not None and not results_summary.empty:
    print("\nSummary Statistics Table:")
    # Display more precision in the summary table
    with pd.option_context('display.float_format', '{:,.4f}'.format):
        print(results_summary)
else:
     print("No model results available to summarize.")
print("="*40)


# --- Plotting Results ---
# Plot only models present in the results_store
if results_store:
    plt.figure(figsize=(14, 8))
    plot_count = 0

    # Plot strategy cumulative returns
    for name, df in results_store.items():
        if df is not None and not df.empty and 'pnl' in df.columns:
            # Add cumulative PnL column if it doesn't exist
            if 'cum_pnl' not in df.columns:
                 # Ensure pnl is numeric, fillna for calculation
                 df['cum_pnl'] = (1 + df['pnl'].fillna(0)).cumprod() - 1

            # Ensure data is available for plotting
            if not df['cum_pnl'].isna().all():
                 plt.plot(df.index, df['cum_pnl'], label=f"{name} (Net)")
                 plot_count += 1
            else:
                 print(f"[INFO] Skipping plot for {name} - cumulative PnL is all NaN.")

    # Add Buy & Hold NVDA Close-to-Open benchmark from a representative run
    bh_df_source = None
    bh_col_name = 'cum_bh'
    # Find a valid results df to source the benchmark y_real
    for name, df in results_store.items():
        if df is not None and not df.empty and 'y_real' in df.columns:
            bh_df_source = df.copy() # Use this df's index and y_real
            # Ensure y_real is numeric, fillna for calculation
            bh_df_source[bh_col_name] = (1 + bh_df_source['y_real'].fillna(0)).cumprod() - 1
            break # Use the first valid one

    if bh_df_source is not None and not bh_df_source[bh_col_name].isna().all():
         plt.plot(bh_df_source.index, bh_df_source[bh_col_name], label='BH NVDA CO', linestyle='--', color='black')
         plot_count += 1
    else:
         print("[WARN] Could not plot Buy & Hold NVDA (CO) benchmark.")

    if plot_count > 0:
        # Determine overall date range for title
        # Correctly find min/max dates across all valid DataFrame indices
        all_min_dates = [df.index.min() for df in results_store.values() if df is not None and not df.empty]
        all_max_dates = [df.index.max() for df in results_store.values() if df is not None and not df.empty]

        if all_min_dates and all_max_dates:
             min_date_str = min(all_min_dates).date()
             max_date_str = max(all_max_dates).date()
             plt.title(f"Cumulative PnL Comparison (Test Periods ending {max_date_str})")
        else:
             plt.title("Cumulative PnL Comparison")

        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        # Save the plot
        try:
             plt.savefig(RESULTS_DIR / "cumulative_pnl_comparison.png")
             print(f"[INFO] Cumulative PnL plot saved to {RESULTS_DIR / 'cumulative_pnl_comparison.png'}")
        except Exception as e:
             print(f"[ERROR] Failed to save cumulative PnL plot: {e}")
        plt.show() # Display the plot
    else:
        print("[INFO] No valid model results found to plot.")

    # --- Plot alpha chosen over time for the LASSO model run ---
    lasso_model_key = "Lasso_TimeRoll_ResidRow" # Key used when registering Lasso results
    if lasso_model_key in results_store:
         lasso_df = results_store[lasso_model_key]
         # Check if the hyperparameter column exists and is not all NaN
         # The generic function saves the best alpha under 'hyperparam'
         param_col_lasso = 'hyperparam'
         if param_col_lasso in lasso_df.columns and not lasso_df[param_col_lasso].isna().all():
              plt.figure(figsize=(14, 5))
              plt.plot(lasso_df.index, lasso_df[param_col_lasso], marker='.', linestyle='None', label='Chosen Alpha')
              plt.yscale('log') # Use log scale for alpha
              plt.title("Lasso Alpha Chosen Over Time (Log Scale)") # Specific title for Lasso
              plt.xlabel("Date")
              plt.ylabel("Alpha (log scale)")
              plt.grid(True)
              # Save the plot
              try:
                   plot_filename_lasso = f"{lasso_model_key}_hyperparams_over_time.png"
                   plt.savefig(RESULTS_DIR / plot_filename_lasso)
                   print(f"[INFO] Lasso hyperparameter plot saved to {RESULTS_DIR / plot_filename_lasso}")
              except Exception as e:
                   print(f"[ERROR] Failed to save hyperparameter plot for {lasso_model_key}: {e}")
              plt.show() # Display the plot
         else:
              print(f"[INFO] Hyperparameter column ('{param_col_lasso}') not available or empty for {lasso_model_key}.")
    else:
        print(f"[INFO] Results for {lasso_model_key} not found in results_store. Cannot plot hyperparameters.")

else:
    print("[INFO] results_store is empty. Nothing to plot.")

# %% [markdown]
# ### Model 2: VAR(1) and VAR(10) Time Series Model

# %%
# ========= VAR(p) on close->close returns (trade close->close) =========
# Uses statsmodels VAR; walk-forward expanding window; p in {1, 10}
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

# 1) Build a compact vector (keep it small/interpretable)
var_cols = [c for c in ["NVDA","SOXX","AMD","MSFT","TSM"] if c in ret_cc.columns]
df_var = ret_cc[var_cols].dropna().copy()

def walk_forward_var(df, target="NVDA", p=1, min_train=250, cost_bps=5):
    dates = df.index
    recs = []
    one_way = cost_bps/10000.0
    pos_prev = 0
    for i in range(min_train, len(dates)-1):
        train = df.iloc[:i]           # up to t-1
        fit = VAR(train.values).fit(maxlags=p, ic=None, trend='c')
        # Forecast t (one step ahead): predict close->close return at t for all variables
        fc  = fit.forecast(train.values[-p:], steps=1)
        y_hat = float(fc[0, list(df.columns).index(target)])
        y_real = float(df.iloc[i][target])             # realized close->close at t

        # Signal: sign(y_hat) with shorting
        sig = 1 if y_hat > 0 else (-1 if y_hat < 0 else 0)
        # Costs on position change
        legs = abs(sig - pos_prev)          # 0,1,2
        cost = legs * one_way
        pnl  = sig * y_real - cost          # trade close->close
        recs.append({"date": dates[i], "y_hat": y_hat, "y_real": y_real, "signal": sig,
                     "signal_prev": pos_prev, "cost": cost, "pnl": pnl})
        pos_prev = sig

    out = pd.DataFrame(recs).set_index("date")
    return out

wf_var1  = walk_forward_var(df_var, target="NVDA", p=1,  min_train=250, cost_bps=5)
wf_var10 = walk_forward_var(df_var, target="NVDA", p=10, min_train=250, cost_bps=5)

# Quick eval helper
def ann_stats(r, ppyr=252):
    r = r.dropna()
    mu  = r.mean()*ppyr
    vol = r.std(ddof=1)*np.sqrt(ppyr)
    shp = mu/vol if vol>0 else np.nan
    return mu, vol, shp

for name, wf in [("VAR(1) C->C", wf_var1), ("VAR(10) C->C", wf_var10)]:
    mu, vol, shp = ann_stats(wf["pnl"])
    mu_bh, vol_bh, shp_bh = ann_stats(wf["y_real"])  # buy & hold close->close
    print(f"[{name}]  net ann_ret={mu:.4f} ann_vol={vol:.4f} Sharpe={shp:.2f}  |  BH C->C Sharpe={shp_bh:.2f}")


# %% [markdown]
# ### Model 3: GARCH(1,1) Regression

# %%
# ============================================================
# ARX–GARCH(1,1) walk-forward for NVDA Close->Open (CO) returns
# Manual mean & variance forecasts (no res.forecast x needed)
# ============================================================

# -----------------------------
# Expect X_y from your prep step:
#   - index: trading dates
#   - columns: peers/controls (close->close returns at t) + 'y_nvda_co' (NVDA close->open label)
# -----------------------------
if 'X_y' not in globals():
    raise RuntimeError("X_y not found. Please run your data prep to create X_y (features at t and y_nvda_co).")

# -----------------------------
# Config
# -----------------------------
MIN_TRAIN   = 250      # warm-up window (~1y)
COST_BPS    = 5        # one-way bps per "leg" when position changes
Z_TAU       = 0.3      # trade only if |mu|/sigma > Z_TAU (set None to always trade sign(mu))
CTRL_COL    = "SOXX"
PEER_CANDIDATES = ["SOXX","AMD","MSFT","TSM","AVGO"]  # small, stable set

GARCH_PEERS = [c for c in PEER_CANDIDATES if c in X_y.columns]
if CTRL_COL not in GARCH_PEERS and CTRL_COL in X_y.columns:
    GARCH_PEERS = [CTRL_COL] + GARCH_PEERS

# -----------------------------
# Helpers
# -----------------------------
def ann_stats(r: pd.Series, periods_per_year=252):
    r = r.dropna()
    if r.empty:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    mu  = r.mean() * periods_per_year
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    shp = mu / vol if vol > 0 else np.nan
    return {"ann_ret": mu, "ann_vol": vol, "sharpe": shp}

# Peer idiosyncratic moves using only past data: peer_res = peer - beta*ctrl
def build_resid_matrix_past(Xhist: pd.DataFrame, cols, ctrl=CTRL_COL):
    out = pd.DataFrame(index=Xhist.index)
    if ctrl in Xhist.columns:
        out[ctrl] = Xhist[ctrl]
    for p in [c for c in cols if c != ctrl and c in Xhist.columns]:
        df_tr = Xhist[[p, ctrl]].dropna()
        if len(df_tr) >= 50 and df_tr[ctrl].std() > 1e-8:
            lr = LinearRegression().fit(df_tr[[ctrl]], df_tr[p])
            beta = float(lr.coef_[0])
            out[p + "_res"] = Xhist[p] - beta * Xhist[ctrl]
        else:
            out[p + "_res"] = np.nan
    return out

# GARCH(1,1) model, where sigma^2_{t+1} = omega + alpha * resid_t^2 + beta * sigma_t^2
def manual_garch11_next_var(res, last_resid: float, last_sigma: float):
    params = res.params
    omega  = float(params.get('omega', 0.0))
    alpha1 = float(params.get('alpha[1]', params.get('alpha', 0.0)))
    beta1  = float(params.get('beta[1]',  params.get('beta',  0.0)))
    return max(omega + alpha1*(last_resid**2) + beta1*(last_sigma**2), 0.0)

def walk_forward_arx_garch_manual(Xy: pd.DataFrame, cols, min_train=250, cost_bps=5, z_tau=0.3):
    dates    = Xy.index
    one_way  = cost_bps / 10000.0
    records  = []
    pos_prev = 0

    # Exogenous order: [CTRL_COL] + residualized peers in fixed order
    exog_cols_base = []
    if CTRL_COL in cols:
        exog_cols_base.append(CTRL_COL)
    exog_cols_base += [c + "_res" for c in cols if c != CTRL_COL]

    for i in range(min_train, len(dates) - 1):
        t_pred = dates[i]

        # --- Past data up to t-1
        X_past = Xy.loc[dates[:i], cols]
        y_past = Xy.loc[dates[:i], "y_nvda_co"]

        # Residualize peers using ONLY past data
        Xpast_res = build_resid_matrix_past(X_past, cols, ctrl=CTRL_COL)
        train_df  = Xpast_res.join(y_past).dropna()
        if train_df.shape[0] < 200:
            continue

        exog_cols = [c for c in exog_cols_base if c in train_df.columns]
        if not exog_cols:
            continue

        X_train = train_df[exog_cols].values.astype(float)    # (n, m)
        y_train = train_df["y_nvda_co"].values.astype(float)  # (n,)

        # Fit ARX mean + GARCH(1,1) vol (rescale=False keeps return units)
        am  = arch_model(y_train, mean='ARX', lags=0, x=X_train,
                         vol='GARCH', p=1, q=1, dist='normal', rescale=False)
        res = am.fit(disp='off')

        # --- Build exogenous vector x_t at time t (residualize row t with past betas)
        rowX = Xy.loc[t_pred, cols]
        row_res = {}
        if CTRL_COL in cols and CTRL_COL in rowX.index:
            row_res[CTRL_COL] = float(rowX[CTRL_COL])
        for p in [c for c in cols if c != CTRL_COL]:
            if p in X_past.columns and CTRL_COL in X_past.columns:
                df_tr = X_past[[p, CTRL_COL]].dropna()
                if len(df_tr) >= 50 and df_tr[CTRL_COL].std() > 1e-8:
                    lr = LinearRegression().fit(df_tr[[CTRL_COL]], df_tr[p])
                    row_res[p + "_res"] = float(rowX.get(p, np.nan)) - float(lr.coef_[0]) * float(rowX.get(CTRL_COL, 0.0))
                else:
                    row_res[p + "_res"] = np.nan

        x_t = np.array([[row_res.get(c, np.nan) for c in exog_cols]], dtype=float)
        if np.isnan(x_t).any():
            continue

        # --- Manual mean forecast: mu_hat = Const + sum_j x_j * beta_j
        params = res.params
        mu_hat = float(params.get('Const', 0.0))
        for j in range(x_t.shape[1]):
            coef_name = f'x{j}'
            if coef_name in params.index:
                mu_hat += float(params[coef_name]) * float(x_t[0, j])

        # --- Manual variance forecast using last residual & last sigma
        last_resid = float(res.resid[-1])                         # epsilon_t
        last_sigma = float(res.conditional_volatility[-1])        # sigma_t
        var_hat    = manual_garch11_next_var(res, last_resid, last_sigma)
        sigma_hat  = float(np.sqrt(var_hat))

        # --- Trading signal
        if z_tau is not None and sigma_hat > 0:
            z = mu_hat / sigma_hat
            sig = 1 if z >  z_tau else (-1 if z < -z_tau else 0)
        else:
            sig = 1 if mu_hat > 0 else (-1 if mu_hat < 0 else 0)

        # --- Realized CO return & PnL
        y_real = float(Xy.loc[t_pred, "y_nvda_co"])
        legs   = abs(sig - pos_prev)      # 0,1,2 legs
        cost   = legs * one_way
        pnl    = sig * y_real - cost

        records.append({
            "date": t_pred,
            "mu_hat": mu_hat,
            "sigma_hat": sigma_hat,
            "z_score": (mu_hat / sigma_hat) if sigma_hat > 0 else np.nan,
            "y_real": y_real,
            "signal": sig,
            "signal_prev": pos_prev,
            "cost": cost,
            "pnl": pnl
        })
        pos_prev = sig

    return pd.DataFrame.from_records(records).set_index("date").sort_index()

# -----------------------------
# Run & evaluate
# -----------------------------
wf_garch = walk_forward_arx_garch_manual(
    X_y, cols=GARCH_PEERS, min_train=MIN_TRAIN,
    cost_bps=COST_BPS, z_tau=Z_TAU
)

print("[INFO] wf_garch head:")
print(wf_garch.head())

def summarize_results(wf: pd.DataFrame):
    net   = wf["pnl"]
    gross = wf["signal"] * wf["y_real"]
    bh    = wf["y_real"]  # buy & hold overnight benchmark

    stats_net   = ann_stats(net)
    stats_gross = ann_stats(gross)
    stats_bh    = ann_stats(bh)

    delta_pos = (wf["signal"] - wf["signal_prev"]).abs()
    turnover_legs_per_day = delta_pos.mean()
    pct_long  = (wf["signal"] ==  1).mean()
    pct_short = (wf["signal"] == -1).mean()
    pct_flat  = (wf["signal"] ==  0).mean()

    print("\n[RESULTS — ARX–GARCH Close→Open]")
    print("Strategy (NET) : ann_ret={ann_ret:.4f} ann_vol={ann_vol:.4f} Sharpe={sharpe:.2f}".format(**stats_net))
    print("Strategy (GROSS): ann_ret={ann_ret:.4f} ann_vol={ann_vol:.4f} Sharpe={sharpe:.2f}".format(**stats_gross))
    print("BH NVDA (C→O)  : ann_ret={ann_ret:.4f} ann_vol={ann_vol:.4f} Sharpe={sharpe:.2f}".format(**stats_bh))
    print(f"Turnover (legs/day): {turnover_legs_per_day:.3f}")
    print(f"Exposure: long={(pct_long):.1%}  short={(pct_short):.1%}  flat={(pct_flat):.1%}")

    # Cumulative curves
    try:
        import matplotlib.pyplot as plt
        wf = wf.copy()
        wf["cumulative_strat_net"]   = (1 + net).cumprod()
        wf["cumulative_strat_gross"] = (1 + gross).cumprod()
        wf["cumulative_bh"]          = (1 + bh).cumprod()

        wf[["cumulative_strat_net","cumulative_bh"]].plot(title="Cumulative Return (CO): ARX–GARCH (Net) vs Buy&Hold")
        plt.xlabel("Date"); plt.ylabel("Growth of $1"); plt.show()

        wf[["cumulative_strat_gross","cumulative_strat_net"]].plot(title="Strategy Gross vs Net (cost drag)")
        plt.xlabel("Date"); plt.ylabel("Growth of $1"); plt.show()
    except Exception as e:
        print("[WARN] Plotting skipped:", e)

summarize_results(wf_garch)

