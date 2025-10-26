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
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from pathlib import Path # Added for results directory
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from arch import arch_model # For GARCH
import warnings # Added to suppress convergence warnings
from itertools import product # Needed for ENet grid search


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
# In evaluating the performance of our daily trading strategies, we utilize specific Buy-and-Hold (BH) benchmarks designed to directly correspond to the strategies' intended holding periods, rather than a simple long-term investment "buy and hold forever" approach. This decision allows for a fairer assessment of the timing value added by the models, instead of comparing it to some investment. 
#
# The benchmarks used are: BH NVDA CO (Close-to-Open), which represents the return achieved by passively buying NVDA stock at the market close each day and selling it at the market open the following day, aligning with strategies targeting overnight movements, and BH NVDA CC (Close-to-Close), representing the standard daily return from holding NVDA from one day's close to the next, used for comparing strategies that target full-day returns. By comparing our strategies against these interval-specific benchmarks, we can better isolate whether our models' signals generated alpha beyond simply holding the asset during the targeted trading windows.

# %% [markdown]
# ### Lasso Models (also could do  Elastic net (Combines Lasso and Ridge) later if needed)

# %%
# Hyperparameter grids
ALPHA_GRID_RIDGE = np.logspace(-4, 2, 13) # Alpha for Ridge
ALPHA_GRID_LASSO = np.logspace(-5, -1, 13) # Alpha for Lasso (typically needs smaller values)
ALPHA_GRID_ENET = np.logspace(-5, -1, 13) # Alpha for ElasticNet
L1_RATIO_GRID_ENET = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99] # L1 ratio for ElasticNet

# %%
# Suppress convergence warnings from Lasso during grid search
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# === Lasso Backtest with Fixed Forward Validation ===
print("\n" + "="*50)
print("Executing Lasso Backtest with Fixed Forward Validation")
print("="*50)

# Required variables: X_y, FEATURE_COLS, LABEL_COL, CTRL_COL, use_peers, TRAIN_OFFSET, VAL_OFFSET, TEST_OFFSET, ALPHA_GRID_LASSO, COST_BPS, ONE_WAY,register_results function

# --- 1. Define Fixed Time Splits ---
dates = X_y.index
first_train_start = dates.min()
train_end_date = first_train_start + TRAIN_OFFSET - pd.Timedelta(days=1)
val_end_date = train_end_date + VAL_OFFSET
test_end_date = val_end_date + TEST_OFFSET # Or use dates.max() if preferred

train_df_raw = X_y.loc[first_train_start : train_end_date]
val_df_raw   = X_y.loc[train_end_date + pd.Timedelta(days=1) : val_end_date]
test_df_raw  = X_y.loc[val_end_date + pd.Timedelta(days=1) : test_end_date] # Use test_end_date

print(f"[INFO] Fixed Splits | Train: {train_df_raw.index.min().date()} to {train_df_raw.index.max().date()} ({len(train_df_raw)} days)")
print(f"[INFO] Fixed Splits | Val  : {val_df_raw.index.min().date()} to {val_df_raw.index.max().date()} ({len(val_df_raw)} days)")
print(f"[INFO] Fixed Splits | Test : {test_df_raw.index.min().date()} to {test_df_raw.index.max().date()} ({len(test_df_raw)} days)")

if train_df_raw.empty or val_df_raw.empty or test_df_raw.empty:
    raise ValueError("One or more fixed data splits are empty. Check offsets and data availability.")

# --- 2. Calculate Betas ONCE on Training Data ---
def calculate_betas(train_data_raw, feature_list, ctrl_col_name, peers_list):
    """Calculates residualization betas based on the raw training data."""
    betas = {}
    if ctrl_col_name in train_data_raw.columns:
        train_features_raw = train_data_raw[feature_list] # Use only feature columns for beta calc
        for p in peers_list:
            if p in train_features_raw.columns:
                df_tr = train_features_raw[[p, ctrl_col_name]].dropna()
                if len(df_tr) >= 50 and df_tr[ctrl_col_name].std() > 1e-8:
                    try:
                        lr = LinearRegression().fit(df_tr[[ctrl_col_name]], df_tr[p])
                        betas[p] = float(lr.coef_[0])
                    except Exception as e:
                        print(f"[WARN] Beta calc failed for {p}: {e}"); betas[p] = np.nan
                else: betas[p] = np.nan
            else: betas[p] = np.nan # Peer not in training data
    else: print(f"[WARN] Control column '{ctrl_col_name}' not in training data.")
    return betas

print("[INFO] Calculating residualization betas on fixed training set...")
betas_fixed = calculate_betas(train_df_raw, FEATURE_COLS, CTRL_COL, use_peers)
print(f"[INFO] Betas calculated: { {k: f'{v:.4f}' for k, v in betas_fixed.items()} }") # Print formatted betas

# --- 3. Apply Residualization to All Fixed Sets ---
def apply_resid_slice(df_raw_slice, betas_dict, feature_list, ctrl_col_name, peers_list):
    """Applies pre-calculated betas to residualize features in a data slice."""
    df_res = pd.DataFrame(index=df_raw_slice.index)
    df_features_raw = df_raw_slice[feature_list] # Work with feature columns only

    if ctrl_col_name in df_features_raw.columns:
         df_res[ctrl_col_name] = df_features_raw[ctrl_col_name]
         soxx_series = df_features_raw[ctrl_col_name]
    else:
         soxx_series = pd.Series(0.0, index=df_features_raw.index)
         print(f"[WARN] Control '{ctrl_col_name}' missing during apply_resid_slice.")

    for p in peers_list:
        beta_p = betas_dict.get(p, np.nan)
        if p in df_features_raw.columns and not np.isnan(beta_p):
            df_res[p + "_res"] = df_features_raw[p] - beta_p * soxx_series
        else:
             df_res[p + "_res"] = np.nan
    return df_res

print("[INFO] Applying residualization to Train, Val, Test sets...")
X_train_res = apply_resid_slice(train_df_raw, betas_fixed, FEATURE_COLS, CTRL_COL, use_peers)
X_val_res   = apply_resid_slice(val_df_raw,   betas_fixed, FEATURE_COLS, CTRL_COL, use_peers)
X_test_res  = apply_resid_slice(test_df_raw,  betas_fixed, FEATURE_COLS, CTRL_COL, use_peers)

# Define final feature columns based on residualization
final_feature_cols = []
if CTRL_COL in FEATURE_COLS: final_feature_cols.append(CTRL_COL)
final_feature_cols += [p + "_res" for p in use_peers]
print(f"[INFO] Using residualized features: {final_feature_cols}")

# Align target variables
y_train = train_df_raw[LABEL_COL]
y_val   = val_df_raw[LABEL_COL]
y_test  = test_df_raw[LABEL_COL]

# --- 4. Tune Alpha using Fixed Train/Val sets with residualized features, then refit with combined train + validation set---
def _best_alpha_by_val_lasso_fixed(X_train_res, y_train, X_val_res, y_val, alphas=ALPHA_GRID_LASSO):
    best_alpha, best_mse = None, np.inf
    common_features = X_train_res.columns.intersection(X_val_res.columns).tolist()
    if not common_features: return 1e-3, None # Default alpha, no model

    # Prepare training data (drop NaNs from features/target alignment)
    train_fit_df = X_train_res[common_features].join(y_train).dropna()
    if train_fit_df.empty: print("[ERROR] Lasso Tuning: Training data empty after NaN drop."); return 1e-3, None
    Xtr_fit = train_fit_df[common_features]
    ytr_fit = train_fit_df[LABEL_COL]

    # Prepare validation data similarly
    val_pred_df = X_val_res[common_features].join(y_val).dropna()
    if val_pred_df.empty: print("[WARN] Lasso Tuning: Validation data empty after NaN drop."); # Continue tuning, but result might be less reliable
    Xva_pred = val_pred_df[common_features]
    yva_eval = val_pred_df[LABEL_COL]

    for a in alphas:
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lasso",  Lasso(alpha=a, random_state=42, max_iter=2000))
        ])
        try:
            pipe.fit(Xtr_fit, ytr_fit)
            if not Xva_pred.empty: # Only calculate MSE if validation data exists
                 y_pred_val = pipe.predict(Xva_pred)
                 mse = mean_squared_error(yva_eval, y_pred_val)
                 if mse < best_mse: best_mse, best_alpha = mse, a
            else: # If val data is empty, just keep track of alpha, maybe pick last one? Or default.
                 best_alpha = a # Keep track of the last tested alpha if no validation comparison possible
        except ValueError as e: print(f"[WARN] Lasso pipe fitting failed for alpha {a}: {e}"); continue

    if best_alpha is None: best_alpha = 1e-3; print("[WARN] No best alpha found for Lasso. Defaulting.")

    # Refit on combined Train + Val
    X_tv_res = pd.concat([X_train_res, X_val_res], axis=0)[common_features]
    y_tv = pd.concat([y_train, y_val], axis=0)
    final_pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("lasso", Lasso(alpha=best_alpha, random_state=42, max_iter=2000))])
    final_fit_df = X_tv_res.join(y_tv).dropna()
    if final_fit_df.empty: print("[ERROR] Lasso final fit data empty."); return best_alpha, None
    try:
        final_pipe.fit(final_fit_df[common_features], final_fit_df[LABEL_COL])
    except ValueError as e: print(f"[ERROR] Lasso final pipe fitting failed: {e}"); return best_alpha, None
    return best_alpha, final_pipe

print("\n[INFO] Tuning Lasso alpha using fixed validation set...")
best_alpha_lasso, final_model_lasso = _best_alpha_by_val_lasso_fixed(X_train_res, y_train, X_val_res, y_val)

if final_model_lasso is None:
    print("[ERROR] Final Lasso model could not be trained. Skipping testing.")
    wf_lasso_fixed_res = pd.DataFrame() # Create empty DF
else:
    print(f"[INFO] Best Lasso Alpha: {best_alpha_lasso:.5f}")
    # --- 5. Test Final Model ---
    print("[INFO] Evaluating final Lasso model on fixed test set...")
    X_test_pred = X_test_res[final_model_lasso.feature_names_in_].dropna() # Use features model was trained on, drop rows with NaNs in features
    test_records = []
    prev_sig = 0 # Initialize prev_sig for the fixed test period

    if X_test_pred.empty:
         print("[WARN] No valid data points in the test set after dropping NaNs.")
    else:
        # Predict on the valid test features
        y_hat_test_array = final_model_lasso.predict(X_test_pred)
        y_hat_test = pd.Series(y_hat_test_array, index=X_test_pred.index)

        # Align predictions with actuals and calculate PnL
        test_results_df = pd.DataFrame({'y_hat': y_hat_test}).join(y_test.rename('y_real')).dropna() # Ensure alignment and drop rows where y_real might be missing

        if test_results_df.empty:
             print("[WARN] No common dates between test predictions and actuals.")
        else:
            test_results_df['signal'] = np.where(test_results_df['y_hat'] > 0, 1, np.where(test_results_df['y_hat'] < 0, -1, 0))
            # Calculate signal_prev correctly for the fixed test block
            test_results_df['signal_prev'] = test_results_df['signal'].shift(1).fillna(0) # Start flat on day 1 of test
            test_results_df['delta_pos'] = (test_results_df['signal'] - test_results_df['signal_prev']).abs()
            test_results_df['cost'] = test_results_df['delta_pos'] * ONE_WAY
            test_results_df['pnl'] = test_results_df['signal'] * test_results_df['y_real'] - test_results_df['cost']
            # Add hyperparam column for consistency with generic output (though it's fixed here)
            test_results_df['hyperparam'] = best_alpha_lasso

    # Store results even if empty for consistent handling later
    wf_lasso_fixed_res = test_results_df.copy() if 'test_results_df' in locals() and not test_results_df.empty else pd.DataFrame()

# --- 6. Register Results ---
register_results("Lasso_FixedFWD_Resid", wf_lasso_fixed_res)

# Print head of results
if not wf_lasso_fixed_res.empty:
    print("\n[INFO] wf_lasso_fixed_res (head of test results):")
    print(wf_lasso_fixed_res.head())
else:
    print("[INFO] No results generated for Lasso Fixed FWD.")
print("="*50)


# %%
# --- Imports needed for plotting (if not already done) ---
import matplotlib.pyplot as plt
import pandas as pd

# Step 6: Evaluation (Using the results registry functions), and plotting results

# Summarize results from the store
print("\n" + "="*40)
print("PERFORMANCE SUMMARY (All Models)")
print("="*40)
# Assuming summarize_results function is defined in a previous cell
# It should already handle saving the summary to CSV.
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
            # Add cumulative PnL column if needed
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
             plt.savefig(RESULTS_DIR / "cumulative_pnl_comparison_lasso.png") # Changed filename slightly
             print(f"[INFO] Cumulative PnL plot saved to {RESULTS_DIR / 'cumulative_pnl_comparison_lasso.png'}")
        except Exception as e:
             print(f"[ERROR] Failed to save cumulative PnL plot: {e}")
        plt.show() # Display the plot
    else:
        print("[INFO] No valid model results found to plot.")

    # --- Plot alpha chosen over time for the LASSO model run (Fixed FWD) ---
    lasso_model_key = "Lasso_FixedFWD_Resid" # Key used in register_results
    if lasso_model_key in results_store:
         lasso_df = results_store[lasso_model_key]
         # Note: For fixed validation, 'hyperparam' column will have the SAME best alpha for all test rows.
         # Plotting it still confirms the alpha chosen.
         if 'hyperparam' in lasso_df.columns and not lasso_df['hyperparam'].isna().all():
              plt.figure(figsize=(14, 5))
              # Use scatter plot as value is constant for fixed validation test period
              plt.scatter(lasso_df.index, lasso_df['hyperparam'], marker='.', label=f'Chosen Alpha ({lasso_df["hyperparam"].iloc[0]:.5f})')
              plt.yscale('log') # Use log scale for alpha
              plt.title(f"Lasso Alpha Chosen During Fixed Validation (Test Period View, Log Scale)")
              plt.xlabel("Date (Test Period)")
              plt.ylabel("Alpha (log scale)")
              # Set y-limits to better view the constant alpha if needed, adjust padding
              min_alpha = lasso_df['hyperparam'].min()
              max_alpha = lasso_df['hyperparam'].max()
              if min_alpha == max_alpha and min_alpha > 0:
                   plt.ylim(min_alpha * 0.5, max_alpha * 2) # Example padding for log scale
              plt.legend()
              plt.grid(True)
              # Save the plot
              try:
                   plot_filename = f"{lasso_model_key}_hyperparam_test_period.png"
                   plt.savefig(RESULTS_DIR / plot_filename)
                   print(f"[INFO] Lasso hyperparameter plot saved to {RESULTS_DIR / plot_filename}")
              except Exception as e:
                   print(f"[ERROR] Failed to save hyperparameter plot for {lasso_model_key}: {e}")
              plt.show() # Display the plot
         else:
              print(f"[INFO] Hyperparameter column ('hyperparam') not available or empty for {lasso_model_key}.")
    else:
         print(f"[INFO] Results for {lasso_model_key} not found in results_store.")

else:
    print("[INFO] results_store is empty. Nothing to plot.")

# %% [markdown]
# ### Elastic Net

# %%
# --- Ensure Config Variables Exist (Assume defined in previous cells) ---
# Required: X_y, FEATURE_COLS, LABEL_COL, CTRL_COL, use_peers, TRAIN_OFFSET, VAL_OFFSET, TEST_OFFSET, ALPHA_GRID_ENET, L1_RATIO_GRID_ENET, COST_BPS, ONE_WAY,
#           calculate_betas, apply_resid_slice, register_results functions

# --- 1. Check if splits exist from a previous block (e.g., Lasso block), otherwise create ---
if 'train_df_raw' not in locals() or 'val_df_raw' not in locals() or 'test_df_raw' not in locals():
     print("[INFO] Redefining fixed time splits for Elastic Net...")
     dates = X_y.index
     first_train_start = dates.min()
     train_end_date = first_train_start + TRAIN_OFFSET - pd.Timedelta(days=1)
     val_end_date = train_end_date + VAL_OFFSET
     test_end_date = val_end_date + TEST_OFFSET # Or use dates.max()

     train_df_raw = X_y.loc[first_train_start : train_end_date]
     val_df_raw   = X_y.loc[train_end_date + pd.Timedelta(days=1) : val_end_date]
     test_df_raw  = X_y.loc[val_end_date + pd.Timedelta(days=1) : test_end_date]

     print(f"[INFO] Fixed Splits | Train: {train_df_raw.index.min().date()} to {train_df_raw.index.max().date()} ({len(train_df_raw)} days)")
     print(f"[INFO] Fixed Splits | Val  : {val_df_raw.index.min().date()} to {val_df_raw.index.max().date()} ({len(val_df_raw)} days)")
     print(f"[INFO] Fixed Splits | Test : {test_df_raw.index.min().date()} to {test_df_raw.index.max().date()} ({len(test_df_raw)} days)")

     if train_df_raw.empty or val_df_raw.empty or test_df_raw.empty:
         raise ValueError("One or more fixed data splits are empty. Check offsets and data availability.")
else:
     print("[INFO] Using existing fixed time splits.")

# --- 2. Check/Recalculate Betas on Training Data ---
if 'betas_fixed' not in locals():
    print("[INFO] Calculating residualization betas on fixed training set for Elastic Net...")
    # Ensure calculate_betas function is defined in a previous cell
    betas_fixed = calculate_betas(train_df_raw, FEATURE_COLS, CTRL_COL, use_peers)
    print(f"[INFO] Betas calculated: { {k: f'{v:.4f}' for k, v in betas_fixed.items()} }")
else:
    print("[INFO] Using existing fixed betas.")

# --- 3. Check/Apply Residualization to All Fixed Sets ---
if ('X_train_res' not in locals() or 'X_val_res' not in locals() or 'X_test_res' not in locals() or
    'final_feature_cols' not in locals() or 'y_train' not in locals() or 'y_val' not in locals() or 'y_test' not in locals()):
    print("[INFO] Applying residualization to Train, Val, Test sets for Elastic Net...")
    # Ensure apply_resid_slice function is defined in a previous cell
    X_train_res = apply_resid_slice(train_df_raw, betas_fixed, FEATURE_COLS, CTRL_COL, use_peers)
    X_val_res   = apply_resid_slice(val_df_raw,   betas_fixed, FEATURE_COLS, CTRL_COL, use_peers)
    X_test_res  = apply_resid_slice(test_df_raw,  betas_fixed, FEATURE_COLS, CTRL_COL, use_peers)

    # Define final feature columns based on residualization
    final_feature_cols = []
    if CTRL_COL in FEATURE_COLS: final_feature_cols.append(CTRL_COL)
    final_feature_cols += [p + "_res" for p in use_peers]
    print(f"[INFO] Using residualized features: {final_feature_cols}")

    # Align target variables
    y_train = train_df_raw[LABEL_COL]
    y_val   = val_df_raw[LABEL_COL]
    y_test  = test_df_raw[LABEL_COL]
else:
    print("[INFO] Using existing residualized Train, Val, Test sets and features.")


# --- 4. Tune Alpha and L1 Ratio (elastic net) using Fixed Train/Val sets (Residualized features) ---
def _best_params_by_val_enet_fixed(X_train_res, y_train, X_val_res, y_val, alphas=ALPHA_GRID_ENET, l1_ratios=L1_RATIO_GRID_ENET):
    best_alpha, best_l1, best_mse = None, None, np.inf
    common_features = X_train_res.columns.intersection(X_val_res.columns).tolist()
    if not common_features: return (1e-3, 0.5), None # Default params, no model

    # Prepare training data
    train_fit_df = X_train_res[common_features].join(y_train).dropna()
    if train_fit_df.empty: print("[ERROR] ENet Tuning: Training data empty after NaN drop."); return (1e-3, 0.5), None
    Xtr_fit = train_fit_df[common_features]
    ytr_fit = train_fit_df[LABEL_COL]

    # Prepare validation data
    val_pred_df = X_val_res[common_features].join(y_val).dropna()
    if val_pred_df.empty: print("[WARN] ENet Tuning: Validation data empty after NaN drop.")
    Xva_pred = val_pred_df[common_features]
    yva_eval = val_pred_df[LABEL_COL]

    for a, l1 in product(alphas, l1_ratios):
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("enet",   ElasticNet(alpha=a, l1_ratio=l1, random_state=42, max_iter=2000)) # Reduced max_iter slightly
        ])
        try:
            pipe.fit(Xtr_fit, ytr_fit)
            if not Xva_pred.empty:
                 y_pred_val = pipe.predict(Xva_pred)
                 mse = mean_squared_error(yva_eval, y_pred_val)
                 if mse < best_mse: best_mse, best_alpha, best_l1 = mse, a, l1
            else:
                 best_alpha, best_l1 = a, l1 # Keep track if no validation comparison possible
        except ValueError as e: print(f"[WARN] ENet pipe fitting failed for alpha {a}, l1 {l1}: {e}"); continue

    if best_alpha is None: best_alpha, best_l1 = 1e-3, 0.5; print("[WARN] No best params found for ENet. Defaulting.")

    # Refit on combined Train + Val
    X_tv_res = pd.concat([X_train_res, X_val_res], axis=0)[common_features]
    y_tv = pd.concat([y_train, y_val], axis=0)
    final_pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("enet", ElasticNet(alpha=best_alpha, l1_ratio=best_l1, random_state=42, max_iter=2000))])
    final_fit_df = X_tv_res.join(y_tv).dropna()
    if final_fit_df.empty: print("[ERROR] ENet final fit data empty."); return (best_alpha, best_l1), None
    try:
        final_pipe.fit(final_fit_df[common_features], final_fit_df[LABEL_COL])
    except ValueError as e: print(f"[ERROR] ENet final pipe fitting failed: {e}"); return (best_alpha, best_l1), None

    return (best_alpha, best_l1), final_pipe

print("\n[INFO] Tuning Elastic Net params using fixed validation set...")
best_params_enet, final_model_enet = _best_params_by_val_enet_fixed(X_train_res, y_train, X_val_res, y_val)

if final_model_enet is None:
    print("[ERROR] Final Elastic Net model could not be trained. Skipping testing.")
    wf_enet_fixed_res = pd.DataFrame() # Create empty DF
else:
    best_alpha_enet, best_l1_enet = best_params_enet
    print(f"[INFO] Best ENet Alpha: {best_alpha_enet:.5f}, Best L1 Ratio: {best_l1_enet:.2f}")
    # --- 5. Test Final Model ---
    print("[INFO] Evaluating final Elastic Net model on fixed test set...")
    X_test_pred = X_test_res[final_model_enet.feature_names_in_].dropna()
    test_results_df = pd.DataFrame() # Initialize empty

    if X_test_pred.empty:
         print("[WARN] No valid data points in the test set after dropping NaNs.")
    else:
        y_hat_test_array = final_model_enet.predict(X_test_pred)
        y_hat_test = pd.Series(y_hat_test_array, index=X_test_pred.index)
        test_results_df = pd.DataFrame({'y_hat': y_hat_test}).join(y_test.rename('y_real')).dropna()

        if test_results_df.empty:
             print("[WARN] No common dates between test predictions and actuals.")
        else:
            test_results_df['signal'] = np.where(test_results_df['y_hat'] > 0, 1, np.where(test_results_df['y_hat'] < 0, -1, 0))
            test_results_df['signal_prev'] = test_results_df['signal'].shift(1).fillna(0)
            test_results_df['delta_pos'] = (test_results_df['signal'] - test_results_df['signal_prev']).abs()
            test_results_df['cost'] = test_results_df['delta_pos'] * ONE_WAY
            test_results_df['pnl'] = test_results_df['signal'] * test_results_df['y_real'] - test_results_df['cost']
            # Add hyperparam column (store as tuple)
            test_results_df['hyperparam'] = [best_params_enet] * len(test_results_df)

    wf_enet_fixed_res = test_results_df.copy()

# --- 6. Register Results ---
register_results("ENet_FixedFWD_Resid", wf_enet_fixed_res)

# Print head of results
if not wf_enet_fixed_res.empty:
    print("\n[INFO] wf_enet_fixed_res (head of test results):")
    print(wf_enet_fixed_res.head())
else:
    print("[INFO] No results generated for Elastic Net Fixed FWD.")
print("="*50)


# %%
# Summarize results from the store
print("\n" + "="*40)
print("PERFORMANCE SUMMARY (All Models)")
print("="*40)
# Assuming summarize_results function is defined in a previous cell
# It should already handle saving the summary to CSV.
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
            # Add cumulative PnL column if needed
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
             # Make filename more general if plotting multiple models
             plot_filename_cum = "cumulative_pnl_comparison_all.png"
             plt.savefig(RESULTS_DIR / plot_filename_cum)
             print(f"[INFO] Cumulative PnL plot saved to {RESULTS_DIR / plot_filename_cum}")
        except Exception as e:
             print(f"[ERROR] Failed to save cumulative PnL plot: {e}")
        plt.show() # Display the plot
    else:
        print("[INFO] No valid model results found to plot.")

    # --- Plot alpha and l1_ratio chosen over time for the ELASTIC NET model run (Fixed FWD) ---
    enet_model_key = "ENet_FixedFWD_Resid" # Key used in register_results
    if enet_model_key in results_store:
         enet_df = results_store[enet_model_key]
         # Note: For fixed validation, 'hyperparam' column will have the SAME best (alpha, l1_ratio) tuple for all test rows.
         if 'hyperparam' in enet_df.columns and not enet_df['hyperparam'].isna().all():
              plt.figure(figsize=(14, 5))

              # Extract the chosen alpha and l1_ratio (they are constant in this case)
              chosen_params = enet_df['hyperparam'].iloc[0] # Get the tuple
              chosen_alpha = chosen_params[0]
              chosen_l1 = chosen_params[1]

              # Use scatter plot as values are constant for fixed validation test period
              # Plot alpha
              plt.scatter(enet_df.index, enet_df['hyperparam'].apply(lambda x: x[0]),
                          marker='.', label=f'Chosen Alpha ({chosen_alpha:.5f})')
              # Plot l1_ratio on the same plot or a secondary axis if scales differ significantly
              # For simplicity, plotting on the same axis first (adjust if needed)
              plt.scatter(enet_df.index, enet_df['hyperparam'].apply(lambda x: x[1]),
                          marker='x', label=f'Chosen L1 Ratio ({chosen_l1:.2f})')

              plt.yscale('log') # Use log scale, mainly relevant for alpha
              plt.title(f"Elastic Net Params Chosen During Fixed Validation (Test Period View, Alpha Log Scale)")
              plt.xlabel("Date (Test Period)")
              plt.ylabel("Hyperparameter Value (Alpha Log)")

              # Adjust y-limits if needed for clarity
              min_val = min(chosen_alpha * 0.5, chosen_l1 * 0.5) if chosen_alpha > 0 and chosen_l1 > 0 else 0.00001
              max_val = max(chosen_alpha * 2, chosen_l1 * 2) if chosen_alpha > 0 and chosen_l1 > 0 else 10
              plt.ylim(min_val, max_val)

              plt.legend()
              plt.grid(True)
              # Save the plot
              try:
                   plot_filename = f"{enet_model_key}_hyperparams_test_period.png"
                   plt.savefig(RESULTS_DIR / plot_filename)
                   print(f"[INFO] Elastic Net hyperparameter plot saved to {RESULTS_DIR / plot_filename}")
              except Exception as e:
                   print(f"[ERROR] Failed to save hyperparameter plot for {enet_model_key}: {e}")
              plt.show() # Display the plot
         else:
              print(f"[INFO] Hyperparameter column ('hyperparam') not available or empty for {enet_model_key}.")
    else:
         print(f"[INFO] Results for {enet_model_key} not found in results_store.")

else:
    print("[INFO] results_store is empty. Nothing to plot.")


# %% [markdown]
# ### Model 2: VAR(1) and VAR(10) Time Series Model

# %%
# --- Ensure Config Variables and Data Exist ---
# Required: ret_cc DataFrame (close-to-close returns), TRAIN_OFFSET, VAL_OFFSET, TEST_OFFSET, COST_BPS, ONE_WAY, register_results, ann_stats functions

# --- Define VAR Columns and Prepare Data ---
VAR_TARGET = "NVDA"
var_cols = ret_cc.columns.tolist()
if VAR_TARGET not in var_cols:
     raise ValueError(f"Target {VAR_TARGET} not found in ret_cc columns for VAR.")
print(f"[INFO] Using ALL {len(var_cols)} features from ret_cc for VAR.")
df_var_full = ret_cc[var_cols].dropna(how='all').copy() # Use all columns, drop rows that are all NaN

# --- 1. Define Fixed Time Splits for VAR Data ---
dates_var = df_var_full.index
first_train_start_var = dates_var.min()
train_end_date_var = first_train_start_var + TRAIN_OFFSET - pd.Timedelta(days=1)
val_end_date_var = train_end_date_var + VAL_OFFSET
test_end_date_var = val_end_date_var + TEST_OFFSET # Or use dates_var.max()

train_df_var = df_var_full.loc[first_train_start_var : train_end_date_var]
val_df_var   = df_var_full.loc[train_end_date_var + pd.Timedelta(days=1) : val_end_date_var]
test_df_var  = df_var_full.loc[val_end_date_var + pd.Timedelta(days=1) : test_end_date_var]

# Drop rows with ANY NaNs within each split, as VAR cannot handle missing values
train_df_var = train_df_var.dropna()
val_df_var = val_df_var.dropna()
test_df_var = test_df_var.dropna()

print(f"[INFO] VAR Fixed Splits (after dropna) | Train: {train_df_var.index.min().date()} to {train_df_var.index.max().date()} ({len(train_df_var)} days)")
print(f"[INFO] VAR Fixed Splits (after dropna) | Val  : {val_df_var.index.min().date()} to {val_df_var.index.max().date()} ({len(val_df_var)} days)")
print(f"[INFO] VAR Fixed Splits (after dropna) | Test : {test_df_var.index.min().date()} to {test_df_var.index.max().date()} ({len(test_df_var)} days)")

if train_df_var.empty or val_df_var.empty or test_df_var.empty:
    raise ValueError("One or more fixed data splits are empty for VAR after dropna. Check data quality.")

# --- 2. Tune VAR Lag 'p' using Fixed Train/Val ---
lag_grid = [1, 10] # Lags to test
validation_results_var = {}

target_col_idx = list(train_df_var.columns).index(VAR_TARGET)

print("\n[INFO] Tuning VAR lag 'p' using fixed validation set...")
for p in lag_grid:
    print(f"  Testing p={p}...")
    try:
        # Fit VAR(p) on the training set
        model_var_train = VAR(train_df_var.values)
        fit_var_train = model_var_train.fit(maxlags=p, ic=None, trend='c')
        # Assign names from the actual training columns
        fit_var_train.names = train_df_var.columns # Add names for forecast indexing

        # Generate rolling 1-step ahead forecasts for the validation period
        history = train_df_var.values[-(fit_var_train.k_ar):]
        val_predictions = []

        for i in range(len(val_df_var)):
            forecast = fit_var_train.forecast(history, steps=1)
            val_predictions.append(forecast[0, target_col_idx])
            actual_obs = val_df_var.values[i:i+1]
            history = np.vstack([history[1:], actual_obs])

        y_hat_val = pd.Series(val_predictions, index=val_df_var.index)
        y_real_val = val_df_var[VAR_TARGET]

        mse_val = mean_squared_error(y_real_val.loc[y_hat_val.index], y_hat_val)
        validation_results_var[p] = mse_val
        print(f"    p={p}: Validation MSE = {mse_val:.6f}")

    except Exception as e:
        print(f"[WARN] VAR fitting/forecasting failed for p={p}: {e}")
        validation_results_var[p] = np.inf

# --- 3. Select Best Lag 'p' ---
if not validation_results_var or all(np.isinf(v) for v in validation_results_var.values()):
    best_p = 1
    print("\n[WARN] VAR tuning failed. Defaulting to p=1.")
else:
    best_p = min(validation_results_var, key=validation_results_var.get)
    print(f"\n[INFO] Best VAR Lag selected: p={best_p} (Validation MSE: {validation_results_var[best_p]:.6f})")

# --- 4. Refit Final Model on Train+Val ---
print(f"\n[INFO] Retraining final VAR model with p={best_p} on Train+Validation data...")
train_val_df_var = pd.concat([train_df_var, val_df_var])
# *** Ensure no NaNs in combined data for refitting ***
train_val_df_var = train_val_df_var.dropna()
if train_val_df_var.empty:
    print("[ERROR] Combined Train+Validation data empty after dropna. Cannot refit VAR.")
    final_fit_var = None
else:
    try:
        final_model_var = VAR(train_val_df_var.values)
        final_fit_var = final_model_var.fit(maxlags=best_p, ic=None, trend='c')
        # Assign names from combined data columns
        final_fit_var.names = train_val_df_var.columns # Add names
        print("[INFO] Final VAR model retrained.")
    except Exception as e:
        print(f"[ERROR] Final VAR model retraining failed: {e}")
        final_fit_var = None

# --- 5. Test Final Model (Rolling Forecast on Test Set) ---
test_records_var = []
prev_sig_var = 0

if final_fit_var is not None and not test_df_var.empty:
    print("[INFO] Evaluating final VAR model on fixed test set (using rolling forecast)...")
    test_history = train_val_df_var.values[-(final_fit_var.k_ar):]
    # Ensure target_col_idx matches refit model's columns 
    target_col_idx_refit = list(train_val_df_var.columns).index(VAR_TARGET)

    for i in range(len(test_df_var)):
        t_test = test_df_var.index[i]
        try:
            test_forecast = final_fit_var.forecast(test_history, steps=1)
            # Use correct index
            y_hat_test = float(test_forecast[0, target_col_idx_refit])
            y_real_test = float(test_df_var.iloc[i][VAR_TARGET])

            sig_test = 1 if y_hat_test > 0 else (-1 if y_hat_test < 0 else 0)
            legs_test = abs(sig_test - prev_sig_var)
            cost_test = legs_test * ONE_WAY
            pnl_test = sig_test * y_real_test - cost_test

            test_records_var.append({
                "date": t_test, "y_hat": y_hat_test, "y_real": y_real_test,
                "signal": sig_test, "signal_prev": prev_sig_var, "cost": cost_test, "pnl": pnl_test,
                "hyperparam": best_p
            })
            prev_sig_var = sig_test

            actual_test_obs = test_df_var.values[i:i+1]
            test_history = np.vstack([test_history[1:], actual_test_obs])

        except Exception as e:
            print(f"[WARN] VAR prediction failed at {t_test.date()}: {e}")
            test_records_var.append({
                "date": t_test, "y_hat": np.nan, "y_real": float(test_df_var.iloc[i][VAR_TARGET]),
                "signal": 0, "signal_prev": prev_sig_var, "cost": 0, "pnl": 0,
                "hyperparam": best_p
            })
            try:
                actual_test_obs = test_df_var.values[i:i+1]
                test_history = np.vstack([test_history[1:], actual_test_obs])
            except: pass
            prev_sig_var = 0

    wf_var_fixed_res = pd.DataFrame.from_records(test_records_var)
    if not wf_var_fixed_res.empty: wf_var_fixed_res = wf_var_fixed_res.set_index("date").sort_index()

else:
    print("[WARN] Final VAR model not trained or test set empty. Skipping evaluation.")
    wf_var_fixed_res = pd.DataFrame()

# --- 6. Register Results ---
# Update model name to reflect ALL features 
var_model_name = f"VAR({best_p})_FixedFWD_CC_AllFeat"
register_results(var_model_name, wf_var_fixed_res)

# Print head of results
if not wf_var_fixed_res.empty:
    print(f"\n[INFO] {var_model_name} (head of test results):")
    print(wf_var_fixed_res.head())
else:
    print(f"[INFO] No results generated for {var_model_name}.")
print("="*50)



# %%
# Summarize results from the store
print("\n" + "="*40)
print("PERFORMANCE SUMMARY (All Models)")
print("="*40)
# Assuming summarize_results function is defined in a previous cell
# It should already handle saving the summary to CSV.
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
            # Add cumulative PnL column if needed
            if 'cum_pnl' not in df.columns:
                 df['cum_pnl'] = (1 + df['pnl'].fillna(0)).cumprod() - 1

            # Ensure data is available for plotting
            if not df['cum_pnl'].isna().all():
                 plt.plot(df.index, df['cum_pnl'], label=f"{name} (Net)")
                 plot_count += 1
            else:
                 print(f"[INFO] Skipping plot for {name} - cumulative PnL is all NaN.")

    # Add Buy & Hold NVDA benchmark (Note: VAR used C-C returns, others used C-O)
    # Be careful comparing C-C strategy PnL to C-O benchmark PnL directly on the same plot.
    # We will plot the C-C benchmark relevant to the VAR model.
    bh_df_source_var = None
    bh_col_name_var = 'cum_bh_cc' # Cumulative Buy & Hold Close-to-Close
    # Find the VAR results df to source the C-C benchmark y_real
    var_model_key_pattern = r"VAR\(\d+\)_FixedFWD_CC_AllFeat" # Regex to find VAR model key
    var_model_key = None
    for key in results_store.keys():
        if re.match(var_model_key_pattern, key):
            var_model_key = key
            break

    if var_model_key and var_model_key in results_store:
        var_df = results_store[var_model_key]
        if var_df is not None and not var_df.empty and 'y_real' in var_df.columns:
            bh_df_source_var = var_df.copy()
            bh_df_source_var[bh_col_name_var] = (1 + bh_df_source_var['y_real'].fillna(0)).cumprod() - 1
            if not bh_df_source_var[bh_col_name_var].isna().all():
                 plt.plot(bh_df_source_var.index, bh_df_source_var[bh_col_name_var], label='BH NVDA CC', linestyle='--', color='green') # Different color for C-C
                 plot_count += 1
            else:
                 bh_df_source_var = None # Reset if BH plot fails
    if bh_df_source_var is None:
         print("[WARN] Could not plot Buy & Hold NVDA (C-C) benchmark for VAR.")


    # Add C-O benchmark for other models if present
    bh_df_source_co = None
    bh_col_name_co = 'cum_bh_co' # Cumulative Buy & Hold Close-to-Open
    for name, df in results_store.items():
         if 'Resid' in name and df is not None and not df.empty and 'y_real' in df.columns: # Find a C-O model
            bh_df_source_co = df.copy()
            bh_df_source_co[bh_col_name_co] = (1 + bh_df_source_co['y_real'].fillna(0)).cumprod() - 1
            break
    if bh_df_source_co is not None and not bh_df_source_co[bh_col_name_co].isna().all():
         plt.plot(bh_df_source_co.index, bh_df_source_co[bh_col_name_co], label='BH NVDA CO', linestyle=':', color='black') # Different style/color for C-O
         plot_count += 1
    else:
         print("[WARN] Could not plot Buy & Hold NVDA (C-O) benchmark for other models.")


    if plot_count > 0:
        # Determine overall date range for title
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
             plot_filename_cum = "cumulative_pnl_comparison_all_inc_var.png" # Updated filename
             plt.savefig(RESULTS_DIR / plot_filename_cum)
             print(f"[INFO] Cumulative PnL plot saved to {RESULTS_DIR / plot_filename_cum}")
        except Exception as e:
             print(f"[ERROR] Failed to save cumulative PnL plot: {e}")
        plt.show() # Display the plot
    else:
        print("[INFO] No valid model results found to plot.")

    # --- Plot lag 'p' chosen over time for the VAR model run (Fixed FWD) ---
    # Find the VAR model key again (could be VAR(1) or VAR(10) depending on validation)
    var_model_key_pattern = r"VAR\(\d+\)_FixedFWD_CC_AllFeat"
    var_model_key_actual = None
    for key in results_store.keys():
        if re.match(var_model_key_pattern, key):
            var_model_key_actual = key
            break

    if var_model_key_actual and var_model_key_actual in results_store:
         var_df = results_store[var_model_key_actual]
         # Note: For fixed validation, 'hyperparam' column will have the SAME best lag 'p' for all test rows.
         if 'hyperparam' in var_df.columns and not var_df['hyperparam'].isna().all():
              plt.figure(figsize=(14, 5))
              chosen_p = int(var_df['hyperparam'].iloc[0]) # Get the constant integer lag

              # Use scatter plot as value is constant for fixed validation test period
              plt.scatter(var_df.index, var_df['hyperparam'], marker='.', label=f'Chosen Lag p ({chosen_p})')
              # Use linear scale for lag p
              plt.title(f"VAR Lag (p) Chosen During Fixed Validation (Test Period View)")
              plt.xlabel("Date (Test Period)")
              plt.ylabel("Lag Order (p)")
              # Set integer ticks for y-axis if only few lags tested
              unique_lags = sorted(var_df['hyperparam'].unique())
              if len(unique_lags) < 15: # Adjust threshold if needed
                   plt.yticks(unique_lags)
              plt.legend()
              plt.grid(True)
              # Save the plot
              try:
                   plot_filename = f"{var_model_key_actual}_hyperparam_test_period.png"
                   plt.savefig(RESULTS_DIR / plot_filename)
                   print(f"[INFO] VAR hyperparameter plot saved to {RESULTS_DIR / plot_filename}")
              except Exception as e:
                   print(f"[ERROR] Failed to save hyperparameter plot for {var_model_key_actual}: {e}")
              plt.show() # Display the plot
         else:
              print(f"[INFO] Hyperparameter column ('hyperparam') not available or empty for {var_model_key_actual}.")
    else:
         print(f"[INFO] Results for a VAR model (e.g., VAR(p)_FixedFWD_CC_AllFeat) not found in results_store.")

else:
    print("[INFO] results_store is empty. Nothing to plot.")

# %% [markdown]
# ### Model 3: GARCH(1,1) Regression

# %%
# --- Ensure Config Variables and Data Exist ---
# Required: X_y, FEATURE_COLS, LABEL_COL, CTRL_COL, use_peers, TRAIN_OFFSET, VAL_OFFSET, TEST_OFFSET, COST_BPS, ONE_WAY, calculate_betas, apply_resid_slice,
#           register_results, ann_stats functions

# --- Add GARCH Specific Config ---
Z_TAU = 0.3 # trade only if |mu|/sigma > Z_TAU (set None to always trade sign(mu))

# --- 1. Check/Define Fixed Time Splits ---
if 'train_df_raw' not in locals() or 'val_df_raw' not in locals() or 'test_df_raw' not in locals():
     print("[INFO] Redefining fixed time splits for GARCH...")
     dates = X_y.index
     first_train_start = dates.min()
     train_end_date = first_train_start + TRAIN_OFFSET - pd.Timedelta(days=1)
     val_end_date = train_end_date + VAL_OFFSET
     test_end_date = val_end_date + TEST_OFFSET # Or use dates.max()

     train_df_raw = X_y.loc[first_train_start : train_end_date]
     val_df_raw   = X_y.loc[train_end_date + pd.Timedelta(days=1) : val_end_date]
     test_df_raw  = X_y.loc[val_end_date + pd.Timedelta(days=1) : test_end_date]

     print(f"[INFO] Fixed Splits | Train: {train_df_raw.index.min().date()} to {train_df_raw.index.max().date()} ({len(train_df_raw)} days)")
     print(f"[INFO] Fixed Splits | Val  : {val_df_raw.index.min().date()} to {val_df_raw.index.max().date()} ({len(val_df_raw)} days)")
     print(f"[INFO] Fixed Splits | Test : {test_df_raw.index.min().date()} to {test_df_raw.index.max().date()} ({len(test_df_raw)} days)")

     if train_df_raw.empty or val_df_raw.empty or test_df_raw.empty:
         raise ValueError("One or more fixed data splits are empty. Check offsets and data availability.")
else:
     print("[INFO] Using existing fixed time splits.")

# --- 2. Check/Recalculate Betas ONCE on Training Data ---
if 'betas_fixed' not in locals():
    print("[INFO] Calculating residualization betas on fixed training set for GARCH...")
    betas_fixed = calculate_betas(train_df_raw, FEATURE_COLS, CTRL_COL, use_peers)
    print(f"[INFO] Betas calculated: { {k: f'{v:.4f}' for k, v in betas_fixed.items()} }")
else:
    print("[INFO] Using existing fixed betas.")

# --- 3. Check/Apply Residualization to All Fixed Sets ---
if ('X_train_res' not in locals() or 'X_val_res' not in locals() or 'X_test_res' not in locals() or
    'final_feature_cols' not in locals() or 'y_train' not in locals() or 'y_val' not in locals() or 'y_test' not in locals()):
    print("[INFO] Applying residualization to Train, Val, Test sets for GARCH...")
    X_train_res = apply_resid_slice(train_df_raw, betas_fixed, FEATURE_COLS, CTRL_COL, use_peers)
    X_val_res   = apply_resid_slice(val_df_raw,   betas_fixed, FEATURE_COLS, CTRL_COL, use_peers)
    X_test_res  = apply_resid_slice(test_df_raw,  betas_fixed, FEATURE_COLS, CTRL_COL, use_peers)

    # Define final feature columns based on residualization
    final_feature_cols = []
    if CTRL_COL in FEATURE_COLS: final_feature_cols.append(CTRL_COL)
    final_feature_cols += [p + "_res" for p in use_peers]
    print(f"[INFO] Using residualized features: {final_feature_cols}")

    # Align target variables
    y_train = train_df_raw[LABEL_COL]
    y_val   = val_df_raw[LABEL_COL]
    y_test  = test_df_raw[LABEL_COL]
else:
    print("[INFO] Using existing residualized Train, Val, Test sets and features.")

# --- Define GARCH Variance Forecast Helper ---
# GARCH(1,1) model: sigma^2_{t+1} = omega + alpha * resid_t^2 + beta * sigma_t^2
def manual_garch11_next_var(res, last_resid: float, last_sigma: float):
    """Calculates the 1-step variance forecast from fitted GARCH(1,1) results."""
    params = res.params
    omega  = float(params.get('omega', 0.0))
    # Handle potential naming variations in arch library output
    alpha1 = float(params.get('alpha[1]', params.get('alpha', 0.0)))
    beta1  = float(params.get('beta[1]',  params.get('beta',  0.0)))
    # Ensure variance is non-negative
    return max(omega + alpha1*(last_resid**2) + beta1*(last_sigma**2), 0.0)


# --- 4. Fit ARX-GARCH Model ONCE on Train+Val ---
print(f"\n[INFO] Fitting final ARX-GARCH(1,1) model on Train+Validation data...")
X_tv_res = pd.concat([X_train_res, X_val_res], axis=0)
y_tv = pd.concat([y_train, y_val], axis=0)

# Prepare combined data for fitting (use common features and drop NaNs)
common_features_fit = X_tv_res.columns.intersection(final_feature_cols).tolist()
if not common_features_fit:
    print("[ERROR] No common residualized features found for GARCH fitting.")
    final_garch_fit = None
else:
    fit_df = X_tv_res[common_features_fit].join(y_tv).dropna()
    if fit_df.empty or len(fit_df) < 200: # Check if enough data after NaN drop
        print(f"[ERROR] Not enough valid data ({len(fit_df)}) in Train+Val for GARCH fitting.")
        final_garch_fit = None
    else:
        Xtv_fit = fit_df[common_features_fit].values.astype(float)
        ytv_fit = fit_df[LABEL_COL].values.astype(float)

        try:
            am = arch_model(ytv_fit, mean='ARX', lags=0, x=Xtv_fit,
                            vol='GARCH', p=1, q=1, dist='normal', rescale=False)
            final_garch_fit = am.fit(disp='off')
            print("[INFO] Final ARX-GARCH model fitted.")
            # print(final_garch_fit.summary()) # Optional: print summary
        except Exception as e:
            print(f"[ERROR] Final ARX-GARCH model fitting failed: {e}")
            final_garch_fit = None

# --- 5. Test Final Model (Iterative 1-step Forecasts) ---
test_records_garch = []
prev_sig_garch = 0

if final_garch_fit is not None and not X_test_res.empty:
    print("[INFO] Evaluating final GARCH model on fixed test set (iterative forecast)...")

    # Get features used in the fitted model
    model_feature_names = common_features_fit # Features used during fitting
    params = final_garch_fit.params

    # Get the last residual and conditional volatility from the fitted model (end of Train+Val)
    last_resid_fit = float(final_garch_fit.resid[-1])
    last_sigma_fit = float(final_garch_fit.conditional_volatility[-1])

    # Initialize forecast variables with the last values from the fit
    current_last_resid = last_resid_fit
    current_last_sigma = last_sigma_fit

    # Prepare test features - use only columns the model was trained on
    X_test_predict = X_test_res[model_feature_names].copy()

    for t in test_df_raw.index: # Iterate through test period dates
        if t not in X_test_predict.index: # Skip if date missing after residualization/dropna
             print(f"[WARN] Skipping forecast for {t.date()} - features missing.")
             continue

        xt_res_values = X_test_predict.loc[[t]].values.astype(float) # Get features for day t
        y_real_test = float(test_df_raw.loc[t, LABEL_COL])

        if np.isnan(xt_res_values).any():
            mu_hat = np.nan
            sigma_hat = np.nan
            print(f"[WARN] NaN features for {t.date()}, cannot forecast.")
        else:
            # --- Manual mean forecast: mu_hat = Const + sum_j x_j * beta_j ---
            mu_hat = float(params.get('Const', 0.0))
            for j in range(xt_res_values.shape[1]):
                coef_name = f'x{j}'
                if coef_name in params.index:
                    mu_hat += float(params[coef_name]) * float(xt_res_values[0, j])

            # --- Manual variance forecast using previous step's residual & sigma ---
            var_hat = manual_garch11_next_var(final_garch_fit, current_last_resid, current_last_sigma) # CALL THE HELPER
            sigma_hat = float(np.sqrt(var_hat))

        # --- Trading signal ---
        if Z_TAU is not None and sigma_hat > 1e-9: # Check sigma_hat > 0
            z = mu_hat / sigma_hat if not pd.isna(mu_hat) else np.nan
            sig_test = 1 if z > Z_TAU else (-1 if z < -Z_TAU else 0) if not pd.isna(z) else 0
        elif not pd.isna(mu_hat):
            sig_test = 1 if mu_hat > 0 else (-1 if mu_hat < 0 else 0)
        else: # Handle NaN mu_hat
            sig_test = 0

        # --- Calculate PnL ---
        if pd.isna(y_real_test) or pd.isna(mu_hat) or pd.isna(sigma_hat):
             pnl_test = 0.0 # Assign 0 PnL if cannot trade or outcome unknown
             legs_test = abs(sig_test - prev_sig_garch)
             cost_test = legs_test * ONE_WAY
             # Ensure signal is 0 if forecast failed
             if pd.isna(mu_hat) or pd.isna(sigma_hat): sig_test = 0
        else:
             legs_test = abs(sig_test - prev_sig_garch)
             gross_pnl_test = sig_test * y_real_test
             cost_test = legs_test * ONE_WAY
             pnl_test = gross_pnl_test - cost_test

        test_records_garch.append({
            "date": t, "mu_hat": mu_hat, "sigma_hat": sigma_hat,
            "z_score": (mu_hat / sigma_hat) if sigma_hat > 1e-9 and not pd.isna(mu_hat) else np.nan,
            "y_real": y_real_test, "signal": sig_test, "signal_prev": prev_sig_garch,
            "cost": cost_test, "pnl": pnl_test, "hyperparam": "GARCH(1,1)" # Store model type
        })

        # --- Update last residual and sigma for the NEXT forecast ---
        # Actual residual for day t (if prediction was possible)
        if not pd.isna(mu_hat) and not pd.isna(y_real_test):
            current_last_resid = y_real_test - mu_hat
        else:
            current_last_resid = 0.0 # Or np.nan? Using 0 might be more stable for variance forecast

        # Sigma used for the next step is the one forecasted for *this* step
        if not pd.isna(sigma_hat):
            current_last_sigma = sigma_hat
        else:
            # Handle NaN sigma forecast - maybe use average historical vol? Or keep last known good one?
            # Keeping the last known good one might be simplest here.
            pass # Keep current_last_sigma as it was

        prev_sig_garch = sig_test # Update previous signal

    wf_garch_fixed_res = pd.DataFrame.from_records(test_records_garch)
    if not wf_garch_fixed_res.empty: wf_garch_fixed_res = wf_garch_fixed_res.set_index("date").sort_index()

else:
    print("[WARN] Final GARCH model not trained or test set empty. Skipping evaluation.")
    wf_garch_fixed_res = pd.DataFrame() # Create empty DF

# --- 6. Register Results ---
garch_model_name = "ARXGARCH_FixedFWD_Resid" # Indicate Fixed Forward
register_results(garch_model_name, wf_garch_fixed_res)

# Optional: Print head of results
if not wf_garch_fixed_res.empty:
    print(f"\n[INFO] {garch_model_name} (head of test results):")
    print(wf_garch_fixed_res.head())
else:
    print(f"[INFO] No results generated for {garch_model_name}.")
print("="*50)

# %%
# Summarize results from the store
print("\n" + "="*40)
print("PERFORMANCE SUMMARY (All Models)")
print("="*40)
# Assuming summarize_results function is defined in a previous cell
# It should already handle saving the summary to CSV.
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
            # Add cumulative PnL column if needed
            if 'cum_pnl' not in df.columns:
                 df['cum_pnl'] = (1 + df['pnl'].fillna(0)).cumprod() - 1

            # Ensure data is available for plotting
            if not df['cum_pnl'].isna().all():
                 plt.plot(df.index, df['cum_pnl'], label=f"{name} (Net)")
                 plot_count += 1
            else:
                 print(f"[INFO] Skipping plot for {name} - cumulative PnL is all NaN.")

    # --- Add Relevant Buy & Hold Benchmarks ---
    # Add C-C benchmark if VAR results exist
    bh_df_source_var = None
    bh_col_name_var = 'cum_bh_cc' # Cumulative Buy & Hold Close-to-Close
    var_model_key_pattern = r"VAR\(\d+\)_FixedFWD_CC_AllFeat" # Regex to find VAR model key
    var_model_key = None
    for key in results_store.keys():
        if re.match(var_model_key_pattern, key):
            var_model_key = key
            break
    if var_model_key and var_model_key in results_store:
        var_df = results_store[var_model_key]
        if var_df is not None and not var_df.empty and 'y_real' in var_df.columns:
            bh_df_source_var = var_df.copy()
            bh_df_source_var[bh_col_name_var] = (1 + bh_df_source_var['y_real'].fillna(0)).cumprod() - 1
            if not bh_df_source_var[bh_col_name_var].isna().all():
                 plt.plot(bh_df_source_var.index, bh_df_source_var[bh_col_name_var], label='BH NVDA CC', linestyle='--', color='green')
                 plot_count += 1
            else: bh_df_source_var = None
    if bh_df_source_var is None: print("[INFO] C-C Benchmark (for VAR) not plotted.")

    # Add C-O benchmark if GARCH or other C-O models exist
    bh_df_source_co = None
    bh_col_name_co = 'cum_bh_co' # Cumulative Buy & Hold Close-to-Open
    for name, df in results_store.items():
         # Find a C-O model (like GARCH or the Resid models)
         if ('Resid' in name or 'GARCH' in name) and df is not None and not df.empty and 'y_real' in df.columns:
            bh_df_source_co = df.copy()
            bh_df_source_co[bh_col_name_co] = (1 + bh_df_source_co['y_real'].fillna(0)).cumprod() - 1
            break
    if bh_df_source_co is not None and not bh_df_source_co[bh_col_name_co].isna().all():
         plt.plot(bh_df_source_co.index, bh_df_source_co[bh_col_name_co], label='BH NVDA CO', linestyle=':', color='black')
         plot_count += 1
    else: print("[WARN] Could not plot Buy & Hold NVDA (C-O) benchmark.")

    # --- Finalize Cumulative Plot ---
    if plot_count > 0:
        all_min_dates = [df.index.min() for df in results_store.values() if df is not None and not df.empty]
        all_max_dates = [df.index.max() for df in results_store.values() if df is not None and not df.empty]
        if all_min_dates and all_max_dates:
             min_date_str = min(all_min_dates).date(); max_date_str = max(all_max_dates).date()
             plt.title(f"Cumulative PnL Comparison (Test Periods ending {max_date_str})")
        else: plt.title("Cumulative PnL Comparison")
        plt.xlabel("Date"); plt.ylabel("Cumulative Return"); plt.legend(); plt.grid(True)
        try:
             plot_filename_cum = "cumulative_pnl_comparison_all_inc_garch.png" # Updated filename
             plt.savefig(RESULTS_DIR / plot_filename_cum)
             print(f"[INFO] Cumulative PnL plot saved to {RESULTS_DIR / plot_filename_cum}")
        except Exception as e: print(f"[ERROR] Failed to save cumulative PnL plot: {e}")
        plt.show()
    else: print("[INFO] No valid model results found to plot cumulative returns.")

    # --- Hyperparameter Plot Section ---
    # Check for GARCH model results - Note: Fixed GARCH(1,1) has no tuned hyperparameter to plot
    garch_model_key = "ARXGARCH_FixedFWD_Resid"
    if garch_model_key in results_store:
         garch_df = results_store[garch_model_key]
         if 'hyperparam' in garch_df.columns and not garch_df['hyperparam'].isna().all():
              # The 'hyperparam' column stores the string "GARCH(1,1)"
              # There's no numerical hyperparameter that was tuned and varied over time to plot.
              print(f"\n[INFO] Model '{garch_model_key}' used a fixed GARCH(1,1) structure.")
              print("[INFO] No hyperparameter tuning plot applicable for this GARCH implementation.")
              # Optionally, you could plot the z_score or sigma_hat over time if desired:
              # if 'z_score' in garch_df.columns:
              #      plt.figure(figsize=(14, 5))
              #      plt.plot(garch_df.index, garch_df['z_score'], label='Forecast Z-Score')
              #      plt.title(f"{garch_model_key} Forecast Z-Score Over Test Period")
              #      plt.xlabel("Date"); plt.ylabel("Z-Score"); plt.legend(); plt.grid(True); plt.show()
              # if 'sigma_hat' in garch_df.columns:
              #      plt.figure(figsize=(14, 5))
              #      plt.plot(garch_df.index, garch_df['sigma_hat'], label='Forecast Sigma (Volatility)')
              #      plt.title(f"{garch_model_key} Forecast Sigma Over Test Period")
              #      plt.xlabel("Date"); plt.ylabel("Sigma"); plt.legend(); plt.grid(True); plt.show()
         else:
              print(f"[INFO] Hyperparameter column ('hyperparam') not available or empty for {garch_model_key}.")
    else:
         print(f"[INFO] Results for {garch_model_key} not found in results_store for hyperparameter check.")

else:
    print("[INFO] results_store is empty. Nothing to plot.")


# %% [markdown]
# #### Potential models:
# 1. Some kind of random forest regressor / tree basd ensemble like XGBoost or LightGBM
# 2. Probbly good to also implement some kind of SVR based on Profs notes, or convert to a classification approach (up or down) and implement SVMs and Logistic Regression

# %% [markdown]
# ### References

# %% [markdown]
# 1. Kelly, B. T., Malamud, S., & Zhou, K. (2024). The virtue of complexity in return prediction
# 2. Chinco, A. M., Clark-Joseph, A. D., & Ye, M. (2019). Sparse signals in the cross‐section of returns.
# 3. https://medium.com/@shruti.dhumne/elastic-net-regression-detailed-guide-99dce30b8e6e
# 4. https://medium.com/@adamhassouni111/a-comprehensive-guide-to-implementing-arima-garch-for-trading-strategies-60a48ac3f08f
# 5. https://www.kaggle.com/code/achrafbenssassi/advanced-trading-strategy-using-garch-model-and-bb
# 6. https://www.statsmodels.org/stable/ (for VAR model)
#
# Also searching some Kaggle Competitions for "stock prediction" or "time series forecasting", they often use LGBM/XGBoost as usual from kaggle, so worth exploring.

# %% [markdown]
# ### Archive

# %%
# # Generic Walk-Forward Function using a specified model helper (need to test), backtest using time-based rolling windows, row-by-row residualization, and a specified tuning 
# # helper.
# def walk_forward_generic(
#     X_y: pd.DataFrame,
#     tuning_helper_func: callable, # e.g., _best_alpha_by_val_ridge
#     model_name: str,              # e.g., "Lasso"
#     feature_cols=FEATURE_COLS,
#     label_col=LABEL_COL,
#     ctrl_col=CTRL_COL,
#     peers_list=use_peers,
#     train_offset=TRAIN_OFFSET,
#     val_offset=VAL_OFFSET,
#     test_offset=TEST_OFFSET,
#     predict_step=PREDICT_STEP,
#     hyperparam_grid=None, # Passed to helper, format depends on helper
#     cost_one_way=ONE_WAY
# ):
#     dates = X_y.index
#     first_train_start = dates.min()
#     first_predict_date = first_train_start + train_offset + val_offset
#     last_date = dates.max()
#     test_period_start_date = last_date - test_offset + pd.Timedelta(days=1)

#     if first_predict_date > last_date: raise ValueError(f"Not enough data for {model_name}.")

#     actual_first_predict_idx = dates.searchsorted(first_predict_date)
#     if actual_first_predict_idx >= len(dates): raise ValueError(f"First prediction date beyond data for {model_name}.")
#     actual_first_predict_date = dates[actual_first_predict_idx]

#     test_start_idx = dates.searchsorted(test_period_start_date)
#     if test_start_idx >= len(dates):
#         print(f"[WARN] {model_name}: Test period start beyond data.")
#         test_start_idx = actual_first_predict_idx

#     print(f"[INFO] {model_name} Backtest | First prediction: {actual_first_predict_date.date()} | Test start: {dates[test_start_idx].date()} | Last date: {last_date.date()}")

#     records = []
#     prev_sig = 0

#     resid_feature_cols = []
#     if ctrl_col in feature_cols: resid_feature_cols.append(ctrl_col)
#     resid_feature_cols += [p + "_res" for p in peers_list]

#     current_predict_idx = actual_first_predict_idx
#     while current_predict_idx < len(dates):
#         t = dates[current_predict_idx]

#         val_end_date = t - pd.Timedelta(days=1)
#         val_start_date = val_end_date - val_offset + pd.Timedelta(days=1)
#         train_end_date = val_start_date - pd.Timedelta(days=1)
#         train_start_date = train_end_date - train_offset + pd.Timedelta(days=1)
#         train_start_date = max(train_start_date, first_train_start)
#         val_start_date = max(val_start_date, first_train_start)

#         hist_train_raw = X_y.loc[train_start_date : train_end_date]
#         hist_val_raw   = X_y.loc[val_start_date : val_end_date]
#         row_t_raw      = X_y.loc[t]

#         if hist_train_raw.empty or hist_val_raw.empty:
#             if t >= dates[test_start_idx]: records.append({"date": t, "y_hat": np.nan, "y_real": float(row_t_raw.get(label_col, np.nan)), "hyperparam": np.nan, "signal": 0, "signal_prev": prev_sig, "cost": 0, "pnl": 0})
#             current_predict_idx += 1; continue

#         train_X_for_resid_t = X_y.loc[train_start_date : val_end_date, feature_cols]
#         xt_res_series = residualize_row(train_X_for_resid_t, row_t_raw[feature_cols], ctrl_col, peers_list)

#         Xva_res_list = []
#         for val_idx, val_row in hist_val_raw.iterrows():
#             train_X_for_resid_val = X_y.loc[train_start_date : val_idx - pd.Timedelta(days=1), feature_cols]
#             if not train_X_for_resid_val.empty: res_row = residualize_row(train_X_for_resid_val, val_row[feature_cols], ctrl_col, peers_list); res_row.name = val_idx; Xva_res_list.append(res_row)
#         Xva_res = pd.DataFrame(Xva_res_list) if Xva_res_list else pd.DataFrame(columns=resid_feature_cols)

#         Xtr_res_list = []
#         for train_idx, train_row in hist_train_raw.iterrows():
#              train_X_for_resid_train = X_y.loc[train_start_date : train_idx - pd.Timedelta(days=1), feature_cols]
#              if not train_X_for_resid_train.empty: res_row = residualize_row(train_X_for_resid_train, train_row[feature_cols], ctrl_col, peers_list); res_row.name = train_idx; Xtr_res_list.append(res_row)
#         Xtr_res = pd.DataFrame(Xtr_res_list) if Xtr_res_list else pd.DataFrame(columns=resid_feature_cols)

#         ytr = hist_train_raw[label_col]; yva = hist_val_raw[label_col]; y_real = float(row_t_raw[label_col])

#         # Call the specific tuning helper
#         if "alphas" in tuning_helper_func.__code__.co_varnames:
#             best_hyperparam, final_model = tuning_helper_func(Xtr_res, ytr, Xva_res, yva, alphas=hyperparam_grid)
#         else: # For ENet helper that takes multiple grids
#             best_hyperparam, final_model = tuning_helper_func(Xtr_res, ytr, Xva_res, yva)


#         if final_model is None:
#             print(f"[WARN] {model_name} model fitting failed for {t.date()}. Skipping.")
#             if t >= dates[test_start_idx]: records.append({"date": t, "y_hat": np.nan, "y_real": y_real, "hyperparam": best_hyperparam, "signal": 0, "signal_prev": prev_sig, "cost": 0, "pnl": 0})
#             current_predict_idx += 1; continue

#         model_features = final_model.feature_names_in_
#         xt_res_series_aligned = xt_res_series.reindex(model_features)
#         xt_res_df = pd.DataFrame([xt_res_series_aligned.values], index=[t], columns=model_features)

#         if xt_res_df.isnull().any().any(): y_hat = np.nan; sig = 0
#         else:
#              try: y_hat = float(final_model.predict(xt_res_df)[0]); sig = 1 if y_hat > 0 else (-1 if y_hat < 0 else 0)
#              except Exception as e: print(f"[ERROR] {model_name} prediction failed at {t.date()}: {e}"); y_hat = np.nan; sig = 0

#         if pd.isna(y_real): pnl = np.nan; legs = abs(sig - prev_sig); trade_cost = legs * cost_one_way
#         else: legs = abs(sig - prev_sig); gross_pnl = sig * y_real; trade_cost = legs * cost_one_way; pnl = gross_pnl - trade_cost

#         if t >= dates[test_start_idx]:
#             records.append({"date": t, "y_hat": y_hat, "y_real": y_real, "hyperparam": best_hyperparam, "signal": sig, "signal_prev": prev_sig, "cost": trade_cost, "pnl": pnl})

#         prev_sig = sig
#         current_predict_idx += 1

#     print(f"[INFO] ...{model_name} Backtest complete.")
#     wf = pd.DataFrame.from_records(records)
#     if not wf.empty: wf = wf.set_index("date").sort_index()
#     return wf


# # --- Helper function for LASSO hyperparameter tuning ---
# def _best_alpha_by_val_lasso(X_train_res, y_train, X_val_res, y_val, alphas=ALPHA_GRID_LASSO):
#     """
#     Trains Lasso for each alpha ON RESIDUALIZED FEATURES, finds lowest MSE on validation,
#     and refits on combined (train + validation) residualized data.
#     """
#     best_alpha, best_mse = None, np.inf
#     common_features = X_train_res.columns.intersection(X_val_res.columns).tolist()
#     if not common_features: return 1e-3, None # Default alpha, no model

#     for a in alphas:
#         # Increase max_iter for Lasso convergence
#         pipe = Pipeline([
#             ("scaler", StandardScaler(with_mean=True, with_std=True)),
#             ("lasso",  Lasso(alpha=a, random_state=42, max_iter=2000))
#         ])
#         train_fit_df = X_train_res[common_features].join(y_train).dropna();
#         if train_fit_df.empty: continue
#         Xtr_fit, ytr_fit = train_fit_df[common_features], train_fit_df[y_train.name]
#         if Xtr_fit.empty: continue
#         try: pipe.fit(Xtr_fit, ytr_fit)
#         except ValueError as e: print(f"[WARN] Lasso pipe fitting failed for alpha {a}: {e}"); continue

#         Xva_pred_ready = X_val_res[common_features].dropna()
#         if Xva_pred_ready.empty: mse = np.inf
#         else:
#              yva_aligned = y_val.loc[Xva_pred_ready.index].dropna()
#              valid_idx = Xva_pred_ready.index.intersection(yva_aligned.index)
#              if valid_idx.empty: mse = np.inf
#              else:
#                   Xva_pred_final = Xva_pred_ready.loc[valid_idx]; yva_aligned_final = yva_aligned.loc[valid_idx]
#                   if Xva_pred_final.empty: mse = np.inf
#                   else: y_pred_val = pipe.predict(Xva_pred_final); mse = mean_squared_error(yva_aligned_final, y_pred_val)
#         if mse < best_mse: best_mse, best_alpha = mse, a

#     if best_alpha is None: best_alpha = 1e-3; print("[WARN] No best alpha found for Lasso. Defaulting.")

#     X_tv_res = pd.concat([X_train_res, X_val_res], axis=0)[common_features]; y_tv = pd.concat([y_train, y_val], axis=0)
#     final_pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("lasso", Lasso(alpha=best_alpha, random_state=42, max_iter=2000))])
#     final_fit_df = X_tv_res.join(y_tv).dropna()
#     if final_fit_df.empty: print("[ERROR] Lasso final fit data empty."); return best_alpha, None
#     try: final_pipe.fit(final_fit_df[common_features], final_fit_df[y_tv.name])
#     except ValueError as e: print(f"[ERROR] Lasso final pipe fitting failed: {e}"); return best_alpha, None
#     return best_alpha, final_pipe


# # --- Execute Lasso Backtest ---
# print("\n" + "="*50); print("Executing Lasso Backtest"); print("="*50)
# wf_lasso_time_res = walk_forward_generic(
#     X_y,
#     _best_alpha_by_val_lasso,           # Lasso-specific tuning helper
#     "Lasso",                            # Model name for logging/registration
#     hyperparam_grid=ALPHA_GRID_LASSO    # Pass the Lasso alpha grid
# )

# # Register the results (saves DF to results_store dict and to a CSV file)
# # The results_store dictionary should already exist from a previous cell.
# register_results("Lasso_TimeRoll_ResidRow", wf_lasso_time_res)

# # Print head of Lasso results DataFrame
# if wf_lasso_time_res is not None and not wf_lasso_time_res.empty:
#     print("\n[INFO] wf_lasso_time_res (head of results):")
#     print(wf_lasso_time_res.head())
# else:
#     print("[INFO] No results generated for Lasso.")
# print("="*50)

# %% [markdown]
#
