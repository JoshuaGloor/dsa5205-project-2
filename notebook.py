# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (p2)
#     language: python
#     name: p2
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
print("hello world")

# %%
# Import functions
import pandas as pd
import numpy as np
import yfinance as yf
import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from arch import arch_model

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

# --------------------
# Data Cleaning
# --------------------

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

# --------------------
# Returns construction
# --------------------
# ret_cc: close(t-1) -> close(t) log-returns (standard daily return)
ret_cc = np.log(adj_close / adj_close.shift(1)).dropna() # shift(1) gets previous day's close

# ret_co: close(t) -> next open(t+1) log-returns (overnight return)
open_next = open_px.shift(-1)                       # shift(-1) gets *next* day's open
ret_co = np.log(open_next / adj_close)              # Calculate return from today's close to tomorrow's open
ret_co = ret_co.dropna(how="all")                   # Drop the last row (will be all NaN due to shift)

# Label (y): Isolate NVDA's close-to-open return
nvda_co = ret_co["NVDA"].dropna()                   # Get the 'NVDA' column from ret_co

print(f"[INFO] Shapes -> ret_cc: {ret_cc.shape} | ret_co: {ret_co.shape} | nvda_co: {nvda_co.shape}")
print("\n[HEAD] Sample ret_cc (features):")
print(ret_cc.filter(items=[c for c in ["NVDA","SOXX","MSFT","AMD"] if c in ret_cc.columns]).head())
print("\n[HEAD] nvda_co (label):")
print(nvda_co.head())

# --------------------
# Modeling-ready table (X_y)
# --------------------

## Definitions:
### X_t = Features: Peer/control daily returns (ret_cc) known at time t
### y_t = Target:   NVDA's overnight return (nvda_co) from time t to t+1

# Use all columns from ret_cc EXCEPT the target (NVDA) as features
feature_cols = [c for c in ret_cc.columns if c not in TARGET]
X = ret_cc[feature_cols].copy()                     # Create the feature matrix X

# Align the target series (y) with the feature matrix (X)
y = nvda_co.reindex(X.index)                        # This aligns dates (index t)
# Join X and y, naming the target column 'y_nvda_co'
X_y = X.join(y.rename("y_nvda_co"))
# Drop any rows where the target is missing (e.g., the last day)
X_y = X_y.dropna(subset=["y_nvda_co"])

print("\n[INFO] Modeling table X_y:")
print(f"Rows: {X_y.shape[0]} | Features: {X_y.shape[1]-1} | Target: y_nvda_co")
print(X_y.head())

# %%
# Step 3 — Turn each features close-to-close return at time t into an idiosyncratic residual by removing the part explained by benchmark
# To avoid leakage, coefficients are fit only on past data in the walk-forward loop.

# Get all column names that are peers (thus not the target NVDA, or the control SOXX)
peers_all = [c for c in X_y.columns if c not in ["y_nvda_co", "NVDA", "SOXX"]]

# Safety check to ensure all peers selected are actually in the DataFrame
use_peers = [c for c in peers_all if c in X_y.columns]

# Calculates residuals for a single day's features (row_X) based on a historical training set (train_X)
def residualize_row(train_X: pd.DataFrame, row_X: pd.Series, ctrl_col="SOXX", peers=use_peers):
    out = {}                                        # Initialize an empty dictionary to store results
    
    # Get the control (SOXX) value for the current day (time t)
    soxx_t = row_X[ctrl_col] if ctrl_col in row_X.index else 0.0                        # row_X = feature at time t to calculate the residual
    # Add the raw SOXX return to the output; it is not residualized
    out[ctrl_col] = soxx_t
    
    for p in peers:                                 # Loop through each peer ticker
        # Check if the peer and control columns exist in the training data
        if ctrl_col in train_X.columns and p in train_X.columns:                        # train_X = features up to time t-1 used for fitting model
            
            # Create a training subset of just this peer and the control, dropping NAs
            df_tr = train_X[[p, ctrl_col]].dropna()
            
            # Check for sufficient data (>= 50 points) and variance (std > 0) to fit a model
            if len(df_tr) >= 50 and df_tr[ctrl_col].std() > 1e-8:
                
                # Fit a simple linear regression: peer_return ~ beta * soxx_return
                lr = LinearRegression().fit(df_tr[[ctrl_col]], df_tr[p])
                # Get the beta (slope) from the fitted model
                beta = float(lr.coef_[0])
                
                # Calculate the residual for time t:
                # residual = actual_peer_return - predicted_peer_return
                out[p + "_res"] = row_X[p] - (beta * soxx_t)
            
            else:
                # Not enough data to fit a model, so set residual to Not-a-Number
                out[p + "_res"] = np.nan
        else:
            # Columns were missing, so set residual to Not-a-Number
            out[p + "_res"] = np.nan
            
    return pd.Series(out)                           # Convert the output dictionary to a pandas Series
    # The function ultimately ouputs a Series containing the residualized peer features for time t.



# %%
# Step 4 — Walk-forward training & prediction (Ridge on residualized peers to account for collinearity, with a fixed alpha and an expanding window)
## Likely can expand this part later if needed

# 4.1 Select a compact, interpretable subset for the first run
core_cols = ["SOXX", "AMD", "MSFT", "TSM", "AVGO"]          # sector control + 4 peers
core_cols = [c for c in core_cols if c in X_y.columns]      # ensure present
print("[INFO] Using feature columns:", core_cols)

# 4.2 Build containers for results
records = []

# 4.3 Indices for walk-forward
min_train = 250  # ~1 trading year of warm-up
dates = X_y.index

model = Ridge(alpha=1.0)                                        # small, fixed shrinkage. We can grid it later if needed (or do VAR(1))

for i in range(min_train, len(dates)-1):                        # predict y at t using info up to t
    dt_train_end = dates[i-1]
    dt_pred      = dates[i]

    # Split past (train) and current row (features at t)
    X_train_full = X_y.loc[dates[:i], core_cols]               # features up to t-1
    y_train      = X_y.loc[dates[:i], "y_nvda_co"]             # labels up to t-1
    x_t          = X_y.loc[dt_pred, core_cols]                 # features at t

    # Residualize peers on SOXX using past only
    res_t = residualize_row(X_train_full, x_t, ctrl_col="SOXX",
                            peers=[c for c in core_cols if c != "SOXX"])

    # Build training matrix of residuals (past)
    # For each past row, compute residuals with past-of-past only (approx: fit once on full past each iter for speed)
    # To keep it tractable, we’ll compute past residuals using a single regression fit on the full past (per peer).
    # This retains no look-ahead w.r.t. time t and is much faster than per-row fitting.
    X_train_res = pd.DataFrame(index=X_train_full.index)
    X_train_res["SOXX"] = X_train_full["SOXX"]

    for p in [c for c in core_cols if c != "SOXX"]:
        df_tr = X_train_full[[p, "SOXX"]].dropna()
        if len(df_tr) >= 50 and df_tr["SOXX"].std() > 1e-8:
            lr = LinearRegression().fit(df_tr[["SOXX"]], df_tr[p])
            beta = float(lr.coef_[0])
            X_train_res[p + "_res"] = X_train_full[p] - beta * X_train_full["SOXX"]
        else:
            X_train_res[p + "_res"] = np.nan

    # Align and drop NA rows in training
    feat_cols = ["SOXX"] + [c + "_res" for c in core_cols if c != "SOXX"]
    train_df = X_train_res.join(y_train).dropna()
    if train_df.empty or train_df[feat_cols].shape[0] < 100:
        continue

    # Fit Ridge on residualized peers + SOXX (SOXX stays as a control feature)
    model.fit(train_df[feat_cols], train_df["y_nvda_co"])

    # Predict y at t
    x_pred = pd.DataFrame([res_t[feat_cols]], index=[dt_pred])
    y_hat  = float(model.predict(x_pred)[0])

    # Store realized next open gap (label at t)
    y_real = float(X_y.loc[dt_pred, "y_nvda_co"])

    records.append({
        "date": dt_pred,
        "y_hat": y_hat,
        "y_real": y_real
    })

wf_df = pd.DataFrame.from_records(records).set_index("date").sort_index()
print("[INFO] Walk-forward result head:")
print(wf_df.head())


# %%
# Step 5: Trading rule + costs (close at t -> next open at t+1)
## The trading rule is long if forecast > 0, and short otherwise. Enter at close(t), exit at open(t+1), earn y_real. Apply a small switch cost when the position changes

COST_BPS = 5                                                # one-way cost per unit position change (arbitarily set for now, will research trading cost)

wf_df = wf_df.copy()

# Signals in {-1, 0, +1}: short / flat / long
wf_df["signal"] = np.where(wf_df["y_hat"] > 0, 1, np.where(wf_df["y_hat"] < 0, -1, 0))

# Previous day position
wf_df["signal_prev"] = wf_df["signal"].shift(1).fillna(0)

# Position change magnitude (how many one-way "legs" we trade today at the close)
delta_pos = (wf_df["signal"] - wf_df["signal_prev"]).abs()

one_way_cost = COST_BPS / 10000.0
wf_df["cost"] = delta_pos * one_way_cost   # cost paid at the close when changing position

# Close-to-open PnL: position at close(t) times NVDA close-to-open return y_real(t), minus trading cost
wf_df["pnl"] = wf_df["signal"] * wf_df["y_real"] - wf_df["cost"]


# %%
# Step 6 — Evaluation metrics and plots (similarly to assignment 1, mainly involves annualized return, volatility, Sharpe and turnover)
# The compare with NVDA buy-and-hold position from close-to-open

def ann_stats(returns: pd.Series, periods_per_year=252):
    r = returns.dropna()
    if r.empty:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    mu  = r.mean() * periods_per_year
    vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    shp = mu / vol if vol > 0 else np.nan
    return {"ann_ret": mu, "ann_vol": vol, "sharpe": shp}

# --- 1) Core series
net = wf_df["pnl"]                # strategy NET close->open return (after trading costs)
gross = (wf_df["signal"] * wf_df["y_real"])  # before costs (for transparency)
costs = wf_df["cost"]             # cost paid on each day (>=0)

# --- 2) Benchmarks
# (a) Buy & hold NVDA overnight (always long close->open)
bh = wf_df["y_real"]

# (b) Zero-exposure (do-nothing) benchmark is just 0; included implicitly by reporting net stats.

# --- 3) Stats
stats_net   = ann_stats(net)
stats_gross = ann_stats(gross)
stats_bh    = ann_stats(bh)

# Turnover: average one-way "legs" per day (0,1,2) from Step 5 delta_pos
# If you didn't keep delta_pos, recompute it here:
delta_pos = (wf_df["signal"] - wf_df["signal_prev"]).abs()
turnover_legs_per_day = delta_pos.mean()

# Exposure diagnostics
pct_long  = (wf_df["signal"] ==  1).mean()
pct_short = (wf_df["signal"] == -1).mean()
pct_flat  = (wf_df["signal"] ==  0).mean()
avg_abs_pos = wf_df["signal"].abs().mean()

print("\n[RESULTS — Close→Open]")
print("Strategy (NET) : ann_ret={ann_ret:.4f} ann_vol={ann_vol:.4f} Sharpe={sharpe:.2f}".format(**stats_net))
print("Strategy (GROSS): ann_ret={ann_ret:.4f} ann_vol={ann_vol:.4f} Sharpe={sharpe:.2f}".format(**stats_gross))
print("BH NVDA (C→O)  : ann_ret={ann_ret:.4f} ann_vol={ann_vol:.4f} Sharpe={sharpe:.2f}".format(**stats_bh))
print(f"Turnover (legs/day): {turnover_legs_per_day:.3f}")
print(f"Exposure: long={pct_long:.1%}  short={pct_short:.1%}  flat={pct_flat:.1%}  avg|pos|={avg_abs_pos:.3f}")

# --- 4) Cumulative curves
wf_df = wf_df.copy()
wf_df["cumulative_strat_net"]   = (1 + net).cumprod()
wf_df["cumulative_strat_gross"] = (1 + gross).cumprod()
wf_df["cumulative_bh"]          = (1 + bh).cumprod()

try:
    import matplotlib.pyplot as plt
    plt.figure()
    wf_df[["cumulative_strat_net","cumulative_bh"]].plot()
    plt.title("Cumulative Return (Close→Open): Strategy (Net) vs Buy&Hold NVDA")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.show()

    plt.figure()
    wf_df[["cumulative_strat_gross","cumulative_strat_net"]].plot()
    plt.title("Strategy Gross vs Net (cost drag)")
    plt.xlabel("Date"); plt.ylabel("Growth of $1")
    plt.show()
except Exception as e:
    print("[WARN] Plotting skipped:", e)

# --- 5) Optional: hit-rates & distribution diagnostics
hit_long  = ((wf_df["signal"] ==  1) & (wf_df["y_real"] > 0)).mean()
hit_short = ((wf_df["signal"] == -1) & (wf_df["y_real"] < 0)).mean()
print(f"Hit-rate: long={hit_long:.1%}  short={hit_short:.1%}")



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


# %%
