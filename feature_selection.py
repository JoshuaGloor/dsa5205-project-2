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
# ### Finding the relevant features

# %% [markdown]
# 1. Cloud Service Providers & AI Infrastructure
# - Microsoft (MSFT): nvidianews.nvidia.com/news/nvidia-partners-ai-infrastructure-america
# - Google (GOOGL): nvidianews.nvidia.com/news/nvidia-partners-ai-infrastructure-america
# - Amazon (AMZN): nvidia.com/en-us/industries/industrial-sector/partners/
# - Oracle (ORCL): nvidianews.nvidia.com/news/nvidia-partners-ai-infrastructure-america
# - Akamai (AKAM): nvidianews.nvidia.com/news/nvidia-partners-ai-infrastructure-america
# - Facebook (META)
# - Apple (APPL)
#
# 2. Telecommunications (5G & 6G Development)
# - Nokia (NOK): nokia.com/newsroom/nvidia-and-nokia-to-pioneer-the-ai-platform-for-6g--powering-americas-return-to-telecommunications-leadership/
# - T-Mobile (TMUS): foxbusiness.com/markets/nvidia-launches-massive-ai-push-major-partnerships-across-multiple-industries
# - AT & T (T): thesiliconreview.com
# - Qualcomm Inc. (QCOM)
#
# 3. Automotive & Autonomous Driving
# - Uber (UBER): nvidianews.nvidia.com/news/nvidia-uber-robotaxi
# - Stellantis (STLA): nvidianews.nvidia.com/news/nvidia-uber-robotaxi
# - Lucid Motors (LCID): nvidianews.nvidia.com/news/nvidia-uber-robotaxi
# - Mercedes-Benz Group (MBGYY): nvidianews.nvidia.com/news/nvidia-uber-robotaxi
# - Volvo (VLVLY): nvidianews.nvidia.com/news/nvidia-uber-robotaxi
# - Li Auto (LI)
# - XPeng (XPEV)
# - NIO Inc. (NIO)
#
# 4. Hardware & System Manufacturers
# - Dell Technologies (DELL): nvidianews.nvidia.com/news/nvidia-partners-ai-infrastructure-america
# - Hewlett Packard Enterprise (HPE): nvidianews.nvidia.com/news/nvidia-partners-ai-infrastructure-america
# - Cisco (CSCO): nvidianews.nvidia.com/news/nvidia-partners-ai-infrastructure-america
# - Supermicro (SMCI): nvidianews.nvidia.com/news/nvidia-partners-ai-infrastructure-america
# - Lenovo (LNVGY): nvidia.com/en-au/industries/industrial-sector/partners/
#
# 5. Healthcare & Pharmaceuticals
# - Johnson & Johnson (JNJ): biospace.com/business/j-j-lilly-crest-ai-wave-with-nvidia-partnerships
# - Eli Lilly (LLY): biospace.com/business/j-j-lilly-crest-ai-wave-with-nvidia-partnerships
# - Illumina (ILMN)
# - IQVIA (IQV)
#
# 6. Software, Manufacturing & Digital Twins
# - Palantir (PLTR): foxbusiness.com/markets/nvidia-launches-massive-ai-push-major-partnerships-across-multiple-industries
# - Lowe's (LOW) foxbusiness.com/markets/nvidia-launches-massive-ai-push-major-partnerships-across-multiple-industries
# - Siemens (SIEGY): nvidia.com/en-au/industries/industrial-sector/partners/
# - TSMC (TSM): foxbusiness.com/markets/nvidia-launches-massive-ai-push-major-partnerships-across-multiple-industries
# - Foxconn (HNHPF): foxbusiness.com/markets/nvidia-launches-massive-ai-push-major-partnerships-across-multiple-industries
# - Caterpillar (CAT): foxbusiness.com/markets/nvidia-launches-massive-ai-push-major-partnerships-across-multiple-industries
# - Dassault Systemes (DASTY)
#
# 7. Enterprise Software Partners
# - Snowflake (SNOW): developer.nvidia.com/blog/nvidia-and-snowflake-collaboration-boosts-data-cloud-ai-capabilities/
# - ServiceNOW (NOW): ittech-pulse.com/news/servicenow-and-nvidia-partner-to-scale-trusted-ai-across-global-industries/
# - SAP (SAP): erp.today/how-sap-and-nvidia-are-rewriting-the-enterprise-ai-playbook/
# - Adobe (ADBE): nvidianews.nvidia.com/news/adobe-and-nvidia-partner-to-unlock-the-power-of-generative-ai
# - Salesforce (CRM)
# - Arista Networks (ANET)
#
# 8. Semiconductors and Equipment
# - Broadcom (AVG)
# - ASML (ASML), Applied Materials (AMAT), KLA Corporation (KLAC), Lam Research (LRCX): nvidianews.nvidia.com/news/nvidia-introduces-breakthrough-in-computational-lithography
# - Micron Technology (MU)
#
# 9. Competitors
# - AMD
# - Intel
#
# 10. Direct investments by NVIDIA
# - Recursion Pharmaceuticals (RXRX): ir.recursion.com
# - SoundHound AI (SOUN)
#
#
# The requirement for the initial selection of shares to sieve through was based on their level of partnership, and a high level research into their relevant to NVIDIA. Furthermore, based on the nature of our trading strategy (Trade close-to-open using feature returns), we wanted to limit to shares that were on the NYSE / NASDAQ or were able to be traded over the counter in the US exchange as well (such as Mercedes Group).
#
#
#
# ## Benchmarks
# - SOXX (Semiconductor ETF): This is the most direct benchmark we can use for NVDIA share performance. The index tracks the semiconductor industry, and this should be used to compare NVIDIA's performance directly against its closest peers (like AMD, Broadcom, etc.)
# - QQQ: This is a strong technology sector benchmark. It tracks the 100 largest non-financial companies on the Nasdaq. Since NVIDIA is a dominant holding here, this benchmark shows how NVIDIA is performing relative to the broader, large-cap tech sector.
# - SPY (S&P 500 ETF): This is a very broad market benchmark. It tracks the 500 largest U.S. companies across all sectors. Use this to see how NVIDIA is performing against the entire U.S. stock market.

# %%
# ============================================================
# Top-10 + 10 Cluster Reps -> Combine -> Corr Matrix -> Prune
# Residualize on SPY only (market-only), save CSVs
# ============================================================

import time, warnings, numpy as np, pandas as pd, yfinance as yf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Config ----------------
START, END = "2023-07-01", "2025-06-30"
TARGET = "NVDA"
MARKET = "SPY"           # market-only residualization
TOP_K = 10               # 1) Top-10 by composite
N_REPS = 10              # 2) 10 cluster reps
POOL_FOR_CLUSTER = 40    # cluster among Top-40 by composite (richer pool)
ROLL_WINDOW = 60         # rolling corr window for scoring
CORR_CAP = 0.85          # pruning threshold on |rho| (applied to SPY-residualized corr)
CSV_RAW   = "combined_corr_raw.csv"
CSV_RESID = "combined_corr_spy_resid.csv"

# ---------- Universe: US-traded (NYSE/NASDAQ) + US OTC ADRs ----------
SEED_ECONOMIC_US = sorted(set([
    # Cloud / AI infra
    "MSFT","GOOGL","AMZN","ORCL","AKAM","META","AAPL",
    # Telco
    "TMUS","T","NOK",
    # Auto / AV (US-traded lines incl. OTC ADRs)
    "UBER","STLA","LCID","MBGYY","LI","XPEV","NIO",
    # Hardware / systems
    "DELL","HPE","CSCO","SMCI","LNVGY",
    # Healthcare
    "JNJ","LLY","ILMN","IQV",
    # Software / manufacturing / twins adjacents
    "PLTR","LOW","SIEGY","TSM","HNHPF","CAT","DASTY",
    # Enterprise software
    "SNOW","NOW","SAP","ADBE","CRM","ANET",
    # Semis & equipment
    "AVGO","ASML","AMAT","KLAC","LRCX","MU",
    # Competitors
    "AMD","INTC",
    # NVDA direct investments
    "RXRX","SOUN",
    # Benchmarks / indices
    "SOXX","QQQ","SPY",
    # Target
    "NVDA"
]))
UNIVERSE = sorted(set(SEED_ECONOMIC_US))

# ---------- Robust download ----------
def fetch_one_strict(t, start, end):
    try:
        h = yf.Ticker(t).history(start=start, end=end, auto_adjust=False)
        if isinstance(h, pd.DataFrame) and not h.empty:
            return h[["Open","Adj Close"]].rename(columns={"Adj Close":"AdjClose"})
    except Exception:
        pass
    try:
        d = yf.download(t, start=start, end=end, auto_adjust=False,
                        actions=False, threads=False, progress=False)
        if isinstance(d, pd.DataFrame) and not d.empty:
            return d[["Open","Adj Close"]].rename(columns={"Adj Close":"AdjClose"})
    except Exception:
        pass
    try:
        w = yf.download(t, start=start, end=end, interval="1wk", auto_adjust=False,
                        actions=False, threads=False, progress=False)
        if isinstance(w, pd.DataFrame) and not w.empty:
            return w[["Open","Adj Close"]].rename(columns={"Adj Close":"AdjClose"}).resample("B").ffill()
    except Exception:
        pass
    return None

def download_prices_resilient(tickers, start, end, pause=0.5):
    opens, adjs, skipped = [], [], []
    for t in tickers:
        df = fetch_one_strict(t, start, end)
        if df is None or df.empty:
            print(f"[SKIP] {t}: could not fetch (dropped).")
            skipped.append(t)
        else:
            df = df.copy()
            df.columns = pd.MultiIndex.from_product([[t], df.columns])
            opens.append(df[(t, "Open")])
            adjs.append(df[(t, "AdjClose")])
        time.sleep(pause)
    if not adjs: raise RuntimeError("No price data downloaded.")
    open_all = pd.concat(opens, axis=1).sort_index()
    adj_all  = pd.concat(adjs,  axis=1).sort_index()
    open_all.columns = [c[0] for c in open_all.columns]
    adj_all.columns  = [c[0] for c in adj_all.columns]
    adj_all = adj_all.loc[:, adj_all.notna().sum() > 0]
    open_all = open_all.reindex(columns=adj_all.columns)
    return adj_all, open_all, skipped

print(f"[INFO] Downloading {len(UNIVERSE)} tickers...")
adj_close, open_px, skipped = download_prices_resilient(UNIVERSE, START, END)
if skipped: print(f"[INFO] Skipped {len(skipped)} tickers: {skipped}")

# Align and keep reliable columns
keep_cols = adj_close.columns[adj_close.notna().mean() >= 0.80]
adj_close = adj_close[keep_cols]
open_px   = open_px.reindex(index=adj_close.index, columns=adj_close.columns)
if TARGET not in adj_close.columns: raise RuntimeError("NVDA missing after download.")

idx_ok = adj_close[~adj_close[TARGET].isna()].index
adj_close = adj_close.loc[idx_ok]; open_px = open_px.loc[idx_ok]

# ---------- Returns ----------
ret_cc = np.log(adj_close / adj_close.shift(1)).dropna(how="all")   # C->C features at t
open_next = open_px.shift(-1)
ret_co = np.log(open_next / adj_close).dropna(how="all")            # C->O label/bench

y_nvda_co = ret_co[TARGET].dropna()
X_cc = ret_cc.reindex(y_nvda_co.index).dropna(how="all")

# ---------- Scoring (composite) ----------
def dcor(x, y):
    x = np.asarray(x).reshape(-1,1); y = np.asarray(y).reshape(-1,1)
    n = x.shape[0]
    if n < 3: return np.nan
    ax = np.abs(x - x.T); ay = np.abs(y - y.T)
    Ax = ax - ax.mean(0) - ax.mean(1, keepdims=True) + ax.mean()
    Ay = ay - ay.mean(0) - ay.mean(1, keepdims=True) + ay.mean()
    vx = np.sqrt((Ax*Ax).sum()/(n*n)); vy = np.sqrt((Ay*Ay).sum()/(n*n))
    vxy= (Ax*Ay).sum()/(n*n)
    if vx==0 or vy==0: return 0.0
    return float(vxy/(vx*vy))

def rolling_corr(x, y, w=ROLL_WINDOW, minp=30):
    s = pd.Series(x, index=y.index); t = pd.Series(y, index=y.index)
    return s.rolling(w, min_periods=minp).corr(t)

EXCLUDE = {TARGET, "SOXX", "QQQ", "SPY"}  # do not use indices as predictors
candidates = [c for c in X_cc.columns if c not in EXCLUDE]

XY = X_cc.join(y_nvda_co.rename("y"), how="inner").dropna(subset=["y"])
X = XY[candidates]; y = XY["y"]

rows = []
for c in candidates:
    xi = X[c].reindex(XY.index)
    try: r = pearsonr(xi.fillna(0), y.loc[xi.index].fillna(0))[0]
    except: r = np.nan
    rc = rolling_corr(xi, y).mean()
    mask = ~(xi.isna() | y.isna())
    mi  = mutual_info_regression(xi[mask].values.reshape(-1,1), y[mask].values, random_state=0)
    mi  = float(mi[0]) if mi.size else np.nan
    dcr = dcor(xi[mask].values, y[mask].values) if mask.sum()>=60 else np.nan
    rows.append({"ticker": c, "pearson": r, "rollcorr": rc, "mi": mi, "dcor": dcr})

score_df = pd.DataFrame(rows).set_index("ticker")
def zscore(s): s = s.replace([np.inf,-np.inf], np.nan); return (s - s.mean())/(s.std(ddof=1)+1e-12)
score_df["z_pearson"]  = zscore(score_df["pearson"])
score_df["z_rollcorr"] = zscore(score_df["rollcorr"])
score_df["z_mi"]       = zscore(score_df["mi"])
score_df["z_dcor"]     = zscore(score_df["dcor"])
score_df["composite"]  = score_df[["z_pearson","z_rollcorr","z_mi","z_dcor"]].mean(axis=1)

# ---------- (1) Top-10 ----------
math_ranked = score_df.sort_values("composite", ascending=False)
topk_list = list(math_ranked.head(TOP_K).index)
print("\n[Top-10 by composite]:", topk_list)

# ---------- (2) 10 cluster reps (from Top-POOL_FOR_CLUSTER; maxclust=10) ----------
pool = list(math_ranked.head(POOL_FOR_CLUSTER).index)
R_pool = X[pool].corr().fillna(0.0).clip(-1, 1)
D = np.sqrt(2*(1 - R_pool))
Z = linkage(squareform(D.values, checks=False), method="average")
labels = fcluster(Z, t=N_REPS, criterion="maxclust")

cluster_df = pd.DataFrame({"ticker": pool, "cluster": labels}).set_index("ticker")
cluster_df["composite"] = score_df["composite"].reindex(cluster_df.index)
rep_by_cluster = (cluster_df
                  .sort_values(["cluster","composite"], ascending=[True, False])
                  .groupby("cluster").head(1).index.tolist())
print("[10 Cluster Representatives]:", rep_by_cluster)

# ---------- Combine set ----------
combined = list(dict.fromkeys(topk_list + rep_by_cluster))
print(f"\n[Combined (Top-10 ∪ Reps), {len(combined)} names]:", combined)

# ---------- Corr matrices (RAW & SPY-residualized) ----------
def residualize_on_market(Xdf: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    m = mkt.loc[Xdf.index].astype(float).dropna()
    # align
    X_al = Xdf.loc[m.index].astype(float)
    C = np.c_[np.ones(len(m)), m.values]       # [1, SPY]
    Xt = X_al.values
    beta = np.linalg.pinv(C.T @ C) @ (C.T @ Xt)
    fitted = C @ beta
    resid = Xt - fitted
    return pd.DataFrame(resid, index=X_al.index, columns=X_al.columns)

X_comb = X_cc[[c for c in combined if c in X_cc.columns]].dropna(how="all")
R_raw = X_comb.corr()
R_raw.to_csv(CSV_RAW)
print(f"\n[Saved RAW corr matrix] -> {CSV_RAW} (shape={R_raw.shape})")

if MARKET in X_cc.columns:
    X_comb_resid = residualize_on_market(X_comb, X_cc[MARKET].dropna())
    R_resid = X_comb_resid.corr()
    R_resid.to_csv(CSV_RESID)
    print(f"[Saved SPY-residualized corr matrix] -> {CSV_RESID} (shape={R_resid.shape})")
else:
    X_comb_resid = X_comb.copy()
    R_resid = R_raw.copy()
    print("[WARN] SPY not available in X_cc; residualized matrix equals RAW.")

# ---------- Prune using SPY-residualized corr (|rho|>CORR_CAP) ----------
ranked_combined = sorted(combined, key=lambda t: score_df.loc[t, "composite"], reverse=True)
kept, dropped = [], []
for t in ranked_combined:
    if t not in X_comb_resid.columns:
        dropped.append((t, "missing_in_X"))
        continue
    if not kept:
        kept.append(t)
        continue
    rho_series = R_resid.loc[t, kept].abs()
    max_rho = rho_series.max() if len(rho_series) else 0.0
    if max_rho <= CORR_CAP:
        kept.append(t)
    else:
        against = rho_series.idxmax()
        dropped.append((t, f"|rho|={max_rho:.2f} vs {against} (> {CORR_CAP})"))

# ---------- Results ----------
print("\n================ RESULTS ================")
print(f"Kept ({len(kept)}): {kept}")
if dropped:
    print("\nDropped due to collinearity (kept higher composite counterpart):")
    for t, why in dropped:
        print(f" - {t}: {why}")
else:
    print("\nNo drops under current CORR_CAP.")

print("\nTop-10 (pre):", topk_list)
print("10 Reps     :", rep_by_cluster)
print("Combined pre:", combined)
print("Final kept  :", kept)
print(f"\nSettings -> CORR_CAP={CORR_CAP}, market_resid=SPY, pool_for_cluster={POOL_FOR_CLUSTER}")

