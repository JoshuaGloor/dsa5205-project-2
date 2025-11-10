from dataclasses import dataclass, field
from typing import Self, Callable
from enum import Enum, auto

import pandas as pd
import numpy as np
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential


class NoDataError(Exception):
    """Raised when data source returns an empty DataFrame."""

    pass


class MethodChainError(Exception):
    """Raised when method chaining not executed in proper order."""

    pass


class State(Enum):
    """Internal processing state of the DataLoader class.

    Members
    -------
    INIT : int
        The initial state before any data has been fetched.
    FETCHED : int
        Indicates that raw data has been successfully downloaded but not yet adjusted.
    ADJUSTED : int
        Indicates that the full adjustment pipeline has been executed and adjusted data is available.

    Notes
    -----
    This enumeration is used internally to enforce correct method call order
    within the data-loading and adjustment pipeline (e.g., preventing
    `adjust_data()` from running before `fetch_data()`).
    """

    INIT = auto()
    FETCHED = auto()
    ADJUSTED = auto()


def add_field(df: pd.DataFrame, field: str, base_field: str, func: Callable) -> pd.DataFrame:
    """Add a new top-level field to a DataFrame with MultiIndex columns.

    This function expects `df` to have MultiIndex columns of the form
    (Field, Ticker), where each `Field` corresponds to a data type (e.g.,
    "close", "adj_close", "volume") and each `Ticker` corresponds to a security.

    The specified `func` is applied column-wise to the sub-DataFrame
    corresponding to `base_field`, operating independently on each ticker's
    time series. The result is then added back to `df` as a new top-level
    field under `field`.

    Use this helper for transformations that operate **along the time axis**
    of each ticker, such as:
      - Shifting or lagging values
      - Cumulative or rolling computations
      - Elementwise transformations (e.g., log, scaling)

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with MultiIndex columns of the form (Field, Ticker).
    field : str
        The name of the new field to add at the top level of the column MultiIndex.
    base_field : str
        The existing field name (level 0 of the MultiIndex) to transform.
    func : Callable
        A function applied independently to each ticker's sub-series within `base_field`.

    Raises
    ------
    ValueError
        If `df` does not have MultiIndex columns or if `base_field` is not present
        in the first (Field) level of the MultiIndex.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new field joined into its MultiIndex columns.
    """

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns, but got a single-level Index.")

    if base_field not in df.columns.get_level_values(0).unique():
        raise ValueError(f"{base_field} is not a field in index level 0.")

    if field in df.columns.get_level_values(0).unique():
        raise ValueError(f"Field '{field}' already exists in data_cube.")

    df = df.join(
        df[base_field]
        .transform(func)
        .set_axis(
            pd.MultiIndex.from_product([[field], df[base_field].columns]),
            axis=1,
        )
    )
    return df


@dataclass
class DataLoader:
    r"""A loader for fetching, adjusting, and verifying OHLCV data.

    This class fetches raw price data from Yahoo Finance, computes adjustment
    factors for splits and dividends, constructs back-adjusted price series,
    and provides verification diagnostics and total-return series.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to fetch, e.g. `["AAPL", "NVDA", "MSFT"]`.
    start : str
        Start date (in `"YYYY-MM-DD"` format) for historical data retrieval.
    end : str or None, optional
        End date (in `"YYYY-MM-DD"` format). If `None`, data are fetched up to the
        most recent available trading date.

    Examples
    --------
    Basic usage::

        >>> from src.pipeline.loader import DataLoader
        >>> dl = DataLoader(["NVDA", "GOOG"], start="2010-01-01")

        # Fetch data from Yahoo Finance
        >>> dl.fetch_data()

        # Apply split and dividend adjustments
        >>> dl.adjust_data()

        # Retrieve fully adjusted total-return data
        >>> df_true = dl.get_adjusted_data("true")
        >>> df_true[("total_return", "NVDA")].tail()

        # Verify alignment and approximation accuracy
        >>> mad_table, event_table = dl.verify_data()
    """

    tickers: list[str]
    start: str  # "yyyy-mm-dd"
    end: str | None = None  # "yyyy-mm-dd" or None, which will return up to latest data available.
    _state: State = field(init=False, default=State.INIT)

    def fetch_data(self) -> Self:
        """Retrieve both raw and adjusted historical data for all tickers.

        This method downloads ticker data twice using internal method:
        once with `adjusted=False` to obtain raw (split adjusted but not dividend adjusted) OHLCV data and
        once with `adjusted=True` to obtain Yahoo Finance's auto-adjusted data.
        The two versions are stored in:
          - `self.raw` : unadjusted data (includes dividends and stock splits)
          - `self.adj` : adjusted data (all prices already adjusted by Yahoo)

        Fetching both datasets is necessary because Yahoo Finance's `auto_adjust=True`
        option modifies not only the closing prices but also the open, high, and low
        fields, making it unsuitable for analyses that require access to both
        unadjusted and adjusted price series.

        Returns
        -------
        Self
            The same instance, with `self.raw` and `self.adj` populated as
            MultiIndex DataFrames indexed by date and structured as `["Field", "Ticker"]`.
        """

        self._state = State.FETCHED

        self.raw = self._fetch_tickers(adjusted=False)
        self.adj = self._fetch_tickers(adjusted=True)
        return self

    def adjust_data(self) -> None:
        r"""Execute the full price-adjustment and total-return pipeline for all tickers.

        This method transforms raw, unadjusted prices into fully split- and
        dividend-adjusted OHLC series and then computes corresponding total
        return series for different adjustment conventions.

        The adjustment pipeline proceeds through the following steps:

            1. `_fsplit()` : adds the split adjustment factor $f_t^{\text{split}}$
            2. `_fdiv()`   : adds the dividend adjustment factor $f_t^{\text{div}}$
            3. `_gt()`     : combines both into the total adjustment factor $g_t$
            4. `_cum_gt()` : computes the cumulative product $\prod_{k > t} g_k$
            5. `_adj()`    : constructs the back-adjusted OHLC price $A_t = P_t \cdot \prod_{k > t} g_k$

        After constructing the adjusted price series, the method computes
        total return series R_t^{\text{TR}} along the time axis via

        $$
        R_t^{\text{TR}} = \frac{A_t}{A_{t - 1}} - 1,
        $$

        where $A_t$ is the adjusted close at day $t$.
        """

        if self._state != State.FETCHED:
            raise RuntimeError("Data is not fetched yet, call `fetch_data` first.")
        if self._state == State.ADJUSTED:
            raise RuntimeError("Data is adjusted already.")
        self._state = State.ADJUSTED

        # Pipeline to adjust data
        self._fsplit()._fdiv()._gt()._cum_gt()._adj()

        # Calculate R_t^{\text{TR}} for Yahoo provided data
        self.adj = add_field(self.adj, "total_return", "close", lambda t: t.pct_change())

        # Calculate R_t^{\text{TR}} for raw data
        self.raw = add_field(self.raw, "total_return_true", "adj_close_true", lambda t: t.pct_change())
        self.raw = add_field(self.raw, "total_return_manual", "adj_close_manual", lambda t: t.pct_change())

    def get_adjusted_data(self, source="true") -> pd.DataFrame:
        r"""Return adjusted OHLCV and total-return data for all tickers.

        This method provides access to fully adjusted price data constructed
        according to the specified adjustment source. Depending on the source,
        it returns either Yahoo Finance's built-in adjusted series, manually
        recomputed series using Yahoo's adjustment logic, or internally derived
        "true" adjusted series based on the exact adjustment factors.

        The adjusted dataset contains OHLCV fields and the corresponding total
        return series:

        $$
        R_t^{\text{TR}} = \frac{A_t}{A_{t-1}} - 1,
        $$

        where $A_t$ is the adjusted close at day $t$.

        Parameters
        ----------
        source : {"true", "yahoo", "manual"}, default "true"
            The adjustment source to use:

              - "true"
                Returns correctly adjusted data computed internally from raw
                prices using exact (non-approximated) adjustment factors.

              - "yahoo"
                Returns Yahoo Finance's pre-adjusted OHLC data as provided by
                the data vendor.

              - "manual"
                Returns manually adjusted data reconstructed using Yahoo's own
                approximate adjustment logic, intended to match the "yahoo"
                series up to machine precision.

        Returns
        -------
        pd.DataFrame
            DataFrame containing adjusted OHLCV and total-return data for all
            tickers.

            - Index: trading dates.
            - Columns: MultiIndex with levels `["Field", "Ticker"]`, where
              `"Field"` includes
              `["open", "high", "low", "close", "volume", "total_return"]`.

        Raises
        ------
        ValueError
            If `source` is not one of {"true", "yahoo", "manual"}.
        RuntimeError
            If the data have not yet been adjusted (i.e., `adjust_data()` was not called).
        """

        sources = ["true", "yahoo", "manual"]
        if source not in sources:
            raise ValueError(f"Invalid source '{source}'. Must be one of {sources}.")

        if self._state != State.ADJUSTED:
            raise RuntimeError("Data is not adjusted yet, call `adjust_data` first.")

        # Easy case, just return the yahoo adjusted data.
        if source == "yahoo":
            return self.adj.copy()

        ohlc = ["open", "high", "low", "close"]
        # Prepare for renaming, e.g., we will rename 'adj_open_true' to 'open'.
        fields_old_new = {f"adj_{f}_{source}": f for f in ohlc}
        # Add total return field
        fields_old_new[f"total_return_{source}"] = "total_return"

        # Copy the old fields together with 'volume'
        df = self.raw[[*fields_old_new.keys(), "volume"]].copy()

        # Set column names of MultiIndex
        df.columns.names = ["OHLCV-TR", "Ticker"]

        # Return OHLCV and total return of desired source
        return df.rename(columns=fields_old_new, level=0)

    def verify_data(self, ticker="") -> tuple[pd.DataFrame]:
        r"""Verify correctness of the adjusted-price and total-return construction.

        This method validates the internal price-adjustment and total-return
        pipeline by comparing results from different adjustment sources and
        inspecting alignment around corporate-action events such as dividends
        and stock splits.

        Two forms of verification are performed:

        1. **Return continuity (mean absolute difference):**
            Computes the mean absolute differences (MAD) of total return series between:

            - *Yahoo vs. True*:
              Quantifies the deviation introduced by Yahoo's approximate
              dividend-adjustment logic compared to the internally computed
              “true” adjustment factors.

            - *Yahoo vs. Manual*:
              Compares Yahoo's own adjusted data to a manual recomputation of
              Yahoo's adjustment logic. Differences should be within machine precision.

        2. **Event alignment inspection:**
           Selects a representative ticker (preferring *NVDA* if available)
           and extracts dividend and split events.
           For each event, the method compiles a small table containing the
           event type and surrounding dates (pre-/post-event).

        These diagnostics allow visual and numerical verification that
        (i) Yahoo's approximated adjustment factors differ only negligibly
        from the mathematically correct ones, and
        (ii) events are properly aligned in the adjustment pipeline.

        Parameters
        ----------
        ticker : str,  default ""
            The ticker for which the verification should be performed.
            - If specified and the ticker exists, that one is chosen.
            - Otherwise "NVDA" is selected if it exists and otherwise
              the ticker which is the first one in the level 1 index.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            A pair of diagnostic tables:

            **(1) df_mad**
                Summary table of mean absolute differences (MAD).
                Columns include:

                - `"MAD Yahoo and True"` - Mean absolute difference between Yahoo's
                  and true total-return series.
                - `"MAD Yahoo and Manual Recomputation"` - Mean absolute difference
                  between Yahoo's and manually recomputed total-return series.

            **(2) df_events**
                Event-level alignment table for one representative ticker.
                Indexed by event dates and containing columns such as:

                - `"Event"`: Event type ("split", "ex-dividend", "pre-split", ...)
                - `"Dividends"`: Dividend amounts
                - `"f_div (True)"` and `"f_div (Yahoo Approx)"`: Dividend adjustment factors
                - `"Prod_{k > t} g_t (True)"` and `"Prod_{k > t} g_t (Yahoo Approx)"`:
                  cumulative adjustment products
                - `"Stock Splits"`: Split ratios
                - `"f_split"`: Split adjustment factor; always 1 because Yahoo data is split adjusted.
                - `"Total Return (True)"`, `"Total Return (Yahoo)"`, `"Total Return (Yahoo Manual)"`:
                  aligned total-return series

        Raises
        ------
        RuntimeError
            If data have not yet been adjusted (`adjust_data()` not called).
        ValueError
            If ticker selected but not part of fetched data.
        """

        if self._state != State.ADJUSTED:
            raise RuntimeError("Data is not adjusted yet, call `adjust_data` first.")
        if ticker != "" and ticker not in self.raw.columns.get_level_values(1).unique():
            raise ValueError("{ticker} not in fetched data.")

        true = self.get_adjusted_data("true")
        yahoo = self.get_adjusted_data("yahoo")
        manual = self.get_adjusted_data("manual")

        # ---------- Mean absolute difference checks ---------- #
        # Calculate mean absolute difference (MAD) between our true return and Yahoo's.
        # This shows the difference between Yahoo's approximated dividend adjustement factor and the true one.
        # This is **NOT** expected to be within machine precision!
        yahoo_true_diff = (yahoo["total_return"] - true["total_return"]).abs()
        mad_yahoo_true = yahoo_true_diff.mean()

        # Calculate mean absolute difference (MAD) between manual recalculation of Yahoo's data and Yahoo's.
        # This is expected to be machine precision.
        yahoo_manual_diff = (yahoo["total_return"] - manual["total_return"]).abs()
        mad_yahoo_manual = yahoo_manual_diff.mean()

        df_mad = pd.DataFrame(
            [mad_yahoo_true, mad_yahoo_manual], index=["MAD Yahoo and True", "MAD Yahoo and Manual Recomputation"]
        ).T

        # ---------- Event alignment checks ---------- #
        # Choose ticker for event inspection.
        # If ticker not selected, we choose NVDA if it is part of the tickers
        # because it is our focus and it had both dividends and splits in recent times.
        if ticker == "":
            tickers = self.raw["close"].columns
            if "NVDA" in tickers:
                ticker = "NVDA"
            else:
                ticker = tickers[0]

        # Extract per-ticker series of return.
        return_true = true[("total_return", ticker)]
        return_yahoo = yahoo[("total_return", ticker)]
        return_manual = manual[("total_return", ticker)]

        # Extract interesting columns.
        f_split = self.raw[("f_split", ticker)]
        f_div_true = self.raw[("f_div", ticker)]
        f_div_yahoo = self.raw[("f_div_yahoo", ticker)]
        cum_gt_true = self.raw[("cum_gt", ticker)]
        cum_gt_yahoo = self.raw[("cum_gt_yahoo", ticker)]

        # Extract dividends and stock_splits.
        div = self.raw[("dividends", ticker)]
        splits = self.raw[("stock_splits", ticker)]

        # Find event dates
        div_dates = div[div.fillna(0) != 0].index
        split_dates = splits[splits.fillna(0) != 0].index

        # Pick one representative dividend and one split if they exist.
        # Oldest will be the most interesting for both cases since we back-adjust.
        dates_to_types = {}
        if len(split_dates) > 0:
            d = split_dates[0]
            dates_to_types[d] = "split"

            # Integer position of `d` in the index.
            pos = self.raw.index.get_loc(d)

            # Add previous and next index of event if within bounds.
            if pos > 0:
                dates_to_types[self.raw.index[pos - 1]] = "pre-split"
            if pos < len(self.raw.index) - 1:
                dates_to_types[self.raw.index[pos + 1]] = "post-split"

        if len(div_dates) > 0:
            d = div_dates[0]
            dates_to_types[d] = "ex-dividend"

            # Integer position of `d` in the index.
            pos = self.raw.index.get_loc(d)

            # Add previous and next index if within bounds.
            if pos > 0:
                dates_to_types[self.raw.index[pos - 1]] = "pre-dividend"
            if pos < len(self.raw.index) - 1:
                dates_to_types[self.raw.index[pos + 1]] = "post-dividend"

        if dates_to_types:
            # Create index of event dates
            sorted_dates_to_types = dict(sorted(dates_to_types.items()))
            idx = pd.Index(sorted_dates_to_types.keys())

            df_events = pd.DataFrame(
                {
                    "Event": sorted_dates_to_types.values(),
                    "Dividends": div.reindex(idx),
                    "f_div (True)": f_div_true.reindex(idx),
                    "f_div (Yahoo Approx)": f_div_yahoo.reindex(idx),
                    r"Prod_{k > t} g_t (True)": cum_gt_true.reindex(idx),
                    r"Prod_{k > t} g_t (Yahoo Approx)": cum_gt_yahoo.reindex(idx),
                    "Stock Splits": splits.reindex(idx),
                    "f_split (1; All Yahoo Data is Split Adjusted)": f_split.reindex(idx),
                    "Total Return (True)": return_true.reindex(idx),
                    "Total Return (Yahoo)": return_yahoo.reindex(idx),
                    "Total Return (Yahoo Manual)": return_manual.reindex(idx),
                },
                index=idx,
            )
        else:
            # No events found for that ticker; return empty alignment table.
            df_events = pd.DataFrame(
                columns=[
                    "Event",
                    "Dividends",
                    "f_div (True)",
                    "f_div (Yahoo Approx)",
                    r"Prod_{k > t} g_t (True)",
                    r"Prod_{k > t} g_t (Yahoo)",
                    "Stock Splits",
                    "f_split (1; All Yahoo Data is Split Adjusted)",
                    "Total Return (True)",
                    "Total Return (Yahoo)",
                    "Total Return (Yahoo Manual)",
                ]
            )
        df_events.index.name = "Date"
        df_events.columns = pd.MultiIndex.from_product([[ticker], df_events.columns], names=["Ticker", "Diagnostics"])

        return df_mad, df_events

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def _fetch_tickers(self, adjusted=True) -> pd.DataFrame:
        """Download historical price data for all tickers in the current instance.

        This method retrieves OHLCV (Open, High, Low, Close, Volume) data — and optionally
        corporate actions (dividends and stock splits) — from Yahoo Finance using `yfinance.download`.
        The resulting DataFrame has a two-level column MultiIndex with levels ["Field", "Ticker"], where the top level
        represents data fields (e.g. "close", "volume") and the second level represents individual tickers.

        Parameters
        ----------
        adjusted : bool, default True
            Whether to request adjusted prices directly from Yahoo Finance.
            - If True, returns prices with corporate actions already adjusted, including the fields:
              ["adj_close", "close", "dividends", "high", "low", "open", "stock_splits", "volume"].
            - If False, returns raw (split adjusted but not dividend adjusted) prices
              plus corporate action data for manual adjustment, with the fields:
              ["close", "high", "low", "open", "volume"].

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with:
            - Index: DatetimeIndex of trading days.
            - Columns: MultiIndex with level names ["Field", "Ticker"].

        Raises
        ------
        NoDataError
            If Yahoo Finance returns an empty dataset

        Notes
        -----
        The column names are normalized to lowercase and have spaces replaced with underscores for consistency.
        The resulting structure is designed for downstream multi-ticker processing within the loader.
        """

        df = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=adjusted,
            actions=not adjusted,  # Only need dividend and split data if close is not auto adjusted.
            progress=False,
        )

        if df.empty:
            raise NoDataError(f"No data returned for ticker list {self.tickers}")

        df.columns = df.columns.set_names(["Field", "Ticker"])

        # Rename fields to lower and replace whitespace with und underscore
        df.columns = df.columns.map(lambda t: (t[0].lower().replace(" ", "_"), t[1]))

        return df

    def _require_fields(self, fields: list[str]):
        """Helper to verify required fields are present"""

        missing = [f for f in fields if f not in self.raw.columns.get_level_values(0)]
        if missing:
            raise MethodChainError(f"Missing required fields: {missing}")

    def _fsplit(self) -> Self:
        r"""Add the stock-split adjustment factor $f_t^{\text{split}}$ to each ticker.

        The split adjustment factor is defined as:

        $$
        f_t^{\text{split}} =
        \begin{cases}
            \frac{1}{S_t}, & \text{if split on } t, \\
            1, & \text{otherwise}.
        \end{cases}
        $$

        where $S_t$ denotes the split ratio (e.g., $S_t = 2$ for a 2-for-1 split).

        Notes
        -----
        Yahoo Finance already provides split-adjusted daily closing prices.
        Therefore, this function currently sets $f_t^{\text{split}} = 1$ for all dates.
        The conditional logic is preserved for reference, illustrating how the
        computation would proceed if unadjusted prices were available.
        """

        # Daily closing prices from Yahoo are already split-adjusted.
        # Yahoo Finance does not provide truly non-split-adjusted prices!
        # See, for example: https://github.com/ranaroussi/yfinance/discussions/1682
        if True:
            # We preserve this branch to document how the logic would work if non-split-adjusted data were available.
            ones = pd.DataFrame(
                1.0,
                index=self.raw.index,
                columns=pd.MultiIndex.from_product([["f_split"], self.raw.columns.levels[1]]),
            )

            self.raw = self.raw.join(ones)
        else:  # This branch should be executed if true unadjusted prices were available.
            self.raw = add_field(self.raw, "f_split", "stock_splits", lambda t: np.where(t.fillna(0) != 0, 1 / t, 1.0))
        return self

    def _fdiv(self) -> Self:
        r"""Add the dividend adjustment factor $f_t^{\text{div}}$ to each ticker.

        The dividend adjustment factor compensates for cash dividends, ensuring that
        price changes reflect only market movements rather than mechanical drops from
        dividend payouts. It is defined as:

        $$
        f_t^{\text{div}} =
        \begin{cases}
            \dfrac{1}{1 + D_t / P_{t-1}}, & \text{if a cash dividend on } t, \\
            1, & \text{otherwise}.
        \end{cases}
        $$

        where:
          - $D_t$ is the cash dividend paid at time $t$ (ex-dividend date),
          - $P_{t-1}$ is the previous day's closing price.

        Notes
        -----
        Yahoo Finance applies a simplified approximation:
        $$
        f_t^{\text{div, Yahoo}} = 1 - \frac{D_t}{P_{t-1}},
        $$
        which is a first-order Taylor approximation of the theoretically correct formula.

        This method computes both versions:
          - `f_div`: the exact adjustment factor
          - `f_div_yahoo`: Yahoo's approximate version
        """

        # Previous day's close per ticker
        self.raw = add_field(self.raw, "prev_close", "close", lambda t: t.shift(1))

        # Dividends per ticker, fill missing with 0
        self.raw = add_field(self.raw, "div_clean", "dividends", lambda t: t.fillna(0.0))

        # Dividend adjustment factor f_div = 1 / (1 + D_t / P_{t - 1}) = P_{t - 1} / (P_{t - 1} + D_t)
        f_div = self.raw["prev_close"] / (self.raw["prev_close"] + self.raw["div_clean"])

        # Yahoo uses an approximation: f_div_yahoo = 1 - D_t / P_{t - 1}
        f_div_yahoo = 1 - self.raw["div_clean"] / self.raw["prev_close"]

        # Add new columns back to MultiIndex
        f_div.columns = pd.MultiIndex.from_product([["f_div"], f_div.columns])
        f_div_yahoo.columns = pd.MultiIndex.from_product([["f_div_yahoo"], f_div_yahoo.columns])
        self.raw = self.raw.join([f_div, f_div_yahoo])

        # Drop helper columns
        self.raw = self.raw.drop(columns="prev_close", level=0)
        self.raw = self.raw.drop(columns="div_clean", level=0)

        return self

    def _gt(self) -> Self:
        r"""Add the combined adjustment factor $g_t$ to each ticker.

        The factor $g_t$ accounts for both stock splits and cash dividends, combining
        their effects into a single multiplicative adjustment used to create
        split and dividend-adjusted price series. It is defined as:

        $$
        g_t = f_t^{\text{split}} \cdot f_t^{\text{div}}.
        $$

        where:
          - $f_t^{\text{split}}$ is the split adjustment factor,
          - $f_t^{\text{div}}$ is the dividend adjustment factor.

        Notes
        -----
        - This function computes two variants:
            - `gt` : uses the exact dividend factor $f_t^{\text{div}}$,
            - `gt_yahoo` : uses Yahoo's approximate dividend factor $f_t^{\text{div, Yahoo}}$.
        """

        self._require_fields(["f_split", "f_div", "f_div_yahoo"])

        # Calculate g_t
        gt = self.raw["f_split"] * self.raw["f_div"]
        gt_yahoo = self.raw["f_split"] * self.raw["f_div_yahoo"]

        # Add new columns back to MultiIndex
        gt.columns = pd.MultiIndex.from_product([["gt"], gt.columns])
        gt_yahoo.columns = pd.MultiIndex.from_product([["gt_yahoo"], gt_yahoo.columns])
        self.raw = self.raw.join([gt, gt_yahoo])

        return self

    def _cum_gt(self) -> Self:
        r"""Add the cumulative adjustment factor $\prod_{k > t} g_k$ to each ticker.

        The cumulative product of adjustment factors is used to construct
        *back-adjusted* price series, where past prices are rescaled to be
        comparable with current ones. The back-adjusted price is defined as:

        $$
        A_t = P_t \cdot \prod_{k > t} g_k,
        $$

        where:
          - $P_t$ is the unadjusted closing price at time $t$,
          - $g_k = f_k^{\text{split}} \cdot f_k^{\text{div}}$ is the combined adjustment factor for day $k$.

        Notes
        -----
        - Two variants are created:
            - `cum_gt`: using the exact dividend adjustment factor,
            - `cum_gt_yahoo`: using Yahoo's approximate dividend adjustment.
        """

        self._require_fields(["gt", "gt_yahoo"])

        # This cum product gives us product k >= t (greater equal, qe), we have to adjust to k > t later
        self.raw = add_field(self.raw, "cum_gt_qe", "gt", lambda t: t.fillna(1).iloc[::-1].cumprod().iloc[::-1])
        self.raw = add_field(
            self.raw, "cum_gt_yahoo_qe", "gt_yahoo", lambda t: t.fillna(1).iloc[::-1].cumprod().iloc[::-1]
        )

        # Adjust to k > t
        cum_gt = self.raw["cum_gt_qe"] / self.raw["gt"]
        cum_gt_yahoo = self.raw["cum_gt_yahoo_qe"] / self.raw["gt_yahoo"]

        # Add new columns back to MultiIndex
        cum_gt.columns = pd.MultiIndex.from_product([["cum_gt"], cum_gt.columns])
        cum_gt_yahoo.columns = pd.MultiIndex.from_product([["cum_gt_yahoo"], cum_gt_yahoo.columns])
        self.raw = self.raw.join([cum_gt, cum_gt_yahoo])

        # Drop helper columns
        self.raw = self.raw.drop(columns="cum_gt_qe", level=0)
        self.raw = self.raw.drop(columns="cum_gt_yahoo_qe", level=0)

        return self

    def _adj(self):
        r"""Add the back-adjusted price $A_t$ to each ticker.

        The *back-adjusted* price rescales historical prices to account for
        all subsequent corporate actions (splits and dividends), ensuring
        continuity in the price series. It is defined as:

        $$
        A_t = P_t \cdot \prod_{k > t} g_k
        $$

        Notes
        -----
        - Two versions of the adjusted prices are added. for f \in {open, high, low, close}:
            - `adj_f_true`: using the exact adjustment factor.
            - `adj_f_manual`: using Yahoo's approximate dividend adjustment.
        """

        self._require_fields(["cum_gt", "cum_gt_yahoo"])
        self._adjusted = True

        # Calculate adjusted prices
        adj_open_true = self.raw["open"] * self.raw["cum_gt"]
        adj_high_true = self.raw["high"] * self.raw["cum_gt"]
        adj_low_true = self.raw["low"] * self.raw["cum_gt"]
        adj_close_true = self.raw["close"] * self.raw["cum_gt"]

        adj_open_manual = self.raw["open"] * self.raw["cum_gt_yahoo"]
        adj_high_manual = self.raw["high"] * self.raw["cum_gt_yahoo"]
        adj_low_manual = self.raw["low"] * self.raw["cum_gt_yahoo"]
        adj_close_manual = self.raw["close"] * self.raw["cum_gt_yahoo"]

        # Add new column back to MultiIndex
        adj_open_true.columns = pd.MultiIndex.from_product([["adj_open_true"], adj_open_true.columns])
        adj_high_true.columns = pd.MultiIndex.from_product([["adj_high_true"], adj_high_true.columns])
        adj_low_true.columns = pd.MultiIndex.from_product([["adj_low_true"], adj_low_true.columns])
        adj_close_true.columns = pd.MultiIndex.from_product([["adj_close_true"], adj_close_true.columns])

        adj_open_manual.columns = pd.MultiIndex.from_product([["adj_open_manual"], adj_open_manual.columns])
        adj_high_manual.columns = pd.MultiIndex.from_product([["adj_high_manual"], adj_high_manual.columns])
        adj_low_manual.columns = pd.MultiIndex.from_product([["adj_low_manual"], adj_low_manual.columns])
        adj_close_manual.columns = pd.MultiIndex.from_product([["adj_close_manual"], adj_close_manual.columns])

        self.raw = self.raw.join(
            [
                adj_open_true,
                adj_high_true,
                adj_low_true,
                adj_close_true,
                adj_open_manual,
                adj_high_manual,
                adj_low_manual,
                adj_close_manual,
            ]
        )

        return self
