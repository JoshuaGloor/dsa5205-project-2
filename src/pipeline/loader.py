from dataclasses import dataclass
from typing import Self, Callable, Sequence

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


@dataclass
class DataLoader:
    tickers: list[str]
    start: str  # "yyyy-mm-dd"
    end: str | None = None  # "yyyy-mm-dd" or None, which will return up to latest data available.

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2))
    def _fetch_tickers(self, adjusted=True) -> pd.DataFrame:
        df = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=adjusted,
            actions=not adjusted,
            progress=False,
        )

        if df.empty:
            raise NoDataError(f"No data returned for ticker list {self.tickers}")

        df.columns = df.columns.set_names(["Field", "Ticker"])

        # Rename fields to lower and replace whitespace with und underscore
        df.columns = df.columns.map(lambda t: (t[0].lower().replace(" ", "_"), t[1]))

        # We don't need high and low
        df = df.drop(columns=["high", "low"], level=0)
        return df

    def fetch_data(self) -> Self:
        """Helper to fetch raw and adj data.

        While the yfinance API with actions=True and auto_adjust=True returns both
        adjusted and raw columns in one call, we keep them separate as suggested in the exercise sheet.
        """

        self.raw = self._fetch_tickers(adjusted=False)
        # Drop adj close, we do not need it.
        self.raw = self.raw.drop(columns="adj_close", level=0)

        self.adj = self._fetch_tickers(adjusted=True)
        return self

    def _add_field_row_op(self, field: str, base_field: str, func: Callable) -> Self:
        """Add a new column to raw MultiIndex with transformation on row level.

        This helper adds the new field 'field' to the MultiIndex.
        The 'func' is a function that operatates on *rows*.

        Examples:
        - shift over time -> column level (this is the wrong function)
        - transform each row -> row level (this is the correct function)
        """

        self.raw = self.raw.join(
            self.raw[base_field]
            .T.groupby(level="Ticker")
            .transform(func)
            .T.set_axis(
                pd.MultiIndex.from_product([[field], self.raw[base_field].columns]),
                axis=1,
            )
        )
        return self

    def _add_field_col_op(self, field: str, base_field: str, func: Callable) -> Self:
        """Add a new column to raw MultiIndex with transformation on col level.

        This helper adds the new field 'field' to the MultiIndex.
        The 'func' is a function that operates on *cols*.

        Examples:
        - shift over time -> column level (this is the correct function)
        - transform each row -> row level (this is the wrong function)
        """

        self.raw = self.raw.join(
            self.raw[base_field]
            .transform(func)
            .set_axis(
                pd.MultiIndex.from_product([[field], self.raw[base_field].columns]),
                axis=1,
            )
        )
        return self

    def _fsplit(self) -> Self:
        r"""Add $f_t^{split}$ to each ticker."""

        # See e.g. this discussion: https://github.com/ranaroussi/yfinance/discussions/1682
        if "Daily closing price is split adjusted already, Yahoo does not provide non-split adjusted prices!":
            # We want to preserve the rest of the logic of how it could be done if we had non-split adjusted data.
            self._add_field_row_op("f_split", "stock_splits", lambda _: 1)  # Dummy column of ones.
        else:  # We would execute this if we had true unadjusted prices.
            self._add_field_row_op("f_split", "stock_splits", lambda s_t: np.where(s_t.fillna(0) != 0, 1 / s_t, 1.0))
        return self

    def _fdiv(self) -> Self:
        r"""Add $f_t^{div}$ to each ticker."""

        # Previous day's close per ticker
        self._add_field_col_op("prev_close", "close", lambda t: t.shift(1))

        # Dividends per ticker, fill missing with 0
        self._add_field_col_op("div_clean", "dividends", lambda t: t.fillna(0.0))

        # Dividend adjustment factor f_div = P_{t - 1} / (P_{t - 1} + D_t)
        # f_div = self.raw["prev_close"] / (self.raw["prev_close"] + self.raw["div_clean"])
        f_div = self.raw["close"] / (self.raw["close"] + self.raw["div_clean"])

        # Add new column back to MultiIndex
        f_div.columns = pd.MultiIndex.from_product([["f_div"], f_div.columns])
        self.raw = self.raw.join(f_div)

        # Drop helper columns
        self.raw = self.raw.drop(columns="prev_close", level=0)
        self.raw = self.raw.drop(columns="div_clean", level=0)

        return self

    def _gt(self) -> Self:
        r"""Add $g_t = f_t^{split} * f_t^{div}$ to each ticker."""

        existing_fields = self.raw.columns.get_level_values(0)
        if "f_split" not in existing_fields or "f_div" not in existing_fields:
            raise MethodChainError(f"'f_div' and 'f_split' must exist before creating 'gt'")

        # Calculate g_t
        gt = self.raw["f_split"] * self.raw["f_div"]

        # Add new column back to MultiIndex
        gt.columns = pd.MultiIndex.from_product([["gt"], gt.columns])
        self.raw = self.raw.join(gt)

        return self

    def _cum_gt(self) -> Self:
        r"""Add $\product_{k > t}g_k$ to each ticker."""

        if "gt" not in self.raw.columns.get_level_values(0):
            raise MethodChainError(f"'gt' must exist before creating 'cum_gt'")

        # This cum product gives us product k >= t (greater equal, qe), we have to adjust to k > t later
        self._add_field_col_op("cum_gt_qe", "gt", lambda t: t.fillna(1).iloc[::-1].cumprod().iloc[::-1])

        # Adjust to k > t
        cum_gt = self.raw["cum_gt_qe"] / self.raw["gt"]

        # Add new column back to MultiIndex
        cum_gt.columns = pd.MultiIndex.from_product([["cum_gt"], cum_gt.columns])
        self.raw = self.raw.join(cum_gt)

        # Drop helper column
        self.raw = self.raw.drop(columns="cum_gt_qe", level=0)

        return self

    def _adj(self):
        r"""Add $A_t = P_t * \product_{k > t}g_k$ to each ticker."""

        if "cum_gt" not in self.raw.columns.get_level_values(0):
            raise MethodChainError(f"'cum_gt' must exist before creating 'adj_close_manual'")

        # Calculate adj_close_manual
        adj = self.raw["close"] * self.raw["cum_gt"]

        # Add new column back to MultiIndex
        adj.columns = pd.MultiIndex.from_product([["adj_close_manual"], adj.columns])
        self.raw = self.raw.join(adj)

        return self

    def verify_data(self) -> pd.DataFrame:
        """Constructs and verifies back adjustment.

        Follows recommendations given in Appendix B of exercise sheet.
        """

        if not hasattr(self, "raw"):
            raise RuntimeError("Call 'fetch_data' first before you call 'verify_data'")

        # Compute back adjusted total return
        self._fsplit()._fdiv()._gt()._cum_gt()._adj()

        # Calculate mean absolute difference (MAD) between our adj close and Yahoo's
        abs_diff = (self.raw["adj_close_manual"] - self.adj["close"]).abs()
        mad = abs_diff.mean()

        return pd.DataFrame(mad, columns=["mean_abs_diff"]).T
