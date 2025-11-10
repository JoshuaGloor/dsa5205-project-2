import numpy as np
import pandas as pd
from functools import wraps

from src.pipeline.loader import add_field


def validate_data_cube(data_cube: pd.DataFrame) -> None:
    """Validate that `data_cube` has MultiIndex columns suitable for an OHLCV/returns/features cube.

    Requirements:
    - Columns must be a MultiIndex and
    - it must have 2 levels
    """

    cols = data_cube.columns

    if not isinstance(cols, pd.MultiIndex):
        raise TypeError(
            "Expected MultiIndex columns where first level are fields ('open', 'volume', etc.) and second level are tickers."
        )

    if cols.nlevels != 2:
        raise ValueError(f"Expected a 2-level MultiIndex, got {cols.nlevels}.")


def requires_data_cube(func):
    @wraps(func)
    def wrapper(data_cube: pd.DataFrame, *args, **kwargs):
        validate_data_cube(data_cube)
        return func(data_cube, *args, **kwargs)

    return wrapper


@requires_data_cube
def add_log_return(data_cube: pd.DataFrame) -> pd.DataFrame:
    r"""Add log returns computed from total returns.

    The log return is defined as:

    $$
    R_t^{\log} = \log(1 + R_t^{\text{TR}})
    $$

    Parameters
    ----------
    data_cube : pd.DataFrame
        Input data cube with a field "total_return".

    Returns
    -------
    pd.DataFrame
        Original data with a new field "log_return".
    """

    return add_field(data_cube, "log_return", "total_return", lambda t: np.log1p(t))


@requires_data_cube
def add_close_to_open_return(data_cube: pd.DataFrame) -> pd.DataFrame:
    r"""Add close-to-open (overnight) log return.

    The close-to-open return measures the price change between a day's
    adjusted close and the following day's adjusted open:

    $$
    R_t^{\text{CO}} = \log(\frac{O_{t+1}}{A_t})
    $$

    Parameters
    ----------
    data_cube : pd.DataFrame
        Input data cube with fields "close" and "open".

    Returns
    -------
    pd.DataFrame
        Original data augmented with a new field "log_ret_co".
    """

    open_next = data_cube["open"].shift(-1)
    log_ret_co = np.log(open_next / data_cube["close"])

    # Add new columns back to MultiIndex
    log_ret_co.columns = pd.MultiIndex.from_product([["log_ret_co"], log_ret_co.columns])
    return data_cube.join([log_ret_co])


@requires_data_cube
def add_realized_annualized_volatility(data_cube: pd.DataFrame, days) -> pd.DataFrame:
    r"""Add realized annualized volatility estimated over a rolling window.

    The volatility is computed as the rolling standard deviation of log returns
    over `days` trading days, annualized assuming 252 trading days per year:

    Parameters
    ----------
    data_cube : pd.DataFrame
        Input data cube with a field "log_return".
    days : int
        Length of the rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Original data with a new field "vol{days}".
    """

    return add_field(
        data_cube, f"vol{days}", "log_return", lambda t: t.rolling(days, min_periods=days).std() * np.sqrt(252 / days)
    )


@requires_data_cube
def add_momentum(data_cube: pd.DataFrame, days) -> pd.DataFrame:
    r"""Add momentum over a specified number of days.

    The momentum is defined as the cumulative log return over the past `days` trading days.

    Parameters
    ----------
    data_cube : pd.DataFrame
        Input data cube with a field "log_return".
    days : int
        Length of the rolling window in trading days.

    Returns
    -------
    pd.DataFrame
        Original data with a new field "mom{days}".
    """

    return add_field(data_cube, f"mom{days}", "log_return", lambda t: t.rolling(days, min_periods=days).sum())


@requires_data_cube
def add_liquidity(data_cube: pd.DataFrame) -> pd.DataFrame:
    r"""Add log dollar volume as a liquidity feature.

    The liquidity proxy is defined as the natural logarithm of daily dollar volume:

    $$
    L_t = \log(A_t \times \text{volume}_t)
    $$

    where $A_t$ is adjusted close. Observations with zero trading volume are replaced
    with NaN before applying the logarithm.

    This metric captures the order of magnitude of trading activity, with higher
    values indicating more liquid securities.

    Parameters
    ----------
    data_cube : pd.DataFrame
        Input data cube with fields "close" and "volume".

    Returns
    -------
    pd.DataFrame
        Original data augmented with a new field "log_dvol".

    Notes
    -------
    Both "volume" and "close" are expected to be both split adjusted
    or both not split adjusted.
    Our loader returns both, "close" and "volume", adjusted because
    Yahoo does not provide non-split adjusted data.
    """

    log_dvol = np.log((data_cube["close"] * data_cube["volume"]).replace(0, np.nan))

    # Add new columns back to MultiIndex
    log_dvol.columns = pd.MultiIndex.from_product([["log_dvol"], log_dvol.columns])
    return data_cube.join([log_dvol])
