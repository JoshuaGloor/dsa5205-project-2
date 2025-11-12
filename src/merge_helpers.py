import pandas as pd
from src.pipeline.preprocess import requires_data_cube


def _col(ticker: str, feature: str, pattern="{ticker}_{feature}") -> str:
    """Generate a standardized column label from ticker and feature names.

    Used in code-merging contexts to ensure consistent column naming across
    modules that flatten MultiIndex structures.

    Parameters
    ----------
    ticker : str
        The asset identifier (e.g., "NVDA").
    feature : str
        The feature field name (e.g., "log_return").
    pattern : str, optional
        Format template combining both identifiers. Must include
        "{ticker}" and "{feature}" placeholders. Default is "{ticker}_{feature}".

    Returns
    -------
    str
        Formatted column name (e.g., "NVDA_log_ret").
    """
    return pattern.format(ticker=ticker, feature=feature)


@requires_data_cube
def flatten_columns(data_cube: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column structure into single-level column names.

    Converts a DataFrame with MultiIndex columns into a flat structure
    with string column names such as "NVDA_log_ret".

    Parameters
    ----------
    data_cube : pd.DataFrame
        DataFrame with MultiIndex columns.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with flattened, single-level columns.
    """

    df = data_cube.copy()
    new_cols = [_col(t, f) for (f, t) in df.columns]
    df.columns = pd.Index(new_cols)
    return df


def check_df_empty(df: pd.DataFrame) -> bool:
    """Check whether a DataFrame is effectively empty or invalid.

    Evaluates several conditions to determine if the input is a usable
    pandas DataFrame. Returns True if any of the following hold:
    - The input is None.
    - The input is not a pandas DataFrame.
    - The DataFrame has zero rows.
    - All entries are NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame to check.

    Returns
    -------
    bool
        True if the DataFrame is empty, invalid, or fully NaN; False otherwise.
    """
    return df is None or not isinstance(df, pd.DataFrame) or df.shape[0] == 0 or df.isna().all().all()
