"""Utility functions for event-data normalization and validation."""

import pandas as pd

from pandas.api.types import is_datetime64_any_dtype


# ============ STANDARD EVENT SCHEMA ============

# Standard DataFrame schema for all event data processing
# This is the baseline format that all event dataframes must conform to
STANDARD_EVENT_SCHEMA = {
    'ord_id': 'object',          # Order ID (can be NULL for trades)
    'upd_act': 'object',         # Update action: 0 (new), 1 (update), 2 (delete), 3 (delete thru), 4(delete from), 5 (overlay)
    'pos': 'int64',              # Position in rpt_seq
    'rpt_seq': 'int64',          # Report sequence number
    'px': 'float64',             # Price
    'size': 'int64',             # Size/Quantity
    'msg_tpe': 'object',         # Message type: 'snapshot' or 'incremental'
    'dat_tim': 'object',         # Datetime string (ISO8601 or formatted)
    'side': 'object'             # Side: 'bid', 'offer', 'trade'
}

STANDARD_EVENT_COLUMNS = list(STANDARD_EVENT_SCHEMA.keys())


def normalize_event_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* coerced to the STANDARD_EVENT_SCHEMA contract."""
    if df.empty:
        return pd.DataFrame(columns=STANDARD_EVENT_COLUMNS)

    normalized = df.copy()

    for column in STANDARD_EVENT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    to_numeric_cols = {
        "upd_act": "Int64",
        "pos": "Int64",
        "rpt_seq": "Int64",
        "size": "Int64",
    }

    for column, dtype in to_numeric_cols.items():
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(dtype)

    normalized["px"] = pd.to_numeric(normalized["px"], errors="coerce")

    normalized["msg_tpe"] = (
        normalized["msg_tpe"]
            .astype("string")
            .str.lower()
            .replace({"w": "snapshot", "x": "incremental"})
    )

    if not is_datetime64_any_dtype(normalized["dat_tim"]):
        normalized["dat_tim"] = pd.to_datetime(normalized["dat_tim"], errors="coerce", utc=True)

    normalized["side"] = normalized["side"].astype("string").str.lower()

    normalized["ord_id"] = normalized["ord_id"].astype("string")

    ordered_columns = STANDARD_EVENT_COLUMNS + [
        column for column in normalized.columns if column not in STANDARD_EVENT_COLUMNS
    ]

    return normalized[ordered_columns]


def validate_event_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that a DataFrame conforms to the standard event schema.
    DataFrame should be preprocessed with ``normalize_event_dataframe`` when
    loaded from raw sources.
    
    Checks:
    - All required columns are present
    - Column types match expected types
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        return True  # Allow empty dataframes
    
    # Check all required columns exist
    missing_cols = set(STANDARD_EVENT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"DataFrame missing required columns: {missing_cols}\n"
            f"Expected columns: {STANDARD_EVENT_COLUMNS}\n"
            f"Got columns: {list(df.columns)}"
        )
    
    if df["dat_tim"].isna().any():
        raise ValueError("DataFrame contains invalid dat_tim entries after normalization")

    return True


