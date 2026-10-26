"""
Core module: BTG-OBD normalization and order-book reconstruction.

Provides data structures and functionality for:
- Event data normalization and validation (STANDARD_EVENT_SCHEMA)
- Incremental order book reconstruction from any data source
- Book state snapshots
"""

from .book_cache import InstrumentBook, BookSide, BookEntry
from .book_types import (
    BookEvent,
    BookMBPSnapshot,
    BookMBPLevelSnapshot,
    BookOrderSnapshot,
)
from .time_utils import current_hw_timestamp, format_date_time_from_timestamp
from .utils import (
    STANDARD_EVENT_SCHEMA,
    STANDARD_EVENT_COLUMNS,
    normalize_event_dataframe,
    validate_event_dataframe,
)
from .reconstruction import (
    reconstruct_from_parquet,
    iter_book_events_from_parquet,
    reconstruct_from_dataframe,
    iter_book_events_from_dataframe,
)

__all__ = [
    # Book structures
    "InstrumentBook",
    "BookSide",
    "BookEntry",
    # Types
    "BookEvent",
    "BookMBPSnapshot",
    "BookMBPLevelSnapshot",
    "BookOrderSnapshot",
    # Utilities
    "current_hw_timestamp",
    "format_date_time_from_timestamp",
    # Standard event schema helpers
    "STANDARD_EVENT_SCHEMA",
    "STANDARD_EVENT_COLUMNS",
    "normalize_event_dataframe",
    "validate_event_dataframe",
    # Reconstruction helpers
    "reconstruct_from_parquet",
    "iter_book_events_from_parquet",
    "reconstruct_from_dataframe",
    "iter_book_events_from_dataframe",
]
