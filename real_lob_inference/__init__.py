"""BTG-OBD event normalization and historical order-book reconstruction."""

from .core import (
    BookEvent,
    InstrumentBook,
    iter_book_events_from_dataframe,
    iter_book_events_from_parquet,
    reconstruct_from_dataframe,
    reconstruct_from_parquet,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "BookEvent",
    "InstrumentBook",
    "iter_book_events_from_dataframe",
    "iter_book_events_from_parquet",
    "reconstruct_from_dataframe",
    "reconstruct_from_parquet",
]
