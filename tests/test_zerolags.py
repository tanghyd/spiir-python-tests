import logging
import time
from functools import partial, update_wrapper
from pathlib import Path
from typing import Set, Dict, List, Callable

import click
import pandas as pd
import numpy as np

from spiir.io.ligolw import load_table_from_xmls
from spiir.io.cli import click_logger_options
from spiir.io.logging import configure_logger

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)

FLOAT_DTYPES: Set[np.dtype] = {np.dtype(np.float32), np.dtype(np.float64)}
INT_DTYPES: Set[np.dtype] = {np.dtype(np.int32), np.dtype(np.int64)}
STR_DTYPES: Set[np.dtype] = {np.dtype("O")}

def wrapped_partial(func: Callable, *args, **kwargs) -> Callable:
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def test_dtypes_equal(a: pd.Series, b: pd.Series):
    assert a.dtype == b.dtype, f"Data types between columns must match."

def test_not_na(a: pd.Series, b: pd.Series):
    assert not a.notna().all() and not b.notna().all()

def test_dtypes(a: pd.Series, b: pd.Series, dtypes: Set[np.dtype]):
    assert a.dtype in dtypes and b.dtypes in dtypes 

def test_diff(a: pd.Series, b: pd.Series):
    if (a != b).any():
        if a.dtype in FLOAT_DTYPES or b.dtype in FLOAT_DTYPES:
            # calculate order of magnitude for matching decimal places between a and b
            with np.errstate(divide='ignore', invalid='ignore'):
                decimals = (-1*np.floor(np.log10((a - b).abs())))
            decimals = decimals.replace(np.inf, np.nan)
            decimals = decimals.astype(pd.Int64Dtype())  # nullable
            decimals = decimals.dropna()

            if len(decimals) > 0:
                stats = {k: getattr(decimals, k)() for k in ("min", "max", "median")}
                stats_summary = ' | '.join([f'{k}: {v}' for k, v in stats.items()])
                err = f"Values do not match up to n decimal places: {stats_summary}"        
                raise AssertionError(err)
            else:
                logger.warning(f"[DEBUG] Unknown behaviour: {stats}")
        else:
            raise AssertionError("Values do not match.")

def test_float_dtypes(a: pd.Series, b: pd.Series):
    test_dtypes(a, b, FLOAT_DTYPES)

def test_int_dtype(a: pd.Series, b: pd.Series):
    test_dtypes(a, b, INT_DTYPES)

def test_str_dtype(a: pd.Series, b: pd.Series):
    test_dtypes(a, b, STR_DTYPES)

def test_str_equal(a: pd.Series, b: pd.Series):
    assert (a == b).all()

def test_str_equal_case_insensitive(a: pd.Series, b: pd.Series):
    assert (a.str.lower() == b.str.lower()).all()


TESTS: Dict[str, List[Callable]] = {
    "all": [test_dtypes_equal],
    "float": [test_diff],
    "int": [test_diff],
    "string": [test_str_equal_case_insensitive, test_str_equal],
}


@click.command
@click.argument("a", type=str)
@click.argument("b", type=str)
@click.option("-t", "--table", type=str, default="postcoh")
@click_logger_options
def main(
    a: str,
    b: str,
    table: str = "postcoh",
    log_level: int = logging.WARNING,
    log_file: str = None
):
    configure_logger(logger, log_level, log_file)
    duration = time.perf_counter()

    logger.info(f"Running tests for tables: {table}")

    logger.info(f"Loading {a}...")
    df_a = load_table_from_xmls(a, table=table)

    logger.info(f"Loading {b}...")
    df_b = load_table_from_xmls(b, table=table)

    dtypes = ["all", "float", "int", "string"]
    for dtype in dtypes:
        for func in TESTS[dtype]:
            if dtype == "all":
                keys = df_a.columns
            else:
                keys = df_a.dtypes.loc[df_a.dtypes.apply(str).str.contains(dtype)].index
            for key in keys:
                try:
                    func(df_a[key], df_b[key])
                except AssertionError as err:
                    logger.warning(f"{key}: {func.__name__} error! {err}")
                else:
                    logger.info(f"{key}: {func.__name__} success!")

    duration = time.perf_counter() - duration
    logger.info(f"{Path(__file__).stem} script ran in {duration:.4f} seconds.")

if __name__ == "__main__":
    main()