import logging
import time
from functools import partial, update_wrapper
from pathlib import Path
from typing import Optional, Union, Set, Dict, List, Callable

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


def test_df_row_count(a: pd.DataFrame, b: pd.DataFrame):
    assert len(a) == len(b), f"Number of rows do not match."

def test_df_col_count(a: pd.DataFrame, b: pd.DataFrame):
    assert len(a.columns) == len(b.columns), f"Number of columns do not match."

def test_df_col_order(a: pd.DataFrame, b: pd.DataFrame):
    assert (a.columns == b.columns).all(), f"Columns do not exactly match in order."

def test_df_col_exists(a: pd.DataFrame, b: pd.DataFrame):
    a_in_b = np.all([col in b.columns for col in a.columns])
    b_in_a = np.all([col in b.columns for col in b.columns])
    assert a_in_b and b_in_a, f"Columns from A in B? {a_in_b}; from B in A? {b_in_a}"

def test_not_na(a: pd.Series, b: pd.Series):
    assert not a.notna().all() and not b.notna().all()

def test_dtypes_equal(a: pd.Series, b: pd.Series):
    assert a.dtype == b.dtype, f"Data types between columns must match."

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

def test_str_equal(a: pd.Series, b: pd.Series):
    assert (a == b).all()

def test_str_equal_case_insensitive(a: pd.Series, b: pd.Series):
    assert (a.str.lower() == b.str.lower()).all()

def test_dtypes(a: pd.Series, b: pd.Series, dtypes: Set[np.dtype]):
    assert a.dtype in dtypes and b.dtypes in dtypes 

def test_float_dtypes(a: pd.Series, b: pd.Series):
    test_dtypes(a, b, FLOAT_DTYPES)

def test_int_dtype(a: pd.Series, b: pd.Series):
    test_dtypes(a, b, INT_DTYPES)

def test_str_dtype(a: pd.Series, b: pd.Series):
    test_dtypes(a, b, STR_DTYPES)

TESTS: Dict[str, List[Callable]] = {
    "required_df": [test_df_row_count, test_df_col_count, test_df_col_exists],
    "df": [test_df_col_order],
    "column": [test_dtypes_equal],
    "float": [test_diff],
    "int": [test_diff],
    "string": [test_str_equal_case_insensitive, test_str_equal],
}

def load_table(p: Union[str, Path], table: str, glob: str = "*") -> pd.DataFrame:
    logger.info(f"Reading {table} table data from {p}...")
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"File or directory at {p} not found.")

    paths = list(path.glob(glob)) if path.is_dir() else [path]
    df = load_table_from_xmls(paths, table=table)
    n, m = len(df), len(df.columns)
    logger.info(f"Loaded {n} rows and {m} columns from {len(paths)} path(s).")
    return df

@click.command
@click.argument("a", type=str)
@click.argument("b", type=str)
@click.option("--table", type=str, default="postcoh")
@click.option("-t", "--tests", type=str, multiple=True)
@click.option("-g", "--glob", type=str, default="*zerolag_*.xml*")
@click_logger_options
def main(
    a: str,
    b: str,
    table: str = "postcoh",
    tests: Optional[Union[str, List[str]]] = None,
    glob: str = "*",
    log_level: int = logging.WARNING,
    log_file: str = None
):
    configure_logger(logger, log_level, log_file)
    duration = time.perf_counter()

    # load two sets of tables from .xml files to compare against eachother
    df_a = load_table(a, table=table, glob=glob)
    df_b = load_table(b, table=table, glob=glob)
    required_test_fail = False  # we check required tests before column-wise tests
    
    tests = tests or list(TESTS.keys())
    logger.info(f"Running zerolag tests: {tests}")
    for key in tests if isinstance(tests, list) else [tests]:
        if required_test_fail:
            logger.warning(f"Required test failed. Aborting...")
            break
        for test in TESTS[key]:
            if "df" in key:
                try:
                    test(df_a, df_b)
                except AssertionError as err:
                    logger.warning(f"{key}: {test.__name__} error! {err}")
                    if "required" in key:
                        required_test_fail = True  # will exit if any df tests fail
                else:
                    logger.info(f"{key}: {test.__name__} success!")
            else:
                if key == "column":
                    cols = df_a.columns
                else:
                    dtype_column_mask = df_a.dtypes.apply(str).str.contains(key)
                    cols = df_a.dtypes.loc[dtype_column_mask].index  # column names

                for col in cols:
                    try:
                        test(df_a[col], df_b[col])
                    except AssertionError as err:
                        logger.warning(f"{key}: {col}: {test.__name__} error! {err}")
                    else:
                        logger.info(f"{key}: {col}: {test.__name__} success!")

    duration = time.perf_counter() - duration
    logger.info(f"{Path(__file__).stem} script ran in {duration:.4f} seconds.")

if __name__ == "__main__":
    main()