import numpy as np
import pandas as pd

date_lookup = pd.DataFrame({"date": pd.date_range("2021-01-01", "2021-12-31")})
date_lookup["month"] = date_lookup["date"].dt.month
date_lookup["quarter"] = date_lookup["date"].dt.quarter


def angle(x: pd.Series, k: int) -> pd.Series:
    return 2 * np.pi * x * k


def _days_in_year(x: pd.Series) -> pd.Series:
    days = pd.Series([365] * x.shape[0])
    days = days.mask(x.dt.is_leap_year, 366)
    return days


def _days_in_quarter(x: pd.Series) -> pd.Series:
    qtr = x.dt.quarter
    qtr_days = {
        1: 31 + 28 + 31,
        2: 30 + 31 + 30,
        3: 31 + 31 + 30,
        4: 31 + 30 + 31,
    }

    if x.dt.is_leap_year:
        qtr_days[1] += 1

    return qtr.map(qtr_days)


def _days_in_quarter_already(x: pd.Series) -> pd.Series:
    qtr = {
        1: 0,
        2: 31,
        3: 31 + 28,
        4: 0,
        5: 30,
        6: 30 + 31,
        7: 0,
        8: 31,
        9: 31 + 31,
        10: 0,
        11: 30,
        12: 30 + 31,
    }

    days = x.dt.month.map(qtr).mask(
        x.dt.is_leap_year & (x.dt.month.eq(3)), 31 + 29
    )
    return days


def day_of_week(x: pd.Series) -> pd.Series:
    day = x.dt.day_of_week + 0.5
    return day / 7


def day_of_month(x: pd.Series) -> pd.Series:
    days_in_mo = x.dt.days_in_month
    day = x.dt.day + 0.5
    return day / days_in_mo


def day_of_quarter(x: pd.Series) -> pd.Series:
    days_in_qtr = _days_in_quarter(x)
    day = x.dt.day + 0.5
    return day / days_in_qtr


def day_of_year(x: pd.Series) -> pd.Series:
    days_in_year = _days_in_year(x)
    day = x.dt.day_of_year + 0.5
    return day / days_in_year
