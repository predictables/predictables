import datetime

import numpy as np
import pandas as pd

default_start_date = datetime.datetime(1970, 1, 1)
default_end_date = datetime.datetime(2100, 1, 1)


def generate_date_lookup(
    start_date: datetime.date = default_start_date,
    end_date: datetime.date = default_end_date,
) -> pd.DataFrame:
    """
    Generates a dataframe with the following columns:
    - date_id: integer
    - date: date
    - year: integer
    - month: integer
    - quarter: integer
    - day: integer
    - days_in_month: integer
    - day_of_week: integer
    - day_of_year: integer
    - days_in_year: integer
    - cos[month]: float
    - sin[month]: float
    - cos[quarter]: float
    - sin[quarter]: float
    - cos[day]: float
    - sin[day]: float
    - cos[day_of_week]: float
    - sin[day_of_week]: float
    - cos[day_of_year]: float
    - sin[day_of_year]: float

    Parameters
    ----------
    start_date : datetime.date, optional
        The start date for the lookup, by default datetime.datetime(1970, 1, 1)
    end_date : datetime.date, optional
        The end date for the lookup, by default datetime.datetime(2100, 1, 1)

    Returns
    -------
    pd.DataFrame
        A dataframe with the columns described above.

    Notes
    -----
    This function is intended to be used to generate a lookup table for dates.
    Every date between the start and end date will be included in the lookup, and
    indexed by an integer date_id. The date_id can then be used to join to other
    tables. The date_id is also useful for doing things like calculating the number
    of days between two dates. Fourier features are also included for the month,
    quarter, day of month, day of week, and day of year.

    """
    # Start with a dataframe of dates
    dates = (
        pd.DataFrame(
            pd.date_range(start=start_date, end=end_date), columns=["date"]
        )
        .reset_index()
        .rename(columns=dict(index="date_id"))
        .set_index("date_id")
    )

    # Extract year and month
    dates["year"] = dates["date"].dt.year
    dates["month"] = dates["date"].dt.month

    # Set up the month as a periodic/cyclical variable -- eg project the month numbers
    # from 1 to 12 onto a circle. this ensures that the distance between 12 and 1 is
    # the same as the distance between 1 and 2, which is not the case if we just
    # treat month as a categorical variable.
    dates["month_cpx"] = dates["date"].dt.month.apply(
        lambda x: (2 * np.pi) * ((x - 1) / 12)  # these range from 0 to 2pi
    )

    # Create complex numbers for the month -- this is the same as the projection
    # above, but in complex number form. This is useful for doing things like
    # calculating the distance between two months.
    dates["cos[month]"] = dates["month_cpx"].apply(
        lambda x: np.cos(x)
    )  # real part
    dates["sin[month]"] = dates["month_cpx"].apply(
        lambda x: np.sin(x)
    )  # imaginary part

    # Note that this implies that the distance between 12 and 1 is the same as the
    # distance between 1 and 2, which is not the case if we just treat month as a
    # categorical variable. This also encodes the fact that December is closer to
    # January than it is to July.

    # Extract quarter
    dates["quarter"] = dates["date"].dt.quarter

    # Similar quarter projection
    dates["quarter_cpx"] = dates["date"].dt.quarter.apply(
        lambda x: (2 * np.pi) * ((x - 1) / 4)  # these range from 0 to 2pi
    )
    dates["cos[quarter]"] = dates["quarter_cpx"].apply(
        lambda x: np.cos(x)
    )  # real part
    dates["sin[quarter]"] = dates["quarter_cpx"].apply(
        lambda x: np.sin(x)
    )  # imaginary part

    # Extract day of month, and info needed to project it onto a circle
    dates["day"] = dates["date"].dt.day
    dates["days_in_month"] = dates["date"].dt.days_in_month

    # Project day onto a circle for the day of MONTH
    dates["day_cpx"] = dates["day days_in_month".split()].apply(
        lambda x: (2 * np.pi) * ((x[0] - 1) / x[1]), axis=1
    )
    dates["cos[day]"] = dates["day_cpx"].apply(lambda x: np.cos(x))
    dates["sin[day]"] = dates["day_cpx"].apply(lambda x: np.sin(x))

    # Project day onto a circle for the day of WEEK
    dates["day_of_week"] = dates["date"].dt.day_of_week
    dates["day_of_week_cpx"] = dates["date"].dt.day_of_week.apply(
        lambda x: (2 * np.pi) * ((x - 1) / 7)
    )
    dates["cos[day_of_week]"] = dates["day_of_week_cpx"].apply(
        lambda x: np.cos(x)
    )
    dates["sin[day_of_week]"] = dates["day_of_week_cpx"].apply(
        lambda x: np.sin(x)
    )

    # Project day onto a circle for the day of YEAR
    dates["day_of_year"] = dates["date"].dt.day_of_year
    dates["days_in_year"] = dates["date"].dt.is_leap_year.apply(
        lambda x: 366 if x else 365
    )
    dates["day_of_year_cpx"] = dates["day_of_year days_in_year".split()].apply(
        lambda x: (2 * np.pi) * ((x[0] - 1) / x[1]), axis=1
    )
    dates["cos[day_of_year]"] = dates["day_of_year_cpx"].apply(
        lambda x: np.cos(x)
    )
    dates["sin[day_of_year]"] = dates["day_of_year_cpx"].apply(
        lambda x: np.sin(x)
    )

    dates.drop(
        columns=[
            "quarter_cpx",
            "month_cpx",
            "day_cpx",
            "day_of_week_cpx",
            "day_of_year_cpx",
        ],
        inplace=True,
    )
