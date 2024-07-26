"""Implement DuckDB connections for various pre-defined databases, and interfaces to extend them."""

from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd
import polars as pl
import duckdb
import pyodbc
from typing import List
import types

__all__ = ["DuckDB", "HitRatioDB", "HitRatioDBT", "DuckPond", "LossDB", "Db2"]

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
    filename="duck_db.log",
)

logger = logging.getLogger(__name__)


@dataclass
class DuckDB:
    """Base DuckDB connection. Handles read and write operations."""

    db_file: str = ":memory:"
    db_conn: duckdb.DuckDBPyConnection | None = None

    slots = "db_file"

    def __post_init__(self) -> None:
        """Initialize the DuckDB connection and log the connection."""
        logger.debug(f"Creating {self.__class__.__name__} with db_file={self.db_file}")

    def __call__(self, query: str) -> pl.DataFrame:
        """Execute a query in a read-only context, and return the result as a polars DataFrame."""
        return self.read(query)

    def write(self, query: str) -> pl.DataFrame:
        """Execute a query in a write context, and return the result as a polars DataFrame."""
        logger.debug(f"query for write op:\n{query}")
        with duckdb.connect(self.db_file, read_only=False) as conn:
            res = conn.sql(query)
            if res is not None:
                return res.pl()
            return None

    def read(self, query: str) -> pl.DataFrame:
        """Execute a query in a read-only context, and return the result as a polars DataFrame."""
        logger.debug(f"query for read op:\n{query}")
        with duckdb.connect(self.db_file, read_only=True) as conn:
            res = conn.sql(query)
            if res is not None:
                return res.pl()
            return None

    def __enter__(self):
        """Enter a with context and open a read/write connection to the DuckDB database."""
        logger.debug(f"Entering a with context on {self.db_file}")
        self.db_conn = duckdb.connect(self.db_file)
        logger.debug("With context opened")
        return self.db_conn

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Exit a with context and close the connection to the DuckDB database."""
        logger.debug(f"Closing a with context on {self.db_file}")
        self.db_conn.close()
        self.db_conn = None
        logger.debug("With context closed")

    def get_user_defined_enum_types(self) -> List[str]:
        """Get the user-defined enum types in the DuckDB database."""
        enum_qry = """
            select type_name
            from duckdb_types()
            where 
                (
                    ends_with(type_name, '__type')
                    or ends_with(type_name, 'type')
                )
                and logical_type='ENUM'
        """
        logger.debug(f"Query to get user-defined enum types:\n{enum_qry}")
        try:
            with duckdb.connect(self.db_file) as conn:
                ud_types__RAW = conn.sql(enum_qry)
                ud_types = (
                    ud_types__RAW.to_pandas()["type_name"].tolist()
                    if ud_types__RAW.shape[0] > 0
                    else []
                )

        except Exception as _:
            ud_types = []

        logger.debug(f"User-defined enum types:\n{ud_types}")
        return ud_types

    def clear_user_defined_enum_type(self, type_name: str) -> None:
        """Drop a user-defined enum type from the DuckDB database."""
        ud_types = self.get_user_defined_enum_types()
        if type_name not in ud_types:
            err_msg = (
                f"`{type_name}` does not appear in the list of user-defined enum types:"
            )
            err_msg += "\n"
            err_msg += "\n".join(ud_types)
            logger.error(err_msg)

        try:
            drop_qry = f"drop type {type_name}"
            logger.debug(f"Query to drop type {type_name}:\n{drop_qry}")
            self.write(drop_qry)
            logger.debug(f"Type {type_name} dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping type {type_name}:\n{e}")

    def clear_all_user_defined_enum_types(self) -> None:
        """Drop all user-defined enum types from the DuckDB database."""
        for t in self.get_user_defined_enum_types():
            self.clear_user_defined_enum_type(t)

    def create_or_replace_string_enum(
        self, old_col: str, new_col: str, table: str
    ) -> None:
        """Create or replace an enum type from a string column in a table."""
        type_name = f"{new_col}__type"

        logger.debug(
            f"Creating or replacing enum type {type_name} from {old_col} in {table}"
        )

        # Drop the enum type if it is already defined
        if type_name in self.get_user_defined_enum_types():
            logger.debug(
                f"Type {type_name} already exists. Dropping it before creating a new one"
            )
            self.clear_user_defined_enum_type(type_name)
        elif type_name.replace("__", "_") in self.get_user_defined_enum_types():
            logger.debug(
                f"Type {type_name} does not exist but {type_name.replace('__', '_')} does. Dropping it before creating a new one"
            )
            self.clear_user_defined_enum_type(type_name.replace("__", "_"))
        else:
            logger.debug(
                f"Type {type_name} does not already exist, so just need to create it."
            )

        # Create the new enum type from the values from the table
        create_type_qry = f"create type {type_name} as enum (select distinct {old_col} from {table} where {old_col} is not null);"  # noqa: S608
        logger.debug(f"Query to create type {type_name}:\n{create_type_qry}")
        try:
            self.write(create_type_qry)
            logger.debug(f"Type {type_name} created successfully")
        except Exception as e:
            logger.error(f"Error creating type {type_name}:\n{e}")


@dataclass
class HitRatioDB(DuckDB):
    """DuckDB connection for the hit ratio database."""

    db_file: str = "/sas/data/project/EG/ActShared/SmallBusiness/Modeling/hit_ratio/bop_model/hit_ratio.db"
    slots = "db_file"


@dataclass
class HitRatioDBT(DuckDB):
    """DuckDB connection for the hit ratio dbt database."""

    db_file: str = "/sas/data/project/EG/ActShared/SmallBusiness/Modeling/hit_ratio/hit_ratio_data_pipeline/hit_ratio.duckdb"
    slots = "db_file"


@dataclass
class DuckPond(DuckDB):
    """DuckDB connection for the duck pond database."""

    db_file: str = "/sas/data/project/EG/ActShared/SmallBusiness/aw/dbt/duckpond.duckdb"
    slots = "db_file"


@dataclass
class LossDB(DuckDB):
    """DuckDB connection for the loss database."""

    db_file: str = (
        "/sas/data/project/EG/ActShared/SmallBusiness/Modeling/dat/loss.duckdb"
    )
    slots = "db_file"


@dataclass
class Db2:
    """DB2 connection."""

    username: str
    password: str
    driver: str = "/opt/ibm/db2/clidriver/lib/libdb2.so"
    hostname: str = "lnxvdb2hq020.cinfin.com"
    port: int = 50004
    protocol: str = "TCPIP"
    database: str = "CFCPSAS"

    slots = (
        "username",
        "password",
        "driver",
        "hostname",
        "port",
        "protocol",
        "database",
    )

    def get_conn_str(self) -> str:
        """Return the connection string for the DB2 connection."""
        return f"Driver={self.driver}; Hostname={self.hostname};Port={self.port};Protocol={self.protocol};Database={self.database};UID={self.username};PWD={self.password};"

    def get_conn(self) -> pyodbc.Connection:
        """Return a connection to the DB2 database."""
        return pyodbc.connect(self.get_conn_str())

    def __repr__(self) -> str:
        """Return a string representation of the DB2 connection."""
        return f"DB2Conn({', '.join(f'{slot}={getattr(self, slot)}' for slot in self.slots)})"

    def __str__(self) -> str:
        """Return a string representation of the DB2 connection."""
        return self.get_conn_str()

    def __call__(self, query: str):
        """Execute a query in a read-only context, and return the result as a polars DataFrame."""
        with self.get_conn() as conn, conn.cursor() as cursor:
            cursor.execute(query)
            raw = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
            return pl.from_pandas(pd.DataFrame.from_records(raw, columns=columns))
