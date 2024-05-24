"""Define the DB class for database connections."""

from __future__ import annotations

import duckdb
from dataclasses import dataclass


@dataclass
class DB:
    """Represent a database connection."""

    db_file: str = "predictables.db"
    schema: str = "log"
    table: str | None = None

    def execute(self, query: str, params: list) -> list:
        """Execute a query on the database."""
        return self.execute(query, params).fetchall()

    def execute_one(self, query: str, params: list) -> list:
        """Execute a query on the database and return the first row."""
        return self.execute(query, params).fetchone()

    def execute_scalar(self, query: str, params: list) -> list:
        """Execute a query on the database and return the first column of the first row."""
        return self.execute(query, params).fetchone()[0]

    def update_table_value(
        self, table: str, filter: str, replacement_value: str
    ) -> None:
        """Update the table attribute."""
        self.execute(
            f"UPDATE {table} SET {replacement_value} where {filter}"  # noqa: S608
        )