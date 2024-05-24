from __future__ import annotations

from uuid import uuid4, UUID
from dataclasses import dataclass
from datetime import datetime
import getpass
from predictables.util.telemetry.db import DB


@dataclass
class User:
    """Represent a user in the database."""

    _user_id: UUID | None = None
    _user_name: str | None = None
    _user_lookup: str | None = None
    _db: DB | None = None

    def does_exist(self) -> bool:
        """Check if the user exists in the database."""
        return (
            self._db.execute_scalar(
                "SELECT COUNT(*) FROM users WHERE user_name.lower() = '?'",
                [self._user_name.lower()],
            )
            > 0
        )

    def update_last_accessed_at(self) -> None:
        """Update the last_accessed_at attribute in the database."""
        self.db.update_table_value(
            "users",
            f"user_name = '{self._user_name}'",
            f"last_accessed_at = '{datetime.now()}'",
        )

    def __post_init__(self):
        """Initialize the static attributes for the user object."""
        self._user_id = uuid4() if self._user_id is None else self._user_id
        self._user_name = (
            getpass.getuser() if self._user_name is None else self._user_name
        )
        self._user_lookup = (
            self._user_name.lower().replace(" ", "_")
            if self._user_lookup is None
            else self._user_lookup.lower().replace(" ", "_")
        )
        self._db = (
            self._db
            if self._db is not None
            else DB(db_file="predictables.db", schema="log", table="user")
        )

        self.update_last_accessed_at()

    @classmethod
    def __call__(cls) -> User:
        """Return a new instance of the User class."""
        user_name = getpass.getuser()
        try:
            user = cls.from_user_name(user_name)
        except ValueError as e:
            user = cls.new(user_name)

        return user

    @classmethod
    def new(cls, user_name: str | None = None) -> User:
        """Create a new user in the database."""
        now = datetime.now()
        # If no user_name is passed, use the OS username
        if user_name is None:
            user_name = getpass.getuser()

        usr = cls(
            user_name=user_name,
            user_id=uuid4(),
            user_lookup=user_name.lower().replace(" ", "_"),
            first_accessed_at=now,
            last_accessed_at=now,
        )

        if not usr.does_exist():
            usr.db.execute(
                "INSERT INTO users (user_id, user_name, first_accessed_at, last_accessed_at, user_lookup) VALUES (?, ?, ?, ?, ?)",
                [
                    usr.get_user_id(),
                    usr.get_user_name(),
                    usr.get_first_accessed_at(),
                    usr.get_last_accessed_at(),
                    usr.get_user_lookup(),
                ],
            )

        return usr

    @classmethod
    def from_user_id(cls, user_id: UUID) -> User:
        """Get a user from the user_id."""
        usr = cls(user_id=user_id)
        if not usr.does_exist():
            raise ValueError(f"User with user_id {user_id} does not exist.")

        return usr

    @classmethod
    def from_user_name(cls, user_name: str) -> User:
        """Get a user from the user_name."""
        usr = cls(user_name=user_name)
        if not usr.does_exist():
            raise ValueError(f"User with user_name {user_name} does not exist.")

        return usr

    # Getters and setters
    @property
    def user_id(self) -> UUID:
        """Get the user_id."""
        return self._user_id

    @property
    def user_name(self) -> str:
        """Get the user_name."""
        return self._user_name

    @property
    def user_lookup(self) -> str:
        """Get the user_lookup."""
        return self._user_lookup

    @property
    def first_accessed_at(self) -> datetime:
        """Get the first_accessed_at."""
        return self._first_accessed_at

    @property
    def last_accessed_at(self) -> datetime:
        """Get the last_accessed_at."""
        return self._last_accessed_at

    @property
    def db(self) -> DB:
        """Get the db."""
        return self._db

    @user_id.setter
    def user_id(self, user_id: UUID | None = None) -> None:
        """Set the user_id."""
        self._user_id = uuid4() if user_id is None else user_id

    @user_name.setter
    def user_name(self, user_name: str | None = None) -> None:
        """Set the user_name."""
        self._user_name = getpass.getuser() if user_name is None else user_name

    @user_lookup.setter
    def user_lookup(self, user_lookup: str | None = None) -> None:
        """Set the user_lookup."""
        user_name = getpass.getuser() if self.user_name() is None else self.user_name()
        self._user_lookup = (
            user_name.lower().replace(" ", "_")
            if user_lookup is None
            else user_lookup.lower().replace(" ", "_")
        )

    @first_accessed_at.setter
    def first_accessed_at(self, first_accessed_at: datetime | None = None) -> None:
        """Set the first_accessed_at."""
        self._first_accessed_at = (
            datetime.now() if first_accessed_at is None else first_accessed_at
        )

    @last_accessed_at.setter
    def last_accessed_at(self, last_accessed_at: datetime | None = None) -> None:
        """Set the last_accessed_at."""
        self._last_accessed_at = (
            datetime.now() if last_accessed_at is None else last_accessed_at
        )

    @db.setter
    def db(self, db: DB | None = None) -> None:
        """Set the db."""
        self._db = (
            db
            if db is not None
            else DB(db_file="predictables.db", schema="log", table="user")
        )
