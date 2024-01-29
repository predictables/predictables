from ._structures import InfinityType, NegativeInfinityType
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

InfiniteTypes = Union[InfinityType, NegativeInfinityType]
PrePostDevType = Union[InfiniteTypes, Tuple[str, int]]
SubLocalType = Union[InfiniteTypes, int, str]
LocalType = Union[
    NegativeInfinityType,
    Tuple[
        Union[
            SubLocalType,
            Tuple[SubLocalType, str],
            Tuple[NegativeInfinityType, SubLocalType],
        ],
        ...,
    ],
]
CmpKey = Tuple[
    int, Tuple[int, ...], PrePostDevType, PrePostDevType, PrePostDevType, LocalType
]
LegacyCmpKey = Tuple[int, Tuple[str, ...]]
VersionComparisonMethod = Callable[
    [Union[CmpKey, LegacyCmpKey], Union[CmpKey, LegacyCmpKey]], bool
]

class _Version(NamedTuple):
    epoch: Any
    release: Any
    dev: Any
    pre: Any
    post: Any
    local: Any

def parse(version: str) -> Union["LegacyVersion", "Version"]: ...

class InvalidVersion(ValueError): ...

class _BaseVersion:
    def __hash__(self) -> int: ...
    def __lt__(self, other: _BaseVersion) -> bool: ...
    def __le__(self, other: _BaseVersion) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ge__(self, other: _BaseVersion) -> bool: ...
    def __gt__(self, other: _BaseVersion) -> bool: ...
    def __ne__(self, other: object) -> bool: ...

class LegacyVersion(_BaseVersion):
    def __init__(self, version: str) -> None: ...
    @property
    def public(self) -> str: ...
    @property
    def base_version(self) -> str: ...
    @property
    def epoch(self) -> int: ...
    @property
    def release(self) -> None: ...
    @property
    def pre(self) -> None: ...
    @property
    def post(self) -> None: ...
    @property
    def dev(self) -> None: ...
    @property
    def local(self) -> None: ...
    @property
    def is_prerelease(self) -> bool: ...
    @property
    def is_postrelease(self) -> bool: ...
    @property
    def is_devrelease(self) -> bool: ...

VERSION_PATTERN: str

class Version(_BaseVersion):
    def __init__(self, version: str) -> None: ...
    @property
    def epoch(self) -> int: ...
    @property
    def release(self) -> Tuple[int, ...]: ...
    @property
    def pre(self) -> Optional[Tuple[str, int]]: ...
    @property
    def post(self) -> Optional[int]: ...
    @property
    def dev(self) -> Optional[int]: ...
    @property
    def local(self) -> Optional[str]: ...
    @property
    def public(self) -> str: ...
    @property
    def base_version(self) -> str: ...
    @property
    def is_prerelease(self) -> bool: ...
    @property
    def is_postrelease(self) -> bool: ...
    @property
    def is_devrelease(self) -> bool: ...
    @property
    def major(self) -> int: ...
    @property
    def minor(self) -> int: ...
    @property
    def micro(self) -> int: ...
