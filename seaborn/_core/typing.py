"""
Type definitions for seaborn core module.

This module provides type hints and protocols used throughout seaborn's
internal type system, enabling better static analysis and IDE support.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from numpy import ndarray
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Series, Index, Timestamp, Timedelta
from matplotlib.colors import Colormap, Normalize


ColumnName = Union[
    str, bytes, date, datetime, timedelta, bool, complex, Timestamp, Timedelta
]
Vector = Union[Series, Index, ndarray]

VariableSpec = Union[ColumnName, Vector, None]
VariableSpecList = Union[List[VariableSpec], Index, None]


@runtime_checkable
class DataFrameProtocol(Protocol):
    """
    Protocol for objects that can be converted to a pandas DataFrame.

    This follows the DataFrame interchange protocol, allowing seaborn
    to work with any DataFrame-like object that implements to_pandas().
    """

    def to_pandas(self) -> DataFrame:
        """
        Convert the object to a pandas DataFrame.

        Returns
        -------
        DataFrame
            The converted pandas DataFrame.
        """
        ...


DataSource = Union[DataFrame, DataFrameProtocol, Mapping[str, Any], None]

OrderSpec = Union[Iterable[Any], None]
NormSpec = Union[Tuple[Optional[float], Optional[float]], Normalize, None]

PaletteSpec = Union[str, List[Any], Dict[Any, Any], Colormap, None]
DiscreteValueSpec = Union[Dict[Any, Any], List[Any], None]
ContinuousValueSpec = Union[
    Tuple[float, float], List[float], Dict[Any, float], None,
]


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class Default:
    """
    Sentinel class for default parameter values.

    Used to distinguish between 'not provided' and 'explicitly None'.
    """

    _instance: ClassVar[Optional["Default"]] = None

    def __new__(cls) -> "Default":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<default>"

    def __bool__(self) -> bool:
        return False


class Deprecated:
    """
    Sentinel class for deprecated parameter values.

    Used to mark parameters that are scheduled for removal.
    """

    _instance: ClassVar[Optional["Deprecated"]] = None

    def __new__(cls) -> "Deprecated":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<deprecated>"

    def __bool__(self) -> bool:
        return False


default = Default()
deprecated = Deprecated()


class TypedDictLike(Protocol[T_co]):
    """Protocol for TypedDict-like objects."""

    def __getitem__(self, key: str) -> T_co:
        ...

    def keys(self) -> Iterable[str]:
        ...

    def values(self) -> Iterable[T_co]:
        ...

    def items(self) -> Iterable[Tuple[str, T_co]]:
        ...


class HasName(Protocol):
    """Protocol for objects with a 'name' attribute."""

    name: Optional[str]


class SizedIterable(Protocol):
    """Protocol for objects that are both sized and iterable."""

    def __len__(self) -> int:
        ...

    def __iter__(self) -> Any:
        ...


NumericArray = Union[NDArray[Any], Series]
ColorType = Union[Tuple[float, float, float], Tuple[float, float, float, float], str]
MarkerType = Union[float, str, Tuple[int, int, float], List[Tuple[float, float]], Any]
DashPattern = Tuple[float, ...]
DashPatternWithOffset = Tuple[float, Optional[DashPattern]]
