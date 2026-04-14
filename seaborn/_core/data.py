"""
Components for parsing variable assignments and internally representing plot data.
"""
from __future__ import annotations

from collections.abc import Mapping, Sized
from typing import (
    Any, Generic, Literal, TypeVar, Protocol, TypedDict, Union, cast, overload
)

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from seaborn._core.typing import DataSource, VariableSpec, ColumnName


class VariableMapping(TypedDict):
    """Type specification for variable mapping dictionaries."""
    x: VariableSpec
    y: VariableSpec
    hue: VariableSpec
    size: VariableSpec
    style: VariableSpec


class DataFrameLike(Protocol):
    """Protocol defining the interface for objects convertible to DataFrame."""

    def to_pandas(self) -> DataFrame:
        ...


T = TypeVar("T")


class PlotData:
    """
    Data table with plot variable schema and mapping to original names.

    Contains logic for parsing variable specification arguments and updating
    the table with layer-specific data and/or mappings.

    Parameters
    ----------
    data
        Input data where variable names map to vector values.
    variables
        Keys are names of plot variables (x, y, ...) each value is one of:

        - name of a column (or index level, or dictionary entry) in `data`
        - vector in any format that can construct a :class:`pandas.DataFrame`

    Attributes
    ----------
    frame
        Data table with column names having defined plot variables.
    names
        Dictionary mapping plot variable names to names in source data structure(s).
    ids
        Dictionary mapping plot variable names to unique data source identifiers.
    source_data
        Reference to the original input data source.
    source_vars
        Dictionary of original variable specifications used to initialize.
    frames
        Dictionary mapping keys to sub-frames (used primarily for faceting).

    """
    frame: DataFrame
    frames: dict[tuple[str, ...], DataFrame]
    names: dict[str, str | None]
    ids: dict[str, str | int]
    source_data: DataSource
    source_vars: dict[str, VariableSpec]

    def __init__(
        self,
        data: DataSource,
        variables: dict[str, VariableSpec],
    ) -> None:

        data = handle_data_source(data)
        frame, names, ids = self._assign_variables(data, variables)

        self.frame = frame
        self.names = names
        self.ids = ids
        self.frames = {}

        self.source_data = data
        self.source_vars = variables

    def __contains__(self, key: str) -> bool:
        """Boolean check on whether a variable is defined in this dataset."""
        if self.frame is None:
            return any(key in df for df in self.frames.values())
        return key in self.frame

    @overload
    def join(self, data: None, variables: None) -> PlotData:
        ...

    @overload
    def join(self, data: DataSource, variables: None) -> PlotData:
        ...

    @overload
    def join(self, data: None, variables: dict[str, VariableSpec] | None) -> PlotData:
        ...

    def join(
        self,
        data: DataSource,
        variables: dict[str, VariableSpec] | None,
    ) -> PlotData:
        """
        Add, replace, or drop variables and return as a new dataset.

        Parameters
        ----------
        data
            Input data source; if None, inherits from parent dataset.
        variables
            Dictionary mapping variable names to their specifications.
            Passing None for a variable will remove it from the dataset.

        Returns
        -------
        New PlotData instance with updated variables.
        """
        if data is None:
            data = self.source_data

        if not variables:
            variables = self.source_vars

        disinherit = [k for k, v in variables.items() if v is None]

        new = PlotData(data, variables)

        drop_cols = [k for k in self.frame if k in new.frame or k in disinherit]
        parts = [self.frame.drop(columns=drop_cols), new.frame]

        frame = pd.concat(parts, axis=1, sort=False)

        names = {k: v for k, v in self.names.items() if k not in disinherit}
        names.update(new.names)

        ids = {k: v for k, v in self.ids.items() if k not in disinherit}
        ids.update(new.ids)

        new.frame = frame
        new.names = names
        new.ids = ids

        new.source_data = self.source_data
        new.source_vars = self.source_vars

        return new

    def _assign_variables(
        self,
        data: DataFrame | Mapping | None,
        variables: dict[str, VariableSpec],
    ) -> tuple[DataFrame, dict[str, str | None], dict[str, str | int]]:
        """
        Assign values for plot variables given long-form data and/or vector inputs.

        Parameters
        ----------
        data
            Input data where variable names map to vector values.
        variables
            Keys are names of plot variables (x, y, ...) each value is one of:

            - name of a column (or index level, or dictionary entry) in `data`
            - vector in any format that can construct a :class:`pandas.DataFrame`

        Returns
        -------
        frame
            Table mapping seaborn variables (x, y, color, ...) to data vectors.
        names
            Keys are defined seaborn variables; values are names inferred from
            the inputs (or None when no name can be determined).
        ids
            Like the `names` dict, but `None` values are replaced by the `id()`
            of the data object that defined the variable.

        Raises
        ------
        TypeError
            When data source is not a DataFrame or Mapping.
        ValueError
            When variables are strings that don't appear in `data`, or when they are
            non-indexed vector datatypes that have a different length from `data`.

        """
        source_data: Mapping | DataFrame
        frame: DataFrame
        names: dict[str, str | None]
        ids: dict[str, str | int]

        plot_data: dict[str, Any] = {}
        names = {}
        ids = {}

        given_data = data is not None
        if data is None:
            source_data = {}
        else:
            source_data = data

        if isinstance(source_data, pd.DataFrame):
            index = source_data.index.to_frame().to_dict("series")
        else:
            index = {}

        for key, val in variables.items():

            if val is None:
                continue

            try:
                hash(val)
                val_is_hashable = True
            except TypeError:
                val_is_hashable = False

            val_as_data_key = (
                (val_is_hashable and val in source_data)
                or (isinstance(val, str) and val in index)
            )

            if val_as_data_key:
                val = cast(ColumnName, val)
                if val in source_data:
                    plot_data[key] = source_data[val]
                elif val in index:
                    plot_data[key] = index[val]
                names[key] = ids[key] = str(val)

            elif isinstance(val, str):

                err = f"Could not interpret value `{val}` for `{key}`. "
                if not given_data:
                    err += "Value is a string, but `data` was not passed."
                else:
                    err += "An entry with this name does not appear in `data`."
                raise ValueError(err)

            else:

                if isinstance(val, Sized) and len(val) == 0:
                    continue

                if isinstance(data, pd.DataFrame) and not isinstance(val, pd.Series):
                    if isinstance(val, Sized) and len(data) != len(val):
                        val_cls = val.__class__.__name__
                        err = (
                            f"Length of {val_cls} vectors must match length of `data`"
                            f" when both are used, but `data` has length {len(data)}"
                            f" and the vector passed to `{key}` has length {len(val)}."
                        )
                        raise ValueError(err)

                plot_data[key] = val

                if hasattr(val, "name"):
                    names[key] = ids[key] = str(val.name)  # type: ignore[attr-defined]
                else:
                    names[key] = None
                    ids[key] = id(val)

        frame = pd.DataFrame(plot_data)

        return frame, names, ids


def handle_data_source(data: DataSource) -> pd.DataFrame | Mapping | None:
    """
    Convert the data source object to a common union representation.

    Parameters
    ----------
    data
        Input data source; may be a DataFrame, Mapping, or object with
        a `to_pandas` method.

    Returns
    -------
    Standardized representation: DataFrame, Mapping, or None.

    Raises
    ------
    TypeError
        If data source cannot be converted to a usable format.
    RuntimeError
        If conversion to pandas DataFrame fails.
    """
    if isinstance(data, pd.DataFrame) or isinstance(data, Mapping) or data is None:
        return data
    elif hasattr(data, "to_pandas"):
        try:
            df = data.to_pandas()  # type: ignore[union-attr]
        except Exception as err:
            msg = (
                "Encountered an exception when converting data source "
                "to a pandas DataFrame. See traceback above for details."
            )
            raise RuntimeError(msg) from err
        if isinstance(df, pd.DataFrame):
            return df

    msg = f"Data source must be a DataFrame or Mapping, not {type(data)!r}."
    raise TypeError(msg)
