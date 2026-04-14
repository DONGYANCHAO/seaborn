from __future__ import annotations

import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING, Any, Callable, Generic, Literal, Protocol,
    Tuple, Optional, ClassVar, TypeVar, TypedDict, Union, cast, overload
)

import numpy as np
import matplotlib as mpl
from matplotlib.ticker import (
    Locator,
    Formatter,
    AutoLocator,
    AutoMinorLocator,
    FixedLocator,
    LinearLocator,
    LogLocator,
    SymmetricalLogLocator,
    MaxNLocator,
    MultipleLocator,
    EngFormatter,
    FuncFormatter,
    LogFormatterSciNotation,
    ScalarFormatter,
    StrMethodFormatter,
)
from matplotlib.dates import (
    AutoDateLocator,
    AutoDateFormatter,
    ConciseDateFormatter,
)
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series

from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default

if TYPE_CHECKING:
    from seaborn._core.plot import Plot
    from seaborn._core.properties import Property
    from numpy.typing import ArrayLike, NDArray

    TransFuncs = Tuple[
        Callable[[ArrayLike], ArrayLike], Callable[[ArrayLike], ArrayLike]]
    Pipeline = Sequence[Optional[Callable[[Any], Any]]]


S = TypeVar("S", bound="Scale")


class TickParams(TypedDict, total=False):
    """Type specification for tick configuration."""
    locator: Locator | None
    formatter: Formatter | None
    counts: tuple[int, int] | None


class LabelParams(TypedDict, total=False):
    """Type specification for label configuration."""
    formatter: Formatter | str | None
    base: float | None
    label: str | None


class ScaleProtocol(Protocol):
    """Protocol defining the Scale interface."""

    _pipeline: Pipeline

    def __call__(self, data: Series) -> ArrayLike:
        ...


class Scale:
    """
    Base class for objects that map data values to visual properties.

    Scales handle the mapping from data values to visual property coordinates.
    They control axis ticking, labeling, and value transformation pipelines.
    """

    values: tuple | str | list | dict | None

    _priority: ClassVar[int]
    _pipeline: Pipeline
    _matplotlib_scale: ScaleBase | None
    _spacer: staticmethod | None
    _legend: tuple[list[Any], list[str]] | None
    _tick_params: dict[str, Any]
    _label_params: dict[str, Any]

    def __post_init__(self):
        """Initialize post-dataclass state."""
        self._tick_params = {}
        self._label_params = {}
        self._legend = None

    def tick(self: S, **kwargs: Any) -> S:
        """
        Configure ticking parameters for this scale.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the underlying tick locator.

        Returns
        -------
        Scale instance with updated tick parameters.
        """
        raise NotImplementedError()

    def label(self: S, **kwargs: Any) -> S:
        """
        Configure labeling parameters for this scale.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to the underlying label formatter.

        Returns
        -------
        Scale instance with updated label parameters.
        """
        raise NotImplementedError()

    def _get_locators(self, **kwargs: Any) -> tuple[Locator, Locator | None]:
        """Return major and minor tick locators for this scale."""
        raise NotImplementedError()

    def _get_formatter(self, locator: Locator | None = None, **kwargs: Any) -> Formatter:
        """Return tick label formatter for this scale."""
        raise NotImplementedError()

    def _get_scale(
        self, name: str, forward: Callable, inverse: Callable
    ) -> ScaleBase:
        """
        Create a custom matplotlib scale with this scale's transform.

        Parameters
        ----------
        name
            Name for the custom scale.
        forward
            Forward transformation function.
        inverse
            Inverse transformation function.

        Returns
        -------
        Matplotlib FuncScale instance with configured locators and formatters.
        """
        major_locator, minor_locator = self._get_locators(**self._tick_params)
        major_formatter = self._get_formatter(major_locator, **self._label_params)

        class InternalScale(mpl.scale.FuncScale):
            def set_default_locators_and_formatters(self, axis: Axis) -> None:
                axis.set_major_locator(major_locator)
                if minor_locator is not None:
                    axis.set_minor_locator(minor_locator)
                axis.set_major_formatter(major_formatter)

        return InternalScale(name, (forward, inverse))

    def _spacing(self, x: Series) -> float:
        """
        Compute spacing for discrete data points.

        Parameters
        ----------
        x
            Input data series.

        Returns
        -------
        Typical spacing between consecutive data values.
        """
        if self._spacer is None:
            return 1.0
        space = self._spacer(x)
        if np.isnan(space):
            return 1
        return float(space)

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Scale:
        """
        Configure the scale based on data and property information.

        Parameters
        ----------
        data
            Input data series to fit the scale to.
        prop
            Visual property this scale maps to.
        axis
            Matplotlib axis for axis scales.

        Returns
        -------
        Configured scale instance.
        """
        raise NotImplementedError()

    def _finalize(self, p: Plot, axis: Axis) -> None:
        """Perform scale-specific axis tweaks after adding artists."""
        pass

    def __call__(self, data: Series | Any) -> ArrayLike | Any:
        """
        Apply scale transformations to input data.

        Parameters
        ----------
        data
            Input data series or scalar value.

        Returns
        -------
        Transformed data values.
        """
        trans_data: Series | NDArray | list

        scalar_data = np.isscalar(data)
        if scalar_data:
            trans_data = np.array([data])
        else:
            trans_data = data

        for func in self._pipeline:
            if func is not None:
                trans_data = func(trans_data)

        if scalar_data:
            return trans_data[0]
        return trans_data

    @staticmethod
    def _identity() -> Scale:
        """Return an identity scale that performs no transformation."""
        class Identity(Scale):
            _pipeline = []
            _spacer = None
            _legend = None
            _matplotlib_scale = None

        return Identity()


@dataclass
class Boolean(Scale):
    """
    Scale for boolean data.

    Maps True/False values to discrete visual coordinates.
    """
    _priority: ClassVar[int] = 1

    def tick(self: Boolean, **kwargs: Any) -> Boolean:
        self._tick_params.update(kwargs)
        return self

    def label(self: Boolean, **kwargs: Any) -> Boolean:
        self._label_params.update(kwargs)
        return self

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Boolean:

        map_func = [False, True]

        def mapper(x: Series) -> Series:
            return x.map(dict(zip(map_func, [0, 1])))

        self._pipeline = [mapper]
        self._spacer = None
        self._matplotlib_scale = None

        if self.values is None:
            self.values = map_func

        return self

    def _get_locators(self, **kwargs: Any) -> tuple[Locator, Locator | None]:
        return FixedLocator([0, 1]), None

    def _get_formatter(self, locator: Locator | None = None, **kwargs: Any) -> Formatter:
        if self.values is None or isinstance(self.values, (tuple, list)):
            labs = self.values or [False, True]
        elif isinstance(self.values, dict):
            labs = [self.values.get(x, str(x)) for x in [False, True]]
        else:
            labs = [False, True]
        return FixedFormatter(labs)


@dataclass
class Nominal(Scale):
    """
    Scale for categorical (unordered) data.

    Maps discrete data values to visual coordinates.
    """
    order: list | None = None

    _priority: ClassVar[int] = 2

    def tick(self: Nominal, **kwargs: Any) -> Nominal:
        self._tick_params.update(kwargs)
        return self

    def label(self: Nominal, **kwargs: Any) -> Nominal:
        self._label_params.update(kwargs)
        return self

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Nominal:

        if self.order is None:
            levels = categorical_order(data)
        else:
            levels = self.order

        def mapper(x: Series) -> Series:
            return x.map(dict(zip(levels, range(len(levels)))))

        def spacer(x: Series) -> float:
            return float(np.mean(np.diff(np.sort(x.unique()))))

        self._pipeline = [mapper]
        self._spacer = staticmethod(spacer)
        self._matplotlib_scale = None

        if self.values is None:
            self.values = levels

        return self

    def _get_locators(self, **kwargs: Any) -> tuple[Locator, Locator | None]:
        n = len(self.values) if self.values is not None else 0
        return FixedLocator(range(n)), None

    def _get_formatter(self, locator: Locator | None = None, **kwargs: Any) -> Formatter:
        if self.values is None:
            return FixedFormatter([])
        if isinstance(self.values, (tuple, list)):
            labs = self.values
        elif isinstance(self.values, dict):
            labs = [self.values.get(x, str(x)) for x in self.values]
        else:
            labs = []
        return FixedFormatter(labs)


@dataclass
class Ordinal(Nominal):
    """
    Scale for ordered categorical data.

    An ordered variant of Nominal scale.
    """
    _priority: ClassVar[int] = 3


@dataclass
class Continuous(Scale):
    """
    Scale for continuous numeric data.

    Parameters
    ----------
    transform
        Name of a matplotlib scale transform (e.g., "log", "symlog", "sqrt").
    """
    transform: str | None = None

    _priority: ClassVar[int] = 5

    def tick(self: Continuous, **kwargs: Any) -> Continuous:
        self._tick_params.update(kwargs)
        return self

    def label(self: Continuous, **kwargs: Any) -> Continuous:
        self._label_params.update(kwargs)
        return self

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Continuous:

        forward, inverse = self._get_transform_funcs()

        def spacer(x: Series) -> float:
            return np.nanmedian(np.diff(np.sort(np.unique(x))))

        self._pipeline: Pipeline = []
        self._spacer = staticmethod(spacer)

        if self.transform is not None:
            if axis is not None:
                self._matplotlib_scale = self._get_scale(self.transform, forward, inverse)
            else:
                self._pipeline.append(forward)

        return self

    def _get_transform_funcs(self) -> TransFuncs:
        """Return forward and inverse functions for scale transformation."""
        if self.transform is None:
            return lambda x: x, lambda x: x

        elif self.transform == "log":
            base = self._label_params.get("base", 10)
            log_base = np.log(base) if base != 1 else 1
            return (
                lambda x, base=base, log_base=log_base: np.log(x) / log_base,
                lambda x, base=base: np.power(base, x),
            )
        elif self.transform == "sqrt":
            return np.sqrt, np.square
        elif self.transform == "pow":
            return np.square, np.sqrt
        else:
            return lambda x: x, lambda x: x

    def _get_locators(self, **kwargs: Any) -> tuple[Locator, Locator | None]:
        if self.transform == "log":
            base = self._tick_params.get("base", 10)
            return LogLocator(base=base), LogLocator(base=base, subs="auto")
        elif self.transform == "symlog":
            base = self._tick_params.get("base", 10)
            linthresh = self._tick_params.get("linthresh", 0.1)
            return SymmetricalLogLocator(base=base, linthresh=linthresh), None
        else:
            return MaxNLocator(integer=True, **kwargs), AutoMinorLocator()

    def _get_formatter(self, locator: Locator | None = None, **kwargs: Any) -> Formatter:
        if self.transform == "log":
            return LogFormatterSciNotation(base=kwargs.get("base", 10))
        elif "unit" in kwargs:
            return EngFormatter(unit=kwargs["unit"])
        else:
            return ScalarFormatter(useMathText=True)


@dataclass
class Temporal(Scale):
    """
    Scale for datetime/timedelta data.

    Maps temporal data to visual coordinates with proper date formatting.
    """
    _priority: ClassVar[int] = 4

    def tick(self: Temporal, **kwargs: Any) -> Temporal:
        self._tick_params.update(kwargs)
        return self

    def label(self: Temporal, **kwargs: Any) -> Temporal:
        self._label_params.update(kwargs)
        return self

    def _setup(
        self, data: Series, prop: Property, axis: Axis | None = None,
    ) -> Temporal:

        def spacer(x: Series) -> float:
            return np.nanmedian(np.diff(np.sort(np.unique(x)))).astype(float)

        self._pipeline: Pipeline = []
        self._spacer = staticmethod(spacer)
        self._matplotlib_scale = None

        return self

    def _get_locators(self, **kwargs: Any) -> tuple[Locator, Locator | None]:
        return AutoDateLocator(**kwargs), None

    def _get_formatter(self, locator: Locator | None = None, **kwargs: Any) -> Formatter:
        if locator is not None:
            return ConciseDateFormatter(locator)
        return AutoDateFormatter(AutoDateLocator())


def FixedFormatter(values: Sequence[Any]) -> FuncFormatter:
    """Create a formatter for fixed categorical tick labels."""
    fmt = FuncFormatter(lambda x, pos: values[int(x)] if 0 <= int(x) < len(values) else "")
    return fmt
