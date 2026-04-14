from __future__ import annotations

import itertools
import warnings
from typing import (
    TYPE_CHECKING, Any, Callable, Generic, Literal, Protocol,
    Tuple, List, Union, Optional, TypeVar, TypedDict, cast, Sequence, Dict
)

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from pandas import Series
import matplotlib as mpl
from matplotlib.colors import to_rgb, to_rgba, to_rgba_array
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path

from seaborn._core.scales import Scale, Boolean, Continuous, Nominal, Temporal
from seaborn._core.rules import categorical_order, variable_type
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette
from seaborn.utils import get_color_cycle

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


RGBTuple = Tuple[float, float, float]
RGBATuple = Tuple[float, float, float, float]
ColorSpec = Union[RGBTuple, RGBATuple, str]

DashPattern = Tuple[float, ...]
DashPatternWithOffset = Tuple[float, Optional[DashPattern]]

MarkerPattern = Union[
    float,
    str,
    Tuple[int, int, float],
    List[Tuple[float, float]],
    Path,
    MarkerStyle,
]

Mapping = Callable[[ArrayLike], ArrayLike]

P = TypeVar("P", bound="Property")


class ColorConfig(TypedDict, total=False):
    """Type specification for color configuration."""
    palette: str | Sequence[ColorSpec] | Colormap | None
    alpha: float
    reverse: bool


class SizeConfig(TypedDict, total=False):
    """Type specification for size configuration."""
    min_size: float
    max_size: float
    range: Tuple[float, float]
    norm: Tuple[float, float] | None


class PropertyProtocol(Protocol):
    """Protocol defining the Property interface."""

    variable: str
    legend: bool
    normed: bool

    def default_scale(self, data: Series) -> Scale:
        ...

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        ...


# =================================================================================== #
# Base classes
# =================================================================================== #


class Property:
    """
    Base class for visual properties that can be set directly or be data scaling.

    Properties define how data values map to visual attributes of plot marks.
    Each property implements default scaling behavior and value mapping logic.

    Parameters
    ----------
    variable
        Name of the corresponding plot variable. If not provided, defaults to
        the lowercase class name.

    Attributes
    ----------
    variable
        Name of the plot variable this property corresponds to.
    legend
        When True, scales for this property will populate the legend by default.
    normed
        When True, scales for this property normalize data to [0, 1] before mapping.
    """

    legend: bool = False
    normed: bool = False
    variable: str

    def __init__(self, variable: str | None = None) -> None:
        """Initialize the property with the name of the corresponding plot variable."""
        if not variable:
            variable = self.__class__.__name__.lower()
        self.variable = variable

    def default_scale(self, data: Series) -> Scale:
        """
        Given data, initialize appropriate scale class.

        Parameters
        ----------
        data
            Input data series to determine scale type from.

        Returns
        -------
        Scale instance appropriate for the data type.
        """
        var_type = variable_type(data, boolean_type="boolean", strict_boolean=True)
        if var_type == "numeric":
            return Continuous()
        elif var_type == "datetime":
            return Temporal()
        elif var_type == "boolean":
            return Boolean()
        else:
            return Nominal()

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        """
        Given data and a scaling argument, initialize appropriate scale class.

        Parameters
        ----------
        arg
            Scale specification argument.
        data
            Input data series.

        Returns
        -------
        Configured Scale instance.

        Raises
        ------
        ValueError
            If arg is an unknown string specification.
        TypeError
            If arg is not a string.
        """
        trans_args = ["log", "symlog", "logit", "pow", "sqrt"]
        if isinstance(arg, str):
            if any(arg.startswith(k) for k in trans_args):
                return Continuous(transform=arg)
            else:
                msg = f"Unknown magic arg for {self.variable} scale: '{arg}'."
                raise ValueError(msg)
        else:
            arg_type = type(arg).__name__
            msg = f"Magic arg for {self.variable} scale must be str, not {arg_type}."
            raise TypeError(msg)

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """
        Return a function that maps from data domain to property range.

        Parameters
        ----------
        scale
            Scale instance defining the transformation.
        data
            Input data series.

        Returns
        -------
        Callable that transforms data values to property values.
        """
        def identity(x: ArrayLike) -> ArrayLike:
            return x
        return identity

    def standardize(self, val: Any = None) -> Any:
        """Coerce flexible property value to standardized representation."""
        return val

    def _check_dict_entries(self, levels: list, values: dict) -> None:
        """
        Input check when values are provided as a dictionary.

        Parameters
        ----------
        levels
            List of expected data levels.
        values
            Dictionary mapping levels to property values.

        Raises
        ------
        ValueError
            If any levels are missing from the dictionary.
        """
        missing = set(levels) - set(values)
        if missing:
            formatted = ", ".join(map(repr, sorted(missing, key=str)))
            err = f"No entry in {self.variable} dictionary for {formatted}"
            raise ValueError(err)

    def _check_list_length(self, levels: list, values: list) -> list:
        """
        Input check when values are provided as a list.

        Parameters
        ----------
        levels
            List of expected data levels.
        values
            List of property values.

        Returns
        -------
        List of values adjusted to match the number of levels.
        """
        message = ""
        if len(levels) > len(values):
            message = " ".join([
                f"\nThe {self.variable} list has fewer values ({len(values)})",
                f"than needed ({len(levels)}) and will cycle, which may",
                "produce an uninterpretable plot."
            ])
            values = [x for _, x in zip(levels, itertools.cycle(values))]

        elif len(values) > len(levels):
            message = " ".join([
                f"The {self.variable} list has more values ({len(values)})",
                f"than needed ({len(levels)}), which may not be intended.",
            ])
            values = values[:len(levels)]

        if message:
            warnings.warn(message, UserWarning)

        return values


# =================================================================================== #
# Properties relating to spatial position of marks on the plotting axes
# =================================================================================== #


class Coordinate(Property):
    """The position of visual marks with respect to the axes of the plot."""
    legend = False
    normed = False


# =================================================================================== #
# Properties with numeric values where scale range can be defined as an interval
# =================================================================================== #


class Float(Property):
    """
    Numeric property with floating-point values in a normalized range.

    Base class for properties like Alpha, Linewidth, etc.
    """
    legend = True
    normed = True

    def _get_values(
        self, scale: Scale, data: Series, min_val: float, max_val: float,
    ) -> Tuple[list[float], list]:
        """
        Get sequence of property values matched to data levels.

        Parameters
        ----------
        scale
            Scale instance with optional explicit values.
        data
            Input data series.
        min_val
            Minimum value in the range.
        max_val
            Maximum value in the range.

        Returns
        -------
        Tuple of (value sequence, data levels).
        """
        levels = categorical_order(data)

        if scale.values is None:
            values = np.linspace(min_val, max_val, len(levels)).tolist()
        elif isinstance(scale.values, dict):
            self._check_dict_entries(levels, scale.values)
            values = [float(scale.values[x]) for x in levels]
        elif isinstance(scale.values, (tuple, list)):
            values = [float(v) for v in self._check_list_length(levels, list(scale.values))]
        else:
            raise ValueError(f"{self.variable} values must be None, list, or dict")

        return values, levels

    def _map(self, x: Series, values: Sequence[float], levels: list) -> Series:
        """Map discrete data values to corresponding property values."""
        mapper = dict(zip(levels, values))
        return x.map(mapper)


class Alpha(Float):
    """Opacity of visual marks, ranging from 0 (transparent) to 1 (opaque)."""

    def default_scale(self, data: Series) -> Continuous:
        """Return a continuous scale for alpha values."""
        return Continuous()

    def get_mapping(self, scale: Scale, data: Series) -> Callable[[Series], Series]:
        """
        Create a mapping function for alpha values.

        Parameters
        ----------
        scale
            Scale instance.
        data
            Input data series.

        Returns
        -------
        Callable that maps data to alpha values in [0.2, 1].
        """
        def interval_map(x: Series) -> Series:
            return 0.2 + 0.8 * x

        levels = categorical_order(data)
        if len(levels) <= 1:
            return lambda x: pd.Series(np.repeat(0.75, len(x)), index=x.index)

        values, levels = self._get_values(scale, data, 0.2, 1.0)

        def mapper(x: Series) -> Series:
            return self._map(x, values, levels)

        return mapper


class Linewidth(Float):
    """Thickness of lines on visual marks, in points."""

    def default_scale(self, data: Series) -> Continuous:
        """Return a continuous scale for linewidth values."""
        return Continuous()

    def get_mapping(self, scale: Scale, data: Series) -> Callable[[Series], Series]:
        """
        Create a mapping function for linewidth values.

        Parameters
        ----------
        scale
            Scale instance.
        data
            Input data series.

        Returns
        -------
        Callable that maps data to linewidth values.
        """
        if isinstance(scale, Nominal):
            values, levels = self._get_values(scale, data, 0.5, 2.5)

            def mapper(x: Series) -> Series:
                return self._map(x, values, levels)

            return mapper
        else:
            def interval_map(x: Series) -> Series:
                return 0.5 + 2.0 * x
            return interval_map


class Edgewidth(Float):
    """Thickness of the border around filled visual marks, in points."""

    def default_scale(self, data: Series) -> Continuous:
        """Return a continuous scale for edgewidth values."""
        return Continuous()

    def get_mapping(self, scale: Scale, data: Series) -> Callable[[Series], Series]:
        """
        Create a mapping function for edgewidth values.

        Parameters
        ----------
        scale
            Scale instance.
        data
            Input data series.

        Returns
        -------
        Callable that maps data to edgewidth values.
        """
        if isinstance(scale, Nominal):
            values, levels = self._get_values(scale, data, 0.25, 1.5)

            def mapper(x: Series) -> Series:
                return self._map(x, values, levels)

            return mapper
        else:
            def interval_map(x: Series) -> Series:
                return 0.25 + 1.25 * x
            return interval_map


class Size(Float):
    """
    Area of visual marks (for point-like marks).

    Note: This property scales mark area, not radius, to ensure perceptual linearity.
    """

    def get_mapping(self, scale: Scale, data: Series) -> Callable[[Series], Series]:
        """
        Create a mapping function for marker size.

        Parameters
        ----------
        scale
            Scale instance.
        data
            Input data series.

        Returns
        -------
        Callable that maps data to marker area in points^2.
        """
        if isinstance(scale, Nominal):
            values, levels = self._get_values(scale, data, 20, 180)

            def mapper(x: Series) -> Series:
                return self._map(x, values, levels)

            return mapper
        else:
            def interval_map(x: Series) -> Series:
                return 20 + 160 * x
            return interval_map

    def standardize(self, val: int | float | None) -> float:
        """
        Standardize marker size value.

        Parameters
        ----------
        val
            Input size value; None defaults to 40.

        Returns
        -------
        Standardized size as float.
        """
        if val is None:
            return 40
        return float(val)


# =================================================================================== #
# Properties with categorical values
# =================================================================================== #


class Color(Property):
    """
    Color of visual marks.

    Supports both discrete and continuous color mapping with optional palette.
    """
    legend = True
    normed = True

    def default_scale(self, data: Series) -> Scale:
        """
        Determine appropriate scale type for color data.

        Parameters
        ----------
        data
            Input data series.

        Returns
        -------
        Nominal or Continuous scale based on data type.
        """
        var_type = variable_type(data, boolean_type="boolean", strict_boolean=True)
        if var_type == "numeric":
            return Continuous()
        elif var_type == "datetime":
            return Continuous()
        elif var_type == "boolean":
            return Boolean()
        else:
            return Nominal()

    def infer_scale(self, arg: str | dict | Sequence | None, data: Series) -> Scale:
        """
        Infer scale from palette or palette name argument.

        Parameters
        ----------
        arg
            Palette specification: name, list of colors, or dictionary.
        data
            Input data series.

        Returns
        -------
        Configured scale with palette values.
        """
        if isinstance(arg, str) and arg in QUAL_PALETTES:
            scale = Nominal()
            scale.values = arg
            return scale
        elif isinstance(arg, str):
            scale = Continuous()
            scale.values = arg
            return scale
        elif isinstance(arg, (dict, list, tuple)):
            scale = self.default_scale(data)
            scale.values = arg
            return scale
        return super().infer_scale(arg, data)

    def get_mapping(self, scale: Scale, data: Series) -> Callable[[Series], NDArray]:
        """
        Create a mapping function for data values to colors.

        Parameters
        ----------
        scale
            Scale instance with optional palette.
        data
            Input data series.

        Returns
        -------
        Callable that maps data to RGBA color arrays.
        """
        levels = categorical_order(data)

        if scale.values is None:
            palette = color_palette(n_colors=len(levels))
        elif isinstance(scale.values, str):
            palette = color_palette(scale.values, n_colors=len(levels))
        elif isinstance(scale.values, dict):
            self._check_dict_entries(levels, scale.values)
            palette = [to_rgb(scale.values[x]) for x in levels]
        elif isinstance(scale.values, (tuple, list)):
            palette = [to_rgb(c) for c in scale.values]
            palette = self._check_list_length(levels, palette)
        else:
            raise ValueError("Color scale values must be None, str, list, or dict")

        palette = cast(List[RGBTuple], palette)
        color_lookup = dict(zip(levels, palette))

        def mapper(x: Series) -> NDArray:
            return to_rgba_array([color_lookup[v] for v in x])

        return mapper


class StrokeColor(Color):
    """Color of the border around filled visual marks."""
    pass


class Marker(Property):
    """
    Symbol used for point-like marks.

    See matplotlib.markers for valid marker specifications.
    """
    legend = True
    normed = False

    def default_scale(self, data: Series) -> Nominal:
        """Return a nominal scale for marker shapes."""
        return Nominal()

    def get_mapping(self, scale: Scale, data: Series) -> Callable[[Series], Series]:
        """
        Create a mapping function for marker symbols.

        Parameters
        ----------
        scale
            Scale instance.
        data
            Input data series.

        Returns
        -------
        Callable that maps data to marker specifications.
        """
        levels = categorical_order(data)

        default_markers = ["o", "X", "^", "s", "D", "v", "p", "*", "h", "8"]

        if scale.values is None:
            markers = default_markers[:len(levels)]
        elif isinstance(scale.values, dict):
            self._check_dict_entries(levels, scale.values)
            markers = [scale.values[x] for x in levels]
        elif isinstance(scale.values, (tuple, list)):
            markers = self._check_list_length(levels, list(scale.values))
        else:
            raise ValueError("Marker values must be None, list, or dict")

        marker_lookup = dict(zip(levels, markers))

        def mapper(x: Series) -> Series:
            return x.map(marker_lookup)

        return mapper

    def standardize(self, val: Any = None) -> MarkerPattern:
        """Standardize marker representation."""
        if val is None:
            return "o"
        return val


class Linestyle(Property):
    """
    Pattern used for drawing lines.

    Valid values: '-', '--', '-.', ':', or custom dash tuple.
    """
    legend = True
    normed = False

    def default_scale(self, data: Series) -> Nominal:
        """Return a nominal scale for line styles."""
        return Nominal()

    def get_mapping(self, scale: Scale, data: Series) -> Callable[[Series], Series]:
        """
        Create a mapping function for line styles.

        Parameters
        ----------
        scale
            Scale instance.
        data
            Input data series.

        Returns
        -------
        Callable that maps data to linestyle specifications.
        """
        levels = categorical_order(data)

        default_styles = ["-", "--", "-.", ":"]

        if scale.values is None:
            styles = default_styles[:len(levels)]
        elif isinstance(scale.values, dict):
            self._check_dict_entries(levels, scale.values)
            styles = [scale.values[x] for x in levels]
        elif isinstance(scale.values, (tuple, list)):
            styles = self._check_list_length(levels, list(scale.values))
        else:
            raise ValueError("Linestyle values must be None, list, or dict")

        style_lookup = dict(zip(levels, styles))

        def mapper(x: Series) -> Series:
            return x.map(style_lookup)

        return mapper


# =================================================================================== #
# Special properties
# =================================================================================== #


class Group(Property):
    """
    Semantic grouping variable without visual aesthetic mapping.

    Used to create separate groups for aggregation or statistical computation
    without producing a corresponding legend.
    """
    legend = False
    normed = False


PROPERTIES: dict[str, type[Property]] = {
    "x": Coordinate,
    "y": Coordinate,
    "color": Color,
    "alpha": Alpha,
    "fill": Color,
    "stroke": StrokeColor,
    "marker": Marker,
    "linestyle": Linestyle,
    "linewidth": Linewidth,
    "edgewidth": Edgewidth,
    "size": Size,
    "group": Group,
}
