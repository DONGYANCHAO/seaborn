"""
Flexible docstring assembly system for seaborn.

This module provides a comprehensive system for building docstrings with support for:
- Parameter inheritance from shared parameter libraries
- Conditional content inclusion based on context
- Future multi-language extension capabilities
- Type-safe docstring composition
"""
from __future__ import annotations

import re
import pydoc
from typing import Any, Callable, ClassVar, Protocol, runtime_checkable
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .external.docscrape import NumpyDocString


# =================================================================================== #
# Core Types and Protocols
# =================================================================================== #

@runtime_checkable
class DocstringProvider(Protocol):
    """Protocol for objects that can provide docstring content."""

    def get_docstring(self, key: str) -> str | None:
        """Retrieve docstring content for a given key."""
        ...


class DocstringError(Exception):
    """Exception raised for docstring assembly errors."""
    pass


# =================================================================================== #
# Parameter Definition Classes
# =================================================================================== #

@dataclass(frozen=True)
class ParameterDef:
    """
    Definition of a function parameter for documentation.

    Attributes
    ----------
    name : str
        Parameter name.
    type_str : str
        Type annotation as a string.
    description : str
        Parameter description.
    optional : bool
        Whether the parameter is optional.
    default : Any
        Default value if optional.
    conditions : tuple[str, ...]
        Conditions under which this parameter applies.
    see_also : tuple[str, ...]
        Related parameters or functions.
    version_added : str | None
        Version when this parameter was added.
    deprecated : str | None
        Deprecation message if parameter is deprecated.
    examples : str | None
        Usage examples for this parameter.
    """
    name: str
    type_str: str
    description: str
    optional: bool = False
    default: Any = None
    conditions: tuple[str, ...] = field(default_factory=tuple)
    see_also: tuple[str, ...] = field(default_factory=tuple)
    version_added: str | None = None
    deprecated: str | None = None
    examples: str | None = None

    def to_docstring(self, include_metadata: bool = False) -> str:
        """Convert parameter definition to docstring format."""
        lines = [f"{self.name} : {self.type_str}"]

        if self.deprecated:
            lines.append(f"    .. deprecated:: {self.deprecated}")
            lines.append("")

        desc_lines = self.description.strip().split('\n')
        for desc_line in desc_lines:
            lines.append(f"    {desc_line}")

        if self.examples and include_metadata:
            lines.append("")
            lines.append(f"    Examples: {self.examples}")

        if self.see_also and include_metadata:
            lines.append("")
            see_also_str = ", ".join(self.see_also)
            lines.append(f"    See also: {see_also_str}")

        return "\n".join(lines)


@dataclass(frozen=True)
class ReturnDef:
    """Definition of a return value for documentation."""
    type_str: str
    description: str

    def to_docstring(self) -> str:
        """Convert return definition to docstring format."""
        lines = [f"{self.type_str}", "    " + self.description.strip()]
        return "\n".join(lines)


@dataclass(frozen=True)
class SeeAlsoDef:
    """Definition of a see-also reference."""
    name: str
    description: str

    def to_docstring(self) -> str:
        """Convert see-also definition to docstring format."""
        return f"{self.name} : {self.description}"


# =================================================================================== #
# Parameter Library
# =================================================================================== #

class ParameterLibrary:
    """
    Central repository for shared parameter definitions.

    This class provides a unified parameter library that can be used
    across all seaborn functions to ensure documentation consistency.
    """

    # Data-related parameters
    DATA = ParameterDef(
        name="data",
        type_str=":class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence",
        description="""
Input data structure. Either a long-form collection of vectors that can be
assigned to named variables or a wide-form dataset that will be internally
reshaped.
        """.strip(),
    )

    X = ParameterDef(
        name="x",
        type_str="vector or key in ``data``",
        description="Variable that specifies positions on the x axis.",
    )

    Y = ParameterDef(
        name="y",
        type_str="vector or key in ``data``",
        description="Variable that specifies positions on the y axis.",
    )

    XY = ParameterDef(
        name="x, y",
        type_str="vectors or keys in ``data``",
        description="Variables that specify positions on the x and y axes.",
    )

    # Semantic mapping parameters
    HUE = ParameterDef(
        name="hue",
        type_str="vector or key in ``data``",
        description="Semantic variable that is mapped to determine the color of plot elements.",
    )

    SIZE = ParameterDef(
        name="size",
        type_str="vector or key in ``data``",
        description="Semantic variable that is mapped to determine the size of plot elements.",
    )

    STYLE = ParameterDef(
        name="style",
        type_str="vector or key in ``data``",
        description="Semantic variable that is mapped to determine the style (e.g., marker or line style) of plot elements.",
    )

    # Palette and color parameters
    PALETTE = ParameterDef(
        name="palette",
        type_str="string, list, dict, or :class:`matplotlib.colors.Colormap`",
        description="""
Method for choosing the colors to use when mapping the ``hue`` semantic.
String values are passed to :func:`color_palette`. List or dict values
imply categorical mapping, while a colormap object implies numeric mapping.
        """.strip(),
    )

    COLOR = ParameterDef(
        name="color",
        type_str=":mod:`matplotlib color <matplotlib.colors>`",
        description="""
Single color specification for when hue mapping is not used. Otherwise, the
plot will try to hook into the matplotlib property cycle.
        """.strip(),
    )

    # Order parameters
    HUE_ORDER = ParameterDef(
        name="hue_order",
        type_str="vector of strings",
        description="Specify the order of processing and plotting for categorical levels of the ``hue`` semantic.",
    )

    SIZE_ORDER = ParameterDef(
        name="size_order",
        type_str="vector of strings",
        description="Specify the order of processing and plotting for categorical levels of the ``size`` semantic.",
    )

    STYLE_ORDER = ParameterDef(
        name="style_order",
        type_str="vector of strings",
        description="Specify the order of processing and plotting for categorical levels of the ``style`` semantic.",
    )

    ORDER = ParameterDef(
        name="order",
        type_str="vector of strings",
        description="Specify the order of processing and plotting for categorical levels.",
    )

    # Normalization parameters
    HUE_NORM = ParameterDef(
        name="hue_norm",
        type_str="tuple or :class:`matplotlib.colors.Normalize`",
        description="""
Either a pair of values that set the normalization range in data units
or an object that will map from data units into a [0, 1] interval. Usage
implies numeric mapping.
        """.strip(),
    )

    SIZE_NORM = ParameterDef(
        name="size_norm",
        type_str="tuple or :class:`matplotlib.colors.Normalize`",
        description="Normalization in data units for scaling when the ``size`` variable is numeric.",
    )

    # Axes parameters
    AX = ParameterDef(
        name="ax",
        type_str=":class:`matplotlib.axes.Axes`",
        description="""
Pre-existing axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca`
internally.
        """.strip(),
    )

    # Legend parameters
    LEGEND = ParameterDef(
        name="legend",
        type_str='"auto", "brief", "full", or False',
        description="""
How to draw the legend. If "brief", numeric semantic variables will be
represented with a sample of evenly spaced values. If "full", every group
will get an entry in the legend. If "auto", choose between brief or full
representation based on number of levels. If ``False``, no legend is drawn.
        """.strip(),
    )

    # Statistical parameters
    ESTIMATOR = ParameterDef(
        name="estimator",
        type_str="name of pandas method, callable, or None",
        description="Method for aggregating across multiple observations at the same level. If ``None``, all observations will be drawn.",
    )

    CI = ParameterDef(
        name="ci",
        type_str="int, \"sd\", or None",
        description="Size of the confidence interval to draw when aggregating.",
        deprecated="0.12.0",
    )

    N_BOOT = ParameterDef(
        name="n_boot",
        type_str="int",
        description="Number of bootstraps to use for computing the confidence interval.",
    )

    SEED = ParameterDef(
        name="seed",
        type_str="int, :class:`numpy.random.Generator`, or :class:`numpy.random.RandomState`",
        description="Seed or random number generator for reproducible bootstrapping.",
    )

    UNITS = ParameterDef(
        name="units",
        type_str="vector or key in ``data``",
        description="""
Grouping variable identifying sampling units. When used, a separate
line will be drawn for each unit with appropriate semantics, but no
legend entry will be added. Useful for showing distribution of
experimental replicates when exact identities are not needed.
        """.strip(),
    )

    # Return definitions
    RETURN_AX = ReturnDef(
        type_str=":class:`matplotlib.axes.Axes`",
        description="The matplotlib axes containing the plot.",
    )

    RETURN_FACETGRID = ReturnDef(
        type_str=":class:`FacetGrid`",
        description="""
An object managing one or more subplots that correspond to conditional data
subsets with convenient methods for batch-setting of axes attributes.
        """.strip(),
    )

    RETURN_JOINTGRID = ReturnDef(
        type_str=":class:`JointGrid`",
        description="""
An object managing multiple subplots that correspond to joint and marginal axes
for plotting a bivariate relationship or distribution.
        """.strip(),
    )

    RETURN_PAIRGRID = ReturnDef(
        type_str=":class:`PairGrid`",
        description="""
An object managing multiple subplots that correspond to joint and marginal axes
for pairwise combinations of multiple variables in a dataset.
        """.strip(),
    )

    @classmethod
    def get(cls, name: str) -> ParameterDef | ReturnDef | SeeAlsoDef | None:
        """Retrieve a parameter definition by name."""
        return getattr(cls, name.upper(), None)

    @classmethod
    def get_many(cls, *names: str) -> dict[str, ParameterDef | ReturnDef | SeeAlsoDef]:
        """Retrieve multiple parameter definitions."""
        return {name: cls.get(name) for name in names if cls.get(name) is not None}


# =================================================================================== #
# Enhanced Docstring Components
# =================================================================================== #

class DocstringComponents:
    """
    Enhanced docstring component assembly system.

    This class provides flexible docstring construction with support for
    parameter inheritance, conditional content, and extensibility.

    Parameters
    ----------
    components : dict or ParameterLibrary
        Dictionary mapping component names to docstring content, or a
        ParameterLibrary instance to extract parameters from.
    strip_whitespace : bool
        If True, strip outer whitespace from component values.
    inherit_from : DocstringComponents | None
        Parent components to inherit from.
    condition : str | None
        Condition that must be met for components to be included.

    Examples
    --------
    Basic usage with dictionary:

    >>> params = DocstringComponents({
    ...     'x': 'x : array\n    Input data.',
    ...     'y': 'y : array\n    Output data.'
    ... })
    >>> print(params.x)
    x : array
        Input data.

    Using parameter library:

    >>> params = DocstringComponents.from_parameters('data', 'x', 'y')
    >>> print(params.data)
    data : pandas.DataFrame, numpy.ndarray, mapping, or sequence
        Input data structure...

    Inheritance and composition:

    >>> base = DocstringComponents.from_parameters('data', 'x', 'y')
    >>> extended = DocstringComponents({'custom': '...'}, inherit_from=base)
    """

    regexp = re.compile(r"\n((\n|.)+)\n\s*", re.MULTILINE)

    def __init__(
        self,
        components: dict[str, str] | ParameterLibrary | None = None,
        *,
        strip_whitespace: bool = True,
        inherit_from: DocstringComponents | None = None,
        condition: str | None = None,
    ):
        self._entries: dict[str, str] = {}
        self._inherit_from = inherit_from
        self._condition = condition
        self._metadata: dict[str, Any] = {}

        if isinstance(components, ParameterLibrary):
            # Extract all parameter definitions from library
            components = self._extract_from_library(components)

        if components:
            if strip_whitespace:
                for key, val in components.items():
                    m = re.match(self.regexp, val)
                    if m is None:
                        self._entries[key] = val
                    else:
                        self._entries[key] = m.group(1)
            else:
                self._entries.update(components)

    def _extract_from_library(self, library: ParameterLibrary) -> dict[str, str]:
        """Extract docstrings from ParameterLibrary."""
        components = {}
        for attr_name in dir(library):
            if attr_name.startswith('_'):
                continue
            attr = getattr(library, attr_name)
            if isinstance(attr, (ParameterDef, ReturnDef)):
                components[attr_name.lower()] = attr.to_docstring()
        return components



    def __contains__(self, key: str) -> bool:
        """Check if a component exists."""
        if key in self._entries:
            return True
        if self._inherit_from is not None:
            return key in self._inherit_from
        return False

    def get(self, key: str, default: str | None = None) -> str | None:
        """Get a component by key with optional default."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def merge(self, *others: DocstringComponents) -> DocstringComponents:
        """Merge multiple DocstringComponents into a new instance."""
        merged_entries = dict(self._entries)
        for other in others:
            merged_entries.update(other._entries)
        return DocstringComponents(merged_entries, strip_whitespace=False)

    def with_condition(self, condition: str) -> DocstringComponents:
        """Create a new instance with a condition applied."""
        new = DocstringComponents(
            self._entries.copy(),
            strip_whitespace=False,
            inherit_from=self._inherit_from,
            condition=condition,
        )
        return new

    def select(self, *keys: str) -> DocstringComponents:
        """Select specific components to create a subset."""
        selected = {k: v for k, v in self._entries.items() if k in keys}
        return DocstringComponents(selected, strip_whitespace=False)

    def exclude(self, *keys: str) -> DocstringComponents:
        """Exclude specific components."""
        filtered = {k: v for k, v in self._entries.items() if k not in keys}
        return DocstringComponents(filtered, strip_whitespace=False)

    @classmethod
    def from_nested_components(cls, **kwargs: DocstringComponents) -> DocstringComponents:
        """Add multiple sub-sets of components with nested access support."""
        # Create a new instance that supports nested attribute access
        instance = cls({}, strip_whitespace=False)

        # Store nested components for dot-style access (e.g., params.core.hue)
        instance._nested: dict[str, DocstringComponents] = {}

        for comp_name, comp_obj in kwargs.items():
            # Store the nested component
            instance._nested[comp_name] = comp_obj
            # Also flatten entries with prefix for backward compatibility
            for key, val in comp_obj._entries.items():
                instance._entries[f"{comp_name}.{key}"] = val

        return instance

    def __getattr__(self, attr: str) -> str | DocstringComponents | None:
        """Provide dot access to entries for clean raw docstrings."""
        # Check own entries first
        if attr in self._entries:
            return self._entries[attr]

        # Check nested components (for nested access like params.core.hue)
        if hasattr(self, '_nested') and attr in self._nested:
            return self._nested[attr]

        # Check inherited entries
        if self._inherit_from is not None:
            try:
                return getattr(self._inherit_from, attr)
            except AttributeError:
                pass

        # Handle -OO optimization
        if not __debug__:
            return None

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{attr}'")

    @classmethod
    def from_function_params(cls, func: Callable) -> DocstringComponents:
        """Use the numpydoc parser to extract components from existing func."""
        params = NumpyDocString(pydoc.getdoc(func))["Parameters"]
        comp_dict: dict[str, str] = {}
        for p in params:
            name = p.name
            type_ = p.type
            desc = "\n    ".join(p.desc)
            comp_dict[name] = f"{name} : {type_}\n    {desc}"

        return cls(comp_dict)

    @classmethod
    def from_parameters(cls, *param_names: str) -> DocstringComponents:
        """Create components from ParameterLibrary definitions."""
        components = {}
        for name in param_names:
            param = ParameterLibrary.get(name)
            if param is not None:
                components[name] = param.to_docstring()
        return cls(components, strip_whitespace=False)

    def build_docstring(
        self,
        description: str | None = None,
        parameters: Sequence[str] | None = None,
        returns: Sequence[str] | None = None,
        see_also: Sequence[str] | None = None,
        examples: str | None = None,
        notes: str | None = None,
        references: str | None = None,
    ) -> str:
        """
        Build a complete docstring from components.

        Parameters
        ----------
        description : str | None
            Main function description.
        parameters : Sequence[str] | None
            Parameter names to include in order.
        returns : Sequence[str] | None
            Return value names to include.
        see_also : Sequence[str] | None
            See-also references to include.
        examples : str | None
            Usage examples.
        notes : str | None
            Additional notes.
        references : str | None
            References section.

        Returns
        -------
        str
            Complete formatted docstring.
        """
        sections = []

        if description:
            sections.append(description.strip())

        if parameters:
            param_docs = []
            for param in parameters:
                doc = self.get(param)
                if doc:
                    param_docs.append(doc)
            if param_docs:
                sections.append("\n".join(["Parameters", "----------"] + param_docs))

        if returns:
            return_docs = []
            for ret in returns:
                doc = self.get(ret)
                if doc:
                    return_docs.append(doc)
            if return_docs:
                sections.append("\n".join(["Returns", "-------"] + return_docs))

        if see_also:
            see_also_docs = [f"    {ref}" for ref in see_also]
            sections.append("\n".join(["See Also", "--------"] + see_also_docs))

        if examples:
            sections.append(f"Examples\n--------\n{examples}")

        if notes:
            sections.append(f"Notes\n-----\n{notes}")

        if references:
            sections.append(f"References\n----------\n{references}")

        return "\n\n".join(sections)


# =================================================================================== #
# Legacy Support - Maintaining backward compatibility
# =================================================================================== #

# Legacy core parameters (kept for backward compatibility)
_core_params = dict(
    data="""
data : :class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence
    Input data structure. Either a long-form collection of vectors that can be
    assigned to named variables or a wide-form dataset that will be internally
    reshaped.
    """,  # TODO add link to user guide narrative when exists
    xy="""
x, y : vectors or keys in ``data``
    Variables that specify positions on the x and y axes.
    """,
    hue="""
hue : vector or key in ``data``
    Semantic variable that is mapped to determine the color of plot elements.
    """,
    palette="""
palette : string, list, dict, or :class:`matplotlib.colors.Colormap`
    Method for choosing the colors to use when mapping the ``hue`` semantic.
    String values are passed to :func:`color_palette`. List or dict values
    imply categorical mapping, while a colormap object implies numeric mapping.
    """,  # noqa: E501
    hue_order="""
hue_order : vector of strings
    Specify the order of processing and plotting for categorical levels of the
    ``hue`` semantic.
    """,
    hue_norm="""
hue_norm : tuple or :class:`matplotlib.colors.Normalize`
    Either a pair of values that set the normalization range in data units
    or an object that will map from data units into a [0, 1] interval. Usage
    implies numeric mapping.
    """,
    color="""
color : :mod:`matplotlib color <matplotlib.colors>`
    Single color specification for when hue mapping is not used. Otherwise, the
    plot will try to hook into the matplotlib property cycle.
    """,
    ax="""
ax : :class:`matplotlib.axes.Axes`
    Pre-existing axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca`
    internally.
    """,  # noqa: E501
)

_core_returns = dict(
    ax="""
:class:`matplotlib.axes.Axes`
    The matplotlib axes containing the plot.
    """,
    facetgrid="""
:class:`FacetGrid`
    An object managing one or more subplots that correspond to conditional data
    subsets with convenient methods for batch-setting of axes attributes.
    """,
    jointgrid="""
:class:`JointGrid`
    An object managing multiple subplots that correspond to joint and marginal axes
    for plotting a bivariate relationship or distribution.
    """,
    pairgrid="""
:class:`PairGrid`
    An object managing multiple subplots that correspond to joint and marginal axes
    for pairwise combinations of multiple variables in a dataset.
    """,
)

_seealso_blurbs = dict(
    # Relational plots
    scatterplot="""
scatterplot : Plot data using points.
    """,
    lineplot="""
lineplot : Plot data using lines.
    """,

    # Distribution plots
    displot="""
displot : Figure-level interface to distribution plot functions.
    """,
    histplot="""
histplot : Plot a histogram of binned counts with optional normalization or smoothing.
    """,
    kdeplot="""
kdeplot : Plot univariate or bivariate distributions using kernel density estimation.
    """,
    ecdfplot="""
ecdfplot : Plot empirical cumulative distribution functions.
    """,
    rugplot="""
rugplot : Plot a tick at each observation value along the x and/or y axes.
    """,

    # Categorical plots
    stripplot="""
stripplot : Plot a categorical scatter with jitter.
    """,
    swarmplot="""
swarmplot : Plot a categorical scatter with non-overlapping points.
    """,
    violinplot="""
violinplot : Draw an enhanced boxplot using kernel density estimation.
    """,
    pointplot="""
pointplot : Plot point estimates and CIs using markers and lines.
    """,

    # Multiples
    jointplot="""
jointplot : Draw a bivariate plot with univariate marginal distributions.
    """,
    pairplot="""
jointplot : Draw multiple bivariate plots with univariate marginal distributions.
    """,
    jointgrid="""
JointGrid : Set up a figure with joint and marginal views on bivariate data.
    """,
    pairgrid="""
PairGrid : Set up a figure with joint and marginal views on multiple variables.
    """,
)

# Legacy _core_docs for backward compatibility
_core_docs = dict(
    params=DocstringComponents(_core_params),
    returns=DocstringComponents(_core_returns),
    seealso=DocstringComponents(_seealso_blurbs),
)


# =================================================================================== #
# Multi-language Support (Future Extension)
# =================================================================================== #

class I18nDocstringManager:
    """
    Internationalization support for docstrings.

    This class provides infrastructure for multi-language docstring support.
    Currently a placeholder for future implementation.
    """

    _translations: ClassVar[dict[str, dict[str, str]]] = {}
    _current_language: ClassVar[str] = "en"

    @classmethod
    def set_language(cls, language: str) -> None:
        """Set the current language for docstrings."""
        cls._current_language = language

    @classmethod
    def register_translation(cls, language: str, translations: dict[str, str]) -> None:
        """Register translations for a language."""
        cls._translations[language] = translations

    @classmethod
    def get_translation(cls, key: str, language: str | None = None) -> str | None:
        """Get a translated string."""
        lang = language or cls._current_language
        if lang in cls._translations:
            return cls._translations[lang].get(key)
        return None


# =================================================================================== #
# Convenience Functions
# =================================================================================== #

def docstring_from(
    *components: DocstringComponents,
    parameters: Sequence[str] | None = None,
    returns: Sequence[str] | None = None,
    description: str | None = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator to apply docstring components to a function.

    Parameters
    ----------
    *components : DocstringComponents
        Components to merge and apply.
    parameters : Sequence[str] | None
        Parameter names to include.
    returns : Sequence[str] | None
        Return value names to include.
    description : str | None
        Function description.

    Returns
    -------
    Callable
        Decorator function.
    """
    def decorator(func: Callable) -> Callable:
        merged = DocstringComponents({})
        for comp in components:
            merged = merged.merge(comp)

        docstring = merged.build_docstring(
            description=description or func.__doc__,
            parameters=parameters,
            returns=returns,
        )
        func.__doc__ = docstring
        return func
    return decorator
