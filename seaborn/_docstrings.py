"""
Flexible docstring assembly system for seaborn.

This module provides a comprehensive system for building and managing docstrings
with support for:
- Parameter inheritance across functions
- Conditional content inclusion
- Multi-language extensibility
- Unified parameter library to reduce duplication
"""
from __future__ import annotations

import re
import pydoc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, TypeVar, Union
from collections.abc import Mapping

from .external.docscrape import NumpyDocString


T = TypeVar("T")


class DocstringComponents:
    """
    Container for reusable docstring components with dot access.

    Parameters
    ----------
    comp_dict : dict
        Dictionary mapping attribute names to docstring components.
    strip_whitespace : bool, default=True
        Whether to strip leading/trailing whitespace from components.

    Examples
    --------
    >>> params = {"x": "x : int\\n    An integer parameter."}
    >>> comps = DocstringComponents(params)
    >>> print(comps.x)
    x : int
        An integer parameter.
    """

    regexp = re.compile(r"\n((\n|.)+)\n\s*", re.MULTILINE)

    def __init__(self, comp_dict: Dict[str, str], strip_whitespace: bool = True):
        if strip_whitespace:
            entries = {}
            for key, val in comp_dict.items():
                m = re.match(self.regexp, val)
                if m is None:
                    entries[key] = val
                else:
                    entries[key] = m.group(1)
        else:
            entries = comp_dict.copy()

        self.entries = entries

    def __getattr__(self, attr: str) -> str:
        if attr in self.entries:
            return self.entries[attr]
        else:
            try:
                return self.__getattribute__(attr)
            except AttributeError as err:
                if __debug__:
                    raise err
                else:
                    pass

    @classmethod
    def from_nested_components(cls, **kwargs: "DocstringComponents") -> "DocstringComponents":
        """
        Combine multiple DocstringComponents into a single container.

        Parameters
        ----------
        **kwargs : DocstringComponents
            Named components to combine.

        Returns
        -------
        DocstringComponents
            Combined components with nested access.
        """
        return cls(kwargs, strip_whitespace=False)

    @classmethod
    def from_function_params(cls, func: Callable) -> "DocstringComponents":
        """
        Extract parameter documentation from a function's docstring.

        Parameters
        ----------
        func : callable
            Function with numpydoc-style docstring.

        Returns
        -------
        DocstringComponents
            Components for each parameter.
        """
        params = NumpyDocString(pydoc.getdoc(func))["Parameters"]
        comp_dict = {}
        for p in params:
            name = p.name
            type_str = p.type
            desc = "\n    ".join(p.desc)
            comp_dict[name] = f"{name} : {type_str}\n    {desc}"

        return cls(comp_dict)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a component by key with optional default."""
        return self.entries.get(key, default)

    def keys(self):
        """Return all component keys."""
        return self.entries.keys()

    def items(self):
        """Return all component items."""
        return self.entries.items()


@dataclass
class ParamSpec:
    """
    Specification for a single parameter in a docstring.

    Attributes
    ----------
    name : str
        Parameter name.
    type_hint : str
        Type annotation string.
    description : str
        Parameter description.
    default : Any, optional
        Default value for the parameter.
    required : bool, default=True
        Whether the parameter is required.
    deprecated : bool, default=False
        Whether the parameter is deprecated.
    deprecation_message : str, optional
        Deprecation warning message.
    aliases : list of str, optional
        Alternative names for this parameter.
    """
    name: str
    type_hint: str
    description: str
    default: Any = None
    required: bool = True
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    aliases: Optional[List[str]] = None

    def to_docstring(self, indent: int = 4) -> str:
        """
        Convert to numpydoc-style docstring component.

        Parameters
        ----------
        indent : int, default=4
            Number of spaces for description indentation.

        Returns
        -------
        str
            Formatted docstring component.
        """
        indent_str = " " * indent
        desc_lines = self.description.strip().split("\n")
        desc_indented = f"\n{indent_str}".join(desc_lines)

        result = f"{self.name} : {self.type_hint}\n{indent_str}{desc_indented}"

        if self.deprecated and self.deprecation_message:
            result += f"\n\n{indent_str}.. deprecated::\n{indent_str}    {self.deprecation_message}"

        return result


class ParameterRegistry:
    """
    Registry for shared parameter definitions to reduce duplication.

    This class provides a centralized location for defining parameters
    that are used across multiple functions, enabling consistent
    documentation and easy updates.

    Examples
    --------
    >>> registry = ParameterRegistry()
    >>> registry.register("data", ParamSpec(
    ...     name="data",
    ...     type_hint="DataFrame or dict",
    ...     description="Input data structure."
    ... ))
    >>> registry.get("data")
    ParamSpec(name='data', ...)
    """

    _instance: Optional["ParameterRegistry"] = None

    def __init__(self):
        self._params: Dict[str, ParamSpec] = {}
        self._categories: Dict[str, List[str]] = {}

    @classmethod
    def get_instance(cls) -> "ParameterRegistry":
        """Get the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, spec: ParamSpec, category: Optional[str] = None) -> None:
        """
        Register a parameter specification.

        Parameters
        ----------
        name : str
            Parameter name.
        spec : ParamSpec
            Parameter specification.
        category : str, optional
            Category to group related parameters.
        """
        self._params[name] = spec
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(name)

    def get(self, name: str) -> Optional[ParamSpec]:
        """Get a parameter specification by name."""
        return self._params.get(name)

    def get_by_category(self, category: str) -> List[ParamSpec]:
        """Get all parameters in a category."""
        names = self._categories.get(category, [])
        return [self._params[name] for name in names]

    def create_components(self, names: Optional[List[str]] = None, category: Optional[str] = None) -> DocstringComponents:
        """
        Create DocstringComponents from registered parameters.

        Parameters
        ----------
        names : list of str, optional
            Specific parameter names to include.
        category : str, optional
            Category of parameters to include.

        Returns
        -------
        DocstringComponents
            Components for the specified parameters.
        """
        if names is None and category is None:
            names = list(self._params.keys())
        elif category is not None:
            names = self._categories.get(category, [])
        elif names is None:
            names = []

        comp_dict = {}
        for name in names:
            spec = self._params.get(name)
            if spec:
                comp_dict[name] = spec.to_docstring()

        return DocstringComponents(comp_dict)


@dataclass
class DocstringBuilder:
    """
    Builder for constructing docstrings with inheritance and composition.

    This class provides a fluent interface for building docstrings
    with support for parameter inheritance, conditional sections,
    and template-based generation.

    Attributes
    ----------
    summary : str, optional
        One-line summary of the function/class.
    extended_summary : str, optional
        Extended description.
    parameters : list of ParamSpec
        Parameter specifications.
    returns : str, optional
        Return value documentation.
    raises : dict, optional
        Exception documentation.
    examples : str, optional
        Usage examples.
    see_also : dict, optional
        Related functions/classes.
    notes : str, optional
        Additional notes.
    references : str, optional
        References.

    Examples
    --------
    >>> builder = DocstringBuilder()
    >>> builder.set_summary("Plot data.")
    >>> builder.add_parameter(ParamSpec("x", "array", "X coordinates."))
    >>> docstring = builder.build()
    """

    summary: Optional[str] = None
    extended_summary: Optional[str] = None
    parameters: List[ParamSpec] = field(default_factory=list)
    returns: Optional[str] = None
    raises: Optional[Dict[str, str]] = None
    examples: Optional[str] = None
    see_also: Optional[Dict[str, str]] = None
    notes: Optional[str] = None
    references: Optional[str] = None
    _inherited_params: Dict[str, ParamSpec] = field(default_factory=dict)

    def set_summary(self, summary: str) -> "DocstringBuilder":
        """Set the one-line summary."""
        self.summary = summary
        return self

    def set_extended_summary(self, extended_summary: str) -> "DocstringBuilder":
        """Set the extended description."""
        self.extended_summary = extended_summary
        return self

    def add_parameter(
        self,
        param: Union[ParamSpec, str],
        type_hint: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> "DocstringBuilder":
        """
        Add a parameter to the docstring.

        Parameters
        ----------
        param : ParamSpec or str
            Parameter specification or name.
        type_hint : str, optional
            Type annotation (if param is a string).
        description : str, optional
            Parameter description (if param is a string).
        **kwargs
            Additional arguments for ParamSpec.

        Returns
        -------
        DocstringBuilder
            Self for method chaining.
        """
        if isinstance(param, str):
            param = ParamSpec(param, type_hint or "Any", description or "", **kwargs)
        self.parameters.append(param)
        return self

    def add_parameters_from_registry(
        self,
        names: List[str],
        registry: Optional[ParameterRegistry] = None,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> "DocstringBuilder":
        """
        Add parameters from the registry.

        Parameters
        ----------
        names : list of str
            Parameter names to add.
        registry : ParameterRegistry, optional
            Registry to use. Defaults to global registry.
        overrides : dict, optional
            Overrides for specific parameters.

        Returns
        -------
        DocstringBuilder
            Self for method chaining.
        """
        if registry is None:
            registry = ParameterRegistry.get_instance()

        overrides = overrides or {}

        for name in names:
            spec = registry.get(name)
            if spec:
                if name in overrides:
                    spec = ParamSpec(
                        name=spec.name,
                        type_hint=overrides[name].get("type_hint", spec.type_hint),
                        description=overrides[name].get("description", spec.description),
                        default=overrides[name].get("default", spec.default),
                        required=overrides[name].get("required", spec.required),
                        deprecated=overrides[name].get("deprecated", spec.deprecated),
                        deprecation_message=overrides[name].get("deprecation_message", spec.deprecation_message),
                    )
                self.parameters.append(spec)

        return self

    def inherit_parameters(
        self,
        func: Callable,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> "DocstringBuilder":
        """
        Inherit parameters from another function's docstring.

        Parameters
        ----------
        func : callable
            Function to inherit parameters from.
        include : list of str, optional
            Specific parameters to include.
        exclude : list of str, optional
            Parameters to exclude.

        Returns
        -------
        DocstringBuilder
            Self for method chaining.
        """
        params = NumpyDocString(pydoc.getdoc(func))["Parameters"]

        for p in params:
            if include and p.name not in include:
                continue
            if exclude and p.name in exclude:
                continue

            spec = ParamSpec(
                name=p.name,
                type_hint=p.type,
                description="\n".join(p.desc),
            )
            self._inherited_params[p.name] = spec

        return self

    def set_returns(self, returns: str) -> "DocstringBuilder":
        """Set the return value documentation."""
        self.returns = returns
        return self

    def set_raises(self, raises: Dict[str, str]) -> "DocstringBuilder":
        """Set the exception documentation."""
        self.raises = raises
        return self

    def set_examples(self, examples: str) -> "DocstringBuilder":
        """Set the usage examples."""
        self.examples = examples
        return self

    def set_see_also(self, see_also: Dict[str, str]) -> "DocstringBuilder":
        """Set the related functions/classes."""
        self.see_also = see_also
        return self

    def set_notes(self, notes: str) -> "DocstringBuilder":
        """Set additional notes."""
        self.notes = notes
        return self

    def set_references(self, references: str) -> "DocstringBuilder":
        """Set references."""
        self.references = references
        return self

    def add_conditional_section(
        self,
        condition: bool,
        section: str,
        content: str
    ) -> "DocstringBuilder":
        """
        Add a section conditionally.

        Parameters
        ----------
        condition : bool
            Whether to include the section.
        section : str
            Section name ('notes', 'examples', etc.).
        content : str
            Section content.

        Returns
        -------
        DocstringBuilder
            Self for method chaining.
        """
        if condition:
            setattr(self, section, content)
        return self

    def build(self, style: str = "numpy") -> str:
        """
        Build the final docstring.

        Parameters
        ----------
        style : str, default="numpy"
            Docstring style ('numpy' or 'google').

        Returns
        -------
        str
            Complete docstring.
        """
        parts = []

        if self.summary:
            parts.append(self.summary)
            parts.append("")

        if self.extended_summary:
            parts.append(self.extended_summary)
            parts.append("")

        all_params = list(self._inherited_params.values()) + self.parameters
        if all_params:
            parts.append("Parameters")
            parts.append("----------")
            seen = set()
            for param in all_params:
                if param.name not in seen:
                    parts.append(param.to_docstring())
                    seen.add(param.name)
            parts.append("")

        if self.returns:
            parts.append("Returns")
            parts.append("-------")
            parts.append(self.returns)
            parts.append("")

        if self.raises:
            parts.append("Raises")
            parts.append("------")
            for exc, desc in self.raises.items():
                parts.append(f"{exc}\n    {desc}")
            parts.append("")

        if self.see_also:
            parts.append("See Also")
            parts.append("--------")
            for name, desc in self.see_also.items():
                parts.append(f"{name} : {desc}")
            parts.append("")

        if self.examples:
            parts.append("Examples")
            parts.append("--------")
            parts.append(self.examples)
            parts.append("")

        if self.notes:
            parts.append("Notes")
            parts.append("-----")
            parts.append(self.notes)
            parts.append("")

        if self.references:
            parts.append("References")
            parts.append("----------")
            parts.append(self.references)
            parts.append("")

        return "\n".join(parts)


def compose_docstring(template: str, **components: str) -> str:
    """
    Compose a docstring from a template and components.

    Parameters
    ----------
    template : str
        Template string with {placeholder} markers.
    **components : str
        Named components to substitute.

    Returns
    -------
    str
        Composed docstring.

    Examples
    --------
    >>> template = "Summary.\\n\\nParameters\\n----------\\n{params}"
    >>> params = "x : int\\n    X value."
    >>> docstring = compose_docstring(template, params=params)
    """
    return template.format(**components)


def merge_docstrings(*funcs: Callable, exclude_params: Optional[List[str]] = None) -> str:
    """
    Merge docstrings from multiple functions.

    Parameters
    ----------
    *funcs : callable
        Functions to merge docstrings from.
    exclude_params : list of str, optional
        Parameter names to exclude.

    Returns
    -------
    str
        Merged docstring.
    """
    builder = DocstringBuilder()
    exclude_params = exclude_params or []

    for func in funcs:
        builder.inherit_parameters(func, exclude=exclude_params)

    return builder.build()


_core_params = dict(
    data="""
data : :class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence
    Input data structure. Either a long-form collection of vectors that can be
    assigned to named variables or a wide-form dataset that will be internally
    reshaped.
    """,
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
    """,
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
    """,
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
    scatterplot="""
scatterplot : Plot data using points.
    """,
    lineplot="""
lineplot : Plot data using lines.
    """,
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


_core_docs = dict(
    params=DocstringComponents(_core_params),
    returns=DocstringComponents(_core_returns),
    seealso=DocstringComponents(_seealso_blurbs),
)


_registry = ParameterRegistry()

_registry.register(
    "data",
    ParamSpec(
        name="data",
        type_hint=":class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence",
        description="""Input data structure. Either a long-form collection of vectors that can be
assigned to named variables or a wide-form dataset that will be internally
reshaped.""",
    ),
    category="core"
)

_registry.register(
    "x",
    ParamSpec(
        name="x",
        type_hint="vector or key in ``data``",
        description="Variable that specifies positions on the x axis.",
    ),
    category="coordinates"
)

_registry.register(
    "y",
    ParamSpec(
        name="y",
        type_hint="vector or key in ``data``",
        description="Variable that specifies positions on the y axis.",
    ),
    category="coordinates"
)

_registry.register(
    "hue",
    ParamSpec(
        name="hue",
        type_hint="vector or key in ``data``",
        description="Semantic variable that is mapped to determine the color of plot elements.",
    ),
    category="semantics"
)

_registry.register(
    "palette",
    ParamSpec(
        name="palette",
        type_hint="string, list, dict, or :class:`matplotlib.colors.Colormap`",
        description="""Method for choosing the colors to use when mapping the ``hue`` semantic.
String values are passed to :func:`color_palette`. List or dict values
imply categorical mapping, while a colormap object implies numeric mapping.""",
    ),
    category="semantics"
)

_registry.register(
    "ax",
    ParamSpec(
        name="ax",
        type_hint=":class:`matplotlib.axes.Axes`",
        description="""Pre-existing axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca`
internally.""",
    ),
    category="plotting"
)


def get_global_registry() -> ParameterRegistry:
    """Get the global parameter registry instance."""
    return _registry
