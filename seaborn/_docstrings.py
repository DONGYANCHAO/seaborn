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
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    TYPE_CHECKING,
)
from collections.abc import Mapping
from enum import Enum, auto

from .external.docscrape import NumpyDocString


if TYPE_CHECKING:
    from typing_extensions import Self


T = TypeVar("T")


class DocstringStyle(Enum):
    """Supported docstring styles."""
    NUMPY = auto()
    GOOGLE = auto()


class LanguageCode(Enum):
    """Supported language codes for multi-language documentation."""
    EN = "en"
    ZH_CN = "zh_CN"


@dataclass
class Localization:
    """
    Multi-language support for docstrings.

    Attributes
    ----------
    default_language : LanguageCode
        The default language for documentation.
    translations : dict
        Dictionary mapping language codes to translated strings.
    """
    default_language: LanguageCode = LanguageCode.EN
    translations: Dict[LanguageCode, Dict[str, str]] = field(default_factory=dict)

    def get(self, key: str, language: Optional[LanguageCode] = None) -> str:
        """Get a translated string for the given key."""
        lang = language or self.default_language
        if lang in self.translations and key in self.translations[lang]:
            return self.translations[lang][key]
        if LanguageCode.EN in self.translations and key in self.translations[LanguageCode.EN]:
            return self.translations[LanguageCode.EN][key]
        return key

    def register(self, language: LanguageCode, key: str, value: str) -> None:
        """Register a translation for a key."""
        if language not in self.translations:
            self.translations[language] = {}
        self.translations[language][key] = value


_localization = Localization()


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
    def from_nested_components(cls, **kwargs: "DocstringComponents") -> "Self":
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
    def from_function_params(cls, func: Callable) -> "Self":
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

    def merge(self, other: "DocstringComponents") -> "Self":
        """
        Merge another DocstringComponents into this one.

        Parameters
        ----------
        other : DocstringComponents
            Components to merge.

        Returns
        -------
        DocstringComponents
            New merged components.
        """
        merged = self.entries.copy()
        merged.update(other.entries)
        return DocstringComponents(merged, strip_whitespace=False)


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
    version_added : str, optional
        Version when this parameter was added.
    version_changed : str, optional
        Version when this parameter was changed.
    """
    name: str
    type_hint: str
    description: str
    default: Any = None
    required: bool = True
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    aliases: Optional[List[str]] = None
    version_added: Optional[str] = None
    version_changed: Optional[str] = None

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

    def with_override(
        self,
        type_hint: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        required: Optional[bool] = None,
        deprecated: Optional[bool] = None,
        deprecation_message: Optional[str] = None,
    ) -> "ParamSpec":
        """
        Create a copy with overridden values.

        Parameters
        ----------
        type_hint : str, optional
            Override type hint.
        description : str, optional
            Override description.
        default : Any, optional
            Override default value.
        required : bool, optional
            Override required flag.
        deprecated : bool, optional
            Override deprecated flag.
        deprecation_message : str, optional
            Override deprecation message.

        Returns
        -------
        ParamSpec
            New ParamSpec with overridden values.
        """
        return ParamSpec(
            name=self.name,
            type_hint=type_hint if type_hint is not None else self.type_hint,
            description=description if description is not None else self.description,
            default=default if default is not None else self.default,
            required=required if required is not None else self.required,
            deprecated=deprecated if deprecated is not None else self.deprecated,
            deprecation_message=deprecation_message if deprecation_message is not None else self.deprecation_message,
            aliases=self.aliases,
            version_added=self.version_added,
            version_changed=self.version_changed,
        )


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
        self._aliases: Dict[str, str] = {}

    @classmethod
    def get_instance(cls) -> "ParameterRegistry":
        """Get the singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        name: str,
        spec: ParamSpec,
        category: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
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
        aliases : list of str, optional
            Alternative names for this parameter.
        """
        self._params[name] = spec
        if category:
            if category not in self._categories:
                self._categories[category] = []
            if name not in self._categories[category]:
                self._categories[category].append(name)
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def get(self, name: str) -> Optional[ParamSpec]:
        """Get a parameter specification by name or alias."""
        if name in self._aliases:
            name = self._aliases[name]
        return self._params.get(name)

    def get_by_category(self, category: str) -> List[ParamSpec]:
        """Get all parameters in a category."""
        names = self._categories.get(category, [])
        return [self._params[name] for name in names]

    def get_names_by_category(self, category: str) -> List[str]:
        """Get all parameter names in a category."""
        return self._categories.get(category, [])

    def create_components(
        self,
        names: Optional[List[str]] = None,
        category: Optional[str] = None,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> DocstringComponents:
        """
        Create DocstringComponents from registered parameters.

        Parameters
        ----------
        names : list of str, optional
            Specific parameter names to include.
        category : str, optional
            Category of parameters to include.
        overrides : dict, optional
            Dictionary of overrides for specific parameters.

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

        overrides = overrides or {}
        comp_dict = {}

        for name in names:
            spec = self.get(name)
            if spec:
                if name in overrides:
                    spec = spec.with_override(**overrides[name])
                comp_dict[name] = spec.to_docstring()

        return DocstringComponents(comp_dict)

    def list_categories(self) -> List[str]:
        """List all registered categories."""
        return list(self._categories.keys())

    def list_params(self) -> List[str]:
        """List all registered parameter names."""
        return list(self._params.keys())


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
    _param_order: List[str] = field(default_factory=list)

    def set_summary(self, summary: str) -> "Self":
        """Set the one-line summary."""
        self.summary = summary
        return self

    def set_extended_summary(self, extended_summary: str) -> "Self":
        """Set the extended description."""
        self.extended_summary = extended_summary
        return self

    def add_parameter(
        self,
        param: Union[ParamSpec, str],
        type_hint: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> "Self":
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
        if param.name not in self._param_order:
            self._param_order.append(param.name)
        return self

    def add_parameters_from_registry(
        self,
        names: List[str],
        registry: Optional[ParameterRegistry] = None,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        order: Optional[List[str]] = None,
    ) -> "Self":
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
        order : list of str, optional
            Custom order for parameters.

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
                    spec = spec.with_override(**overrides[name])
                self.parameters.append(spec)
                if name not in self._param_order:
                    self._param_order.append(name)

        return self

    def inherit_parameters(
        self,
        func: Callable,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> "Self":
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
            if p.name not in self._param_order:
                self._param_order.append(p.name)

        return self

    def set_returns(self, returns: str) -> "Self":
        """Set the return value documentation."""
        self.returns = returns
        return self

    def set_raises(self, raises: Dict[str, str]) -> "Self":
        """Set the exception documentation."""
        self.raises = raises
        return self

    def set_examples(self, examples: str) -> "Self":
        """Set the usage examples."""
        self.examples = examples
        return self

    def set_see_also(self, see_also: Dict[str, str]) -> "Self":
        """Set the related functions/classes."""
        self.see_also = see_also
        return self

    def set_notes(self, notes: str) -> "Self":
        """Set additional notes."""
        self.notes = notes
        return self

    def set_references(self, references: str) -> "Self":
        """Set references."""
        self.references = references
        return self

    def add_conditional_section(
        self,
        condition: bool,
        section: str,
        content: str,
    ) -> "Self":
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

    def build(self, style: DocstringStyle = DocstringStyle.NUMPY) -> str:
        """
        Build the final docstring.

        Parameters
        ----------
        style : DocstringStyle, default=NUMPY
            Docstring style to use.

        Returns
        -------
        str
            Complete docstring.
        """
        if style == DocstringStyle.NUMPY:
            return self._build_numpy()
        elif style == DocstringStyle.GOOGLE:
            return self._build_google()
        else:
            raise ValueError(f"Unsupported docstring style: {style}")

    def _build_numpy(self) -> str:
        """Build numpy-style docstring."""
        parts = []

        if self.summary:
            parts.append(self.summary)
            parts.append("")

        if self.extended_summary:
            parts.append(self.extended_summary)
            parts.append("")

        all_params: Dict[str, ParamSpec] = {}
        for param in self._inherited_params.values():
            all_params[param.name] = param
        for param in self.parameters:
            all_params[param.name] = param

        if all_params:
            parts.append("Parameters")
            parts.append("----------")
            for name in self._param_order:
                if name in all_params:
                    parts.append(all_params[name].to_docstring())
            for name, param in all_params.items():
                if name not in self._param_order:
                    parts.append(param.to_docstring())
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

    def _build_google(self) -> str:
        """Build google-style docstring."""
        parts = []

        if self.summary:
            parts.append(self.summary)
            parts.append("")

        if self.extended_summary:
            parts.append(self.extended_summary)
            parts.append("")

        all_params: Dict[str, ParamSpec] = {}
        for param in self._inherited_params.values():
            all_params[param.name] = param
        for param in self.parameters:
            all_params[param.name] = param

        if all_params:
            parts.append("Args:")
            for name in self._param_order:
                if name in all_params:
                    param = all_params[name]
                    parts.append(f"    {name} ({param.type_hint}): {param.description}")
            parts.append("")

        if self.returns:
            parts.append("Returns:")
            parts.append(f"    {self.returns}")
            parts.append("")

        if self.raises:
            parts.append("Raises:")
            for exc, desc in self.raises.items():
                parts.append(f"    {exc}: {desc}")
            parts.append("")

        if self.examples:
            parts.append("Examples:")
            parts.append(self.examples)
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


def merge_docstrings(
    *funcs: Callable,
    exclude_params: Optional[List[str]] = None,
) -> str:
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


def _init_parameter_registry() -> ParameterRegistry:
    """
    Initialize the global parameter registry with common parameters.

    Returns
    -------
    ParameterRegistry
        Initialized registry.
    """
    registry = ParameterRegistry()

    registry.register(
        "data",
        ParamSpec(
            name="data",
            type_hint=":class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence",
            description="""Input data structure. Either a long-form collection of vectors that can be
assigned to named variables or a wide-form dataset that will be internally
reshaped.""",
        ),
        category="core",
    )

    registry.register(
        "x",
        ParamSpec(
            name="x",
            type_hint="vector or key in ``data``",
            description="Variable that specifies positions on the x axis.",
        ),
        category="core",
    )

    registry.register(
        "y",
        ParamSpec(
            name="y",
            type_hint="vector or key in ``data``",
            description="Variable that specifies positions on the y axis.",
        ),
        category="core",
    )

    registry.register(
        "hue",
        ParamSpec(
            name="hue",
            type_hint="vector or key in ``data``",
            description="Semantic variable that is mapped to determine the color of plot elements.",
        ),
        category="semantic",
    )

    registry.register(
        "size",
        ParamSpec(
            name="size",
            type_hint="vector or key in ``data``",
            description="Semantic variable that is mapped to determine the size of plot elements.",
        ),
        category="semantic",
    )

    registry.register(
        "style",
        ParamSpec(
            name="style",
            type_hint="vector or key in ``data``",
            description="Semantic variable that is mapped to determine the style of plot elements.",
        ),
        category="semantic",
    )

    registry.register(
        "palette",
        ParamSpec(
            name="palette",
            type_hint="string, list, dict, or :class:`matplotlib.colors.Colormap`",
            description="""Method for choosing the colors to use when mapping the ``hue`` semantic.
String values are passed to :func:`color_palette`. List or dict values
imply categorical mapping, while a colormap object implies numeric mapping.""",
        ),
        category="semantic",
    )

    registry.register(
        "hue_order",
        ParamSpec(
            name="hue_order",
            type_hint="vector of strings",
            description="""Specify the order of processing and plotting for categorical levels of the
``hue`` semantic.""",
        ),
        category="semantic",
    )

    registry.register(
        "hue_norm",
        ParamSpec(
            name="hue_norm",
            type_hint="tuple or :class:`matplotlib.colors.Normalize`",
            description="""Either a pair of values that set the normalization range in data units
or an object that will map from data units into a [0, 1] interval. Usage
implies numeric mapping.""",
        ),
        category="semantic",
    )

    registry.register(
        "color",
        ParamSpec(
            name="color",
            type_hint=":mod:`matplotlib color <matplotlib.colors>`",
            description="""Single color specification for when hue mapping is not used. Otherwise, the
plot will try to hook into the matplotlib property cycle.""",
        ),
        category="aesthetic",
    )

    registry.register(
        "ax",
        ParamSpec(
            name="ax",
            type_hint=":class:`matplotlib.axes.Axes`",
            description="""Pre-existing axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca`
internally.""",
        ),
        category="output",
    )

    registry.register(
        "order",
        ParamSpec(
            name="order",
            type_hint="vector of strings",
            description="Specify the order of plotting for categorical levels.",
        ),
        category="categorical",
    )

    registry.register(
        "hue_order_cat",
        ParamSpec(
            name="hue_order",
            type_hint="vector of strings",
            description="Specify the order of processing and plotting for categorical levels of the ``hue`` semantic.",
        ),
        category="categorical",
        aliases=["hue_order"],
    )

    registry.register(
        "orient",
        ParamSpec(
            name="orient",
            type_hint='"v" | "h" | "x" | "y"',
            description="""Orientation of the plot (vertical or horizontal). This is usually
inferred based on the type of the input variables, but it can be used
to resolve ambiguity when both `x` and `y` are numeric.""",
        ),
        category="categorical",
    )

    registry.register(
        "stat",
        ParamSpec(
            name="stat",
            type_hint='string',
            description="""Aggregate statistic to compute in each bin.""",
        ),
        category="distribution",
    )

    registry.register(
        "bins",
        ParamSpec(
            name="bins",
            type_hint='string, number, vector, or a pair of such values',
            description="""Generic bin parameter that can be the name of a reference rule,
the number of bins, or the breaks of the bins.""",
        ),
        category="distribution",
    )

    registry.register(
        "binwidth",
        ParamSpec(
            name="binwidth",
            type_hint='number or pair of numbers',
            description="Width of each bin, overrides ``bins`` but can be used with ``binrange``.",
        ),
        category="distribution",
    )

    registry.register(
        "binrange",
        ParamSpec(
            name="binrange",
            type_hint='pair of numbers or a pair of pairs',
            description="Lowest and highest value for bin edges; can be used either with ``bins`` or ``binwidth``.",
        ),
        category="distribution",
    )

    registry.register(
        "discrete",
        ParamSpec(
            name="discrete",
            type_hint='bool or pair of bools',
            description="If True, set ``binwidth`` and ``binrange`` such that bin edges cover integer values in the dataset.",
        ),
        category="distribution",
    )

    registry.register(
        "cumulative",
        ParamSpec(
            name="cumulative",
            type_hint='bool',
            description="If True, return the cumulative statistic.",
        ),
        category="distribution",
    )

    registry.register(
        "element",
        ParamSpec(
            name="element",
            type_hint='"bars" | "step" | "poly"',
            description="Visual representation of the histogram statistic.",
        ),
        category="distribution",
    )

    registry.register(
        "fill",
        ParamSpec(
            name="fill",
            type_hint='bool',
            description="If True, fill in the area under univariate density curve or between bars.",
        ),
        category="distribution",
    )

    registry.register(
        "kde",
        ParamSpec(
            name="kde",
            type_hint='bool',
            description="If True, compute a kernel density estimate to smooth the distribution and show on the plot.",
        ),
        category="distribution",
    )

    registry.register(
        "bw_method",
        ParamSpec(
            name="bw_method",
            type_hint='string, scalar, or callable',
            description="""Method for determining the smoothing bandwidth to use; passed to
:class:`scipy.stats.gaussian_kde`.""",
        ),
        category="kde",
    )

    registry.register(
        "bw_adjust",
        ParamSpec(
            name="bw_adjust",
            type_hint='number',
            description="""Factor that multiplicatively scales the value chosen using
``bw_method``. Increasing will make the curve smoother.""",
        ),
        category="kde",
    )

    registry.register(
        "gridsize",
        ParamSpec(
            name="gridsize",
            type_hint='int',
            description="Number of points on each dimension of the evaluation grid.",
        ),
        category="kde",
    )

    registry.register(
        "cut",
        ParamSpec(
            name="cut",
            type_hint='number',
            description="""Factor, multiplied by the smoothing bandwidth, that determines how
far the evaluation grid extends past the extreme datapoints. When
set to 0, truncate the curve at the data limits.""",
        ),
        category="kde",
    )

    registry.register(
        "clip",
        ParamSpec(
            name="clip",
            type_hint='pair of numbers or None, or a pair of such pairs',
            description="Do not evaluate the density outside of these limits.",
        ),
        category="kde",
    )

    registry.register(
        "estimator",
        ParamSpec(
            name="estimator",
            type_hint='name of pandas method or callable or None',
            description="Method for aggregating across multiple observations of the `y` variable at the same `x` level.",
        ),
        category="aggregation",
    )

    registry.register(
        "errorbar",
        ParamSpec(
            name="errorbar",
            type_hint='string, (string, number) tuple, or callable',
            description="""Name of errorbar method (either "ci", "pi", "se", or "sd"), or a tuple
with a method name and a level parameter, or a function that maps from a
vector to a (min, max) interval, or None to hide errorbar.""",
        ),
        category="aggregation",
    )

    registry.register(
        "n_boot",
        ParamSpec(
            name="n_boot",
            type_hint='int',
            description="Number of bootstraps to use for computing the confidence interval.",
        ),
        category="aggregation",
    )

    registry.register(
        "seed",
        ParamSpec(
            name="seed",
            type_hint='int, numpy.random.Generator, or numpy.random.RandomState',
            description="Seed or random number generator for reproducible bootstrapping.",
        ),
        category="aggregation",
    )

    registry.register(
        "legend",
        ParamSpec(
            name="legend",
            type_hint='"auto", "brief", "full", or False',
            description="""How to draw the legend. If "brief", numeric `hue` and `size`
variables will be represented with a sample of evenly spaced values.
If "full", every group will get an entry in the legend. If "auto",
choose between brief or full representation based on number of levels.
If `False`, no legend data is added and no legend is drawn.""",
        ),
        category="output",
    )

    return registry


_registry = _init_parameter_registry()


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
    subsets with convenient methods for batch-setting of attributes.
    """,
    jointgrid="""
:class:`JointGrid`
    An object with multiple subplots for bivariate visualization.
    """,
    pairgrid="""
:class:`PairGrid`
    An object with multiple subplots for pairwise relationships.
    """,
)


_core_seealso = dict(
    scatterplot="""
scatterplot : Plot data points with semantic mappings.
    """,
    lineplot="""
lineplot : Plot data as lines with semantic mappings.
    """,
    histplot="""
histplot : Plot univariate or bivariate histograms.
    """,
    kdeplot="""
kdeplot : Plot univariate or bivariate kernel density estimates.
    """,
    ecdfplot="""
ecdfplot : Plot empirical cumulative distribution functions.
    """,
    rugplot="""
rugplot : Plot a tick at each observation value along the x and/or y axes.
    """,
    displot="""
displot : Figure-level interface to distribution plots.
    """,
    relplot="""
relplot : Figure-level interface to relational plots.
    """,
    catplot="""
catplot : Figure-level interface to categorical plots.
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
pairplot : Plot pairwise relationships in a dataset.
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
    seealso=DocstringComponents(_core_seealso),
)


def get_registry() -> ParameterRegistry:
    """
    Get the global parameter registry.

    Returns
    -------
    ParameterRegistry
        The global parameter registry instance.
    """
    return _registry
