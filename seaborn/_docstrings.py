import re
import pydoc
import inspect
from typing import (
    Any, Callable, Dict, Generic, Optional, Protocol,
    TypeVar, Union, cast, TYPE_CHECKING, Set, List, Tuple
)
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass, field

from .external.docscrape import NumpyDocString

if TYPE_CHECKING:
    from typing_extensions import ParamSpec, TypeAlias
    P = ParamSpec("P")
    R = TypeVar("R")

    DocstringComponent: TypeAlias = Union[str, "DocstringTemplate"]
    ParameterMap: TypeAlias = Dict[str, "DocstringComponent"]


@dataclass
class DocstringValidationResult:
    """Result of docstring validation."""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_parameters: List[str] = field(default_factory=list)
    mismatched_parameters: List[str] = field(default_factory=list)
    missing_returns: bool = False
    missing_examples: bool = False
    missing_notes: bool = False

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.passed = False
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def summary(self) -> str:
        """Generate a summary report."""
        lines = []
        if self.passed:
            lines.append("✅ Docstring validation PASSED")
        else:
            lines.append(f"❌ Docstring validation FAILED with {len(self.errors)} errors")

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        if self.missing_parameters:
            lines.append(f"\nMissing parameter docs: {', '.join(self.missing_parameters)}")

        if self.mismatched_parameters:
            lines.append(f"\nMismatched parameters: {', '.join(self.mismatched_parameters)}")

        return "\n".join(lines)


class DocstringValidator:
    """
    Validator for ensuring docstring completeness and consistency.

    This class provides mechanisms to validate that public API functions
    have complete, consistent docstrings matching their signatures.
    """

    def __init__(
        self,
        require_parameters: bool = True,
        require_returns: bool = True,
        require_examples: bool = False,
        require_notes: bool = False,
    ) -> None:
        self.require_parameters = require_parameters
        self.require_returns = require_returns
        self.require_examples = require_examples
        self.require_notes = require_notes
        self._validation_history: Dict[str, DocstringValidationResult] = {}

    def validate_function(
        self,
        func: Callable[..., Any],
        ignore_private: bool = True,
    ) -> DocstringValidationResult:
        """
        Validate a single function's docstring for completeness and consistency.

        Parameters
        ----------
        func : callable
            Function to validate.
        ignore_private : bool
            If True, skip validation for private functions (leading underscore).

        Returns
        -------
        DocstringValidationResult
            Validation result containing errors and warnings.
        """
        result = DocstringValidationResult()
        func_name = getattr(func, "__name__", str(func))

        if ignore_private and func_name.startswith("_"):
            return result

        doc = pydoc.getdoc(func)
        if not doc:
            result.add_error(f"Function '{func_name}' has no docstring")
            self._validation_history[func_name] = result
            return result

        parsed = NumpyDocString(doc)

        try:
            sig = inspect.signature(func)
            actual_params = list(sig.parameters.keys())
        except (ValueError, TypeError):
            actual_params = []

        documented_params = [p.name for p in parsed["Parameters"]]

        for param in actual_params:
            if param not in documented_params and param != "self":
                if param not in ["args", "kwargs", "*args", "**kwargs"]:
                    result.missing_parameters.append(param)
                    result.add_warning(
                        f"Parameter '{param}' in signature but not in docstring"
                    )

        for param in documented_params:
            if param not in actual_params and param not in ["kwargs", "args"]:
                result.mismatched_parameters.append(param)
                result.add_warning(
                    f"Parameter '{param}' in docstring but not in signature"
                )

        if self.require_returns:
            return_anno = func.__annotations__.get("return", None)
            has_return_doc = len(parsed["Returns"]) > 0
            if return_anno is not None and return_anno != type(None):
                if not has_return_doc:
                    result.missing_returns = True
                    result.add_warning("Return value has type annotation but no documentation")

        if self.require_examples and not parsed["Examples"]:
            result.missing_examples = True
            result.add_warning("No Examples section in docstring")

        if self.require_notes and not parsed["Notes"]:
            result.missing_notes = True
            result.add_warning("No Notes section in docstring")

        summary_first_line = parsed["Summary"][0].strip() if parsed["Summary"] else ""
        if not summary_first_line:
            result.add_error("Docstring has no summary line")
        elif not summary_first_line[0].isupper():
            result.add_warning("Summary should start with a capital letter")

        if self.require_parameters and result.missing_parameters:
            missing = ", ".join(result.missing_parameters)
            result.add_error(f"Missing documentation for parameters: {missing}")

        self._validation_history[func_name] = result
        return result

    def validate_module(
        self,
        module: Any,
        public_only: bool = True,
    ) -> Dict[str, DocstringValidationResult]:
        """
        Validate all functions/classes in a module.

        Parameters
        ----------
        module : module
            Module to validate.
        public_only : bool
            If True, only validate public API members.

        Returns
        -------
        dict
            Dictionary mapping function/class names to validation results.
        """
        results = {}

        for name, member in inspect.getmembers(module):
            if public_only and name.startswith("_"):
                continue

            if inspect.isfunction(member):
                results[name] = self.validate_function(member, ignore_private=False)
            elif inspect.isclass(member):
                for method_name, method in inspect.getmembers(member):
                    if inspect.isfunction(method) or inspect.ismethod(method):
                        full_name = f"{name}.{method_name}"
                        results[full_name] = self.validate_function(
                            method, ignore_private=public_only
                        )

        return results

    def get_validation_report(
        self,
        results: Dict[str, DocstringValidationResult],
    ) -> str:
        """
        Generate a human-readable validation report.

        Parameters
        ----------
        results : dict
            Results from validate_module.

        Returns
        -------
        str
            Formatted validation report.
        """
        total = len(results)
        passed = sum(1 for r in results.values() if r.passed)
        failed = total - passed

        lines = [
            "=" * 60,
            f"DOCSTRING VALIDATION REPORT: {passed}/{total} PASSED",
            "=" * 60,
            "",
        ]

        if failed > 0:
            lines.append(f"FAILED ({failed}):")
            for name, result in sorted(results.items()):
                if not result.passed:
                    lines.append(f"\n  {name}:")
                    for err in result.errors:
                        lines.append(f"    ✗ {err}")
                    for warn in result.warnings:
                        lines.append(f"    ⚠ {warn}")

        lines.extend(["", "-" * 60])
        lines.append(f"Summary: {passed} passed, {failed} failed")

        return "\n".join(lines)


class LanguageSupport:
    """
    Multi-language support for docstrings.

    This class provides concrete multi-language docstring translations
    and language switching capabilities.
    """

    SUPPORTED_LANGUAGES: Set[str] = {"en", "zh_CN"}
    DEFAULT_LANGUAGE: str = "en"

    def __init__(self, default_language: str = "en") -> None:
        self._current_language = default_language
        self._translations: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._build_translations()

    def _build_translations(self) -> None:
        """Build concrete translations for all supported languages."""

        self._translations["en"] = self._get_english_translations()
        self._translations["zh_CN"] = self._get_chinese_translations()

    @staticmethod
    def _get_english_translations() -> Dict[str, str]:
        """Get English parameter translations."""
        return {
            "data": """data : :class:`pandas.DataFrame`, :class:`numpy.ndarray`, mapping, or sequence
    Input data structure. Either a long-form collection of vectors that can be
    assigned to named variables or a wide-form dataset that will be internally
    reshaped.
    """,
            "x": """x : vector or key in ``data``
    Variable that specifies positions on the x axis.
    """,
            "y": """y : vector or key in ``data``
    Variable that specifies positions on the y axis.
    """,
            "hue": """hue : vector or key in ``data``
    Semantic variable that is mapped to determine the color of plot elements.
    """,
            "palette": """palette : string, list, dict, or :class:`matplotlib.colors.Colormap`
    Method for choosing the colors to use when mapping the ``hue`` semantic.
    String values are passed to :func:`color_palette`. List or dict values
    imply categorical mapping, while a colormap object implies numeric mapping.
    """,
            "ax": """ax : :class:`matplotlib.axes.Axes`
    Pre-existing axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca`
    internally.
    """,
            "size": """size : vector or key in ``data``
    Semantic variable that is mapped to determine the size of plot elements.
    """,
            "style": """style : vector or key in ``data``
    Semantic variable that is mapped to determine the style of plot elements.
    """,
            "ci": """ci : int or "sd" or None
    Size of the confidence interval when drawing error bars.
    """,
            "seed": """seed : int, numpy.random.Generator, or numpy.random.RandomState
    Seed or random number generator for reproducible bootstrapping.
    """,
        }

    @staticmethod
    def _get_chinese_translations() -> Dict[str, str]:
        """Get Chinese parameter translations (concrete implementation)."""
        return {
            "data": """data : :class:`pandas.DataFrame`、:class:`numpy.ndarray`、字典或序列
    输入数据结构。可以是长格式的向量集合，可以分配给命名变量，
    也可以是宽格式数据集，将在内部进行重塑。
    """,
            "x": """x : 向量或 ``data`` 中的键名
    指定x轴位置的变量。
    """,
            "y": """y : 向量或 ``data`` 中的键名
    指定y轴位置的变量。
    """,
            "hue": """hue : 向量或 ``data`` 中的键名
    语义变量，用于确定绘图元素的颜色。
    """,
            "palette": """palette : 字符串、列表、字典或 :class:`matplotlib.colors.Colormap`
    用于映射 ``hue`` 语义时选择颜色的方法。
    字符串值会传递给 :func:`color_palette`。列表或字典值表示分类映射，
    而colormap对象表示数值映射。
    """,
            "ax": """ax : :class:`matplotlib.axes.Axes`
    预存在的绘图坐标轴。如果未提供，则内部调用 :func:`matplotlib.pyplot.gca`。
    """,
            "size": """size : 向量或 ``data`` 中的键名
    语义变量，用于确定绘图元素的大小。
    """,
            "style": """style : 向量或 ``data`` 中的键名
    语义变量，用于确定绘图元素的样式。
    """,
            "ci": """ci : 整数、"sd" 或 None
    绘制误差棒时的置信区间大小。
    """,
            "seed": """seed : 整数、numpy.random.Generator 或 numpy.random.RandomState
    用于可重复抽样的随机数种子或生成器。
    """,
        }

    @property
    def current_language(self) -> str:
        """Get the current language."""
        return self._current_language

    def set_language(self, language: str) -> None:
        """
        Set the current language for docstrings.

        Parameters
        ----------
        language : str
            Language code (e.g., 'en', 'zh_CN').

        Raises
        ------
        ValueError
            If language is not supported.
        """
        if language not in self.SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(self.SUPPORTED_LANGUAGES))
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Supported languages: {supported}"
            )
        self._current_language = language

    def get_parameter(
        self,
        param_name: str,
        language: Optional[str] = None,
    ) -> str:
        """
        Get a parameter description in the specified language.

        Parameters
        ----------
        param_name : str
            Name of the parameter.
        language : str, optional
            Language code. Uses current language if not specified.

        Returns
        -------
        str
            Parameter description in the requested language.
        """
        lang = language or self._current_language
        if lang not in self._translations:
            lang = self.DEFAULT_LANGUAGE

        return self._translations[lang].get(
            param_name,
            self._translations[self.DEFAULT_LANGUAGE].get(param_name, param_name),
        )

    def get_available_languages(self) -> List[str]:
        """Get list of available languages."""
        return sorted(self.SUPPORTED_LANGUAGES)

    def list_translated_parameters(self, language: str) -> List[str]:
        """List all parameters translated for a given language."""
        return sorted(self._translations.get(language, {}).keys())


class DocstringTemplate:
    """
    A template for building docstrings with conditional content and inheritance.

    This class provides a flexible way to construct docstrings programmatically,
    supporting parameter inheritance, conditional sections, and multi-language
    extensions with concrete implementations.
    """

    def __init__(
        self,
        template: str,
        params: Optional["ParameterMap"] = None,
        language: str = "en",
    ) -> None:
        self.template = template
        self.params = params or {}
        self.language = language
        self._conditions: Dict[str, bool] = {}

    def __add__(self, other: "DocstringTemplate") -> "DocstringTemplate":
        """Combine two docstring templates."""
        combined = self.template + "\n\n" + other.template
        combined_params = {**self.params, **other.params}
        return DocstringTemplate(combined, combined_params, self.language)

    def set_condition(self, name: str, value: bool) -> None:
        """Set a condition for conditional content rendering."""
        self._conditions[name] = value

    def render(
        self,
        params: Optional["ParameterMap"] = None,
        conditions: Optional[Dict[str, bool]] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Render the template with given parameters, conditions, and language.

        Parameters
        ----------
        params : dict, optional
            Additional parameters to use in rendering.
        conditions : dict, optional
            Conditions to evaluate for conditional sections.
        language : str, optional
            Language for parameter lookups (concrete multi-language support).

        Returns
        -------
        str
            The rendered docstring.
        """
        render_params = {**self.params, **(params or {})}
        render_conditions = {**self._conditions, **(conditions or {})}
        render_language = language or self.language
        result = self.template

        for name, value in render_conditions.items():
            pattern = re.compile(
                rf"@IF\({name}\)(.*?)@ENDIF",
                re.DOTALL | re.MULTILINE
            )
            if value:
                result = pattern.sub(r"\1", result)
            else:
                result = pattern.sub("", result)

        lang_support = LanguageSupport(render_language)
        for key, value in render_params.items():
            if isinstance(value, DocstringTemplate):
                value = value.render(params, conditions, language)
            elif isinstance(value, str) and value.startswith("PARAM:"):
                param_name = value[6:]
                value = lang_support.get_parameter(param_name)
            result = result.replace(f"{{{key}}}", str(value))

        param_pattern = re.compile(r"\{PARAM:(\w+)\}")
        for match in param_pattern.finditer(result):
            param_name = match.group(1)
            translated = lang_support.get_parameter(param_name)
            result = result.replace(f"{{PARAM:{param_name}}}", translated)

        return result.strip()


class ParameterLibrary:
    """
    A centralized library for reusable parameter documentation.

    This class stores and manages parameter descriptions, allowing them to be
    reused across multiple functions while ensuring consistency.
    Supports multi-language parameter lookups.
    """

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._parameters: Dict[str, Dict[str, str]] = defaultdict(dict)
        self._returns: Dict[str, str] = {}
        self._seealso: Dict[str, str] = {}
        self._notes: Dict[str, str] = {}
        self._lang_support = LanguageSupport(language)

    @property
    def lang_support(self) -> LanguageSupport:
        """Access the language support instance."""
        return self._lang_support

    def set_language(self, language: str) -> None:
        """Set the current language for parameter lookups."""
        self._lang_support.set_language(language)
        self.language = language

    def add_parameter(
        self,
        name: str,
        description: str,
        category: str = "core",
        language: Optional[str] = None,
    ) -> None:
        """
        Add a parameter to the library.

        Parameters
        ----------
        name : str
            Name of the parameter.
        description : str
            Parameter description in numpydoc format.
        category : str
            Category for organizing parameters (e.g., 'core', 'plotting', 'stats').
        language : str, optional
            Language for this parameter description.
        """
        self._parameters[category][name] = self._normalize_whitespace(description)
        if language:
            self._lang_support._translations[language][name] = description

    def add_parameters(
        self,
        parameters: Dict[str, str],
        category: str = "core",
        language: Optional[str] = None,
    ) -> None:
        """Add multiple parameters to the library."""
        for name, desc in parameters.items():
            self.add_parameter(name, desc, category, language)

    def get(
        self,
        name: str,
        category: str = "core",
        language: Optional[str] = None,
    ) -> str:
        """Get a parameter description with language support."""
        lang_result = self._lang_support.get_parameter(name, language)
        if lang_result != name:
            return lang_result

        try:
            return self._parameters[category][name]
        except KeyError:
            raise KeyError(
                f"Parameter '{name}' not found in category '{category}'"
            ) from None

    def get_many(
        self,
        *names: str,
        category: str = "core",
        language: Optional[str] = None,
    ) -> Dict[str, str]:
        """Get multiple parameter descriptions."""
        return {name: self.get(name, category, language) for name in names}

    def inherit(
        self,
        func: Callable[..., Any],
        *params: str,
    ) -> Dict[str, str]:
        """
        Extract parameter descriptions from an existing function.

        Parameters
        ----------
        func : callable
            Function from which to extract docstring.
        *params : str
            Names of parameters to extract. If empty, extract all.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their descriptions.
        """
        doc = pydoc.getdoc(func)
        if not doc:
            return {}

        parsed = NumpyDocString(doc)
        result = {}

        for p in parsed["Parameters"]:
            if not params or p.name in params:
                name = p.name
                type_str = p.type or ""
                desc = "\n    ".join(p.desc)
                result[name] = f"{name} : {type_str}\n    {desc}"

        return result

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace in docstring entries."""
        match = re.match(r"\n((\n|.)+)\n\s*", text, re.MULTILINE)
        if match:
            return match.group(1)
        return text.strip()


F = TypeVar("F", bound=Callable[..., Any])


class DocstringDecorator(Generic[F]):
    """
    Decorator for applying docstring templates with inheritance.

    This decorator allows functions to inherit docstring components from
    a library or from another function, with multi-language support.
    """

    def __init__(
        self,
        template: Optional[Union[str, DocstringTemplate]] = None,
        param_library: Optional[ParameterLibrary] = None,
        inherit_from: Optional[Callable[..., Any]] = None,
        language: str = "en",
        **params: str,
    ) -> None:
        self.template = template
        self.param_library = param_library
        self.inherit_from = inherit_from
        self.language = language
        self.params = params

    def __call__(self, func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper.__doc__ = self._build_docstring(func)
        return cast(F, wrapper)

    def _build_docstring(self, func: F) -> Optional[str]:
        """Build the final docstring for the function."""
        if __debug__ is False:
            return None

        inherited_params = {}
        if self.inherit_from is not None and self.param_library is not None:
            inherited_params = self.param_library.inherit(self.inherit_from)

        all_params = {**inherited_params, **self.params}

        if isinstance(self.template, DocstringTemplate):
            return self.template.render(all_params, language=self.language)
        elif isinstance(self.template, str):
            template = DocstringTemplate(self.template, all_params, self.language)
            return template.render()
        elif func.__doc__:
            template = DocstringTemplate(func.__doc__, all_params, self.language)
            return template.render()

        return None


def docstring(
    template: Optional[Union[str, DocstringTemplate]] = None,
    *,
    library: Optional[ParameterLibrary] = None,
    inherit: Optional[Callable[..., Any]] = None,
    language: str = "en",
    **params: str,
) -> Callable[[F], F]:
    """
    Decorate a function with a docstring template.

    Parameters
    ----------
    template : str or DocstringTemplate, optional
        Template string or object to use.
    library : ParameterLibrary, optional
        Parameter library for multi-language lookups.
    inherit : callable, optional
        Function from which to inherit parameters.
    language : str, optional
        Language for docstring rendering (e.g., 'en', 'zh_CN').
    **params : str
        Additional parameter mappings.

    Returns
    -------
    callable
        Decorated function.
    """
    decorator = DocstringDecorator(
        template=template,
        param_library=library,
        inherit_from=inherit,
        language=language,
        **params,
    )
    return decorator  # type: ignore[return-value]


T = TypeVar("T")


class DocstringInheritor(type):
    """Metaclass that enables docstring inheritance in subclasses."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: Dict[str, Any],
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace)

        for attr_name, attr_value in namespace.items():
            if callable(attr_value) and not attr_value.__doc__:
                for base in bases:
                    base_method = getattr(base, attr_name, None)
                    if base_method and base_method.__doc__:
                        attr_value.__doc__ = base_method.__doc__
                        break

        return cls


def validate_public_api(module: Any) -> Tuple[bool, str]:
    """
    Validate all public API docstrings for completeness and consistency.

    This is a concrete implementation of the docstring validation feature.

    Parameters
    ----------
    module : module
        Module to validate (typically import seaborn).

    Returns
    -------
    tuple[bool, str]
        Success flag and validation report.
    """
    validator = DocstringValidator(
        require_parameters=True,
        require_returns=True,
        require_examples=False,
        require_notes=False,
    )
    results = validator.validate_module(module, public_only=True)
    report = validator.get_validation_report(results)
    all_passed = all(r.passed for r in results.values())
    return all_passed, report


_language_support = LanguageSupport()
_param_library = ParameterLibrary()
_docstring_validator = DocstringValidator()

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
    size="""
size : vector or key in ``data``
    Semantic variable that is mapped to determine the size of plot elements.
    """,
    sizes="""
sizes : tuple of floats or dict
    Range of sizes to use for level mapping, or mapping of levels to sizes.
    """,
    size_order="""
size_order : vector of strings
    Specify the order of processing for categorical levels of the ``size`` semantic.
    """,
    size_norm="""
size_norm : tuple
    Either a pair of values that set the normalization range in data units for size.
    """,
    style="""
style : vector or key in ``data``
    Semantic variable that is mapped to determine the style of plot elements.
    """,
    dashes="""
dashes : boolean, list, or dict
    Object determining how to draw the dashes for different levels.
    """,
    markers="""
markers : boolean, list, or dict
    Object determining how to draw the markers for different levels.
    """,
    style_order="""
style_order : vector of strings
    Specify the order of processing for categorical levels of the ``style`` semantic.
    """,
    units="""
units : vector or key in ``data``
    Grouping variable identifying sampling units.
    """,
    seed="""
seed : int, numpy.random.Generator, or numpy.random.RandomState
    Seed or random number generator for reproducible jittering.
    """,
    orient="""
orient : "v" | "h"
    Orientation of the plot (vertical or horizontal).
    """,
    ci="""
ci : int or "sd" or None
    Size of the confidence interval when drawing error bars.
    """,
    n_boot="""
n_boot : int
    Number of bootstrap samples to use when computing confidence intervals.
    """,
    kwargs="""
kwargs : key, value mappings
    Other keyword arguments are passed through to the underlying plotting function.
    """,
)

_param_library.add_parameters(_core_params, "core")


class _DocstringComponents:
    """
    Backward compatibility class for existing seaborn code.
    Wraps parameter library for legacy docstring component access.
    Supports nested attribute access like params.core.hue_order.
    """

    def __init__(self, components: Dict[str, Any]) -> None:
        self._components = components

    @classmethod
    def from_nested_components(cls, **kwargs: Any) -> "_DocstringComponents":
        """Create from nested component structure."""
        nested = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                nested[key] = cls(value)
            else:
                nested[key] = value
        return cls(nested)

    @classmethod
    def from_function_params(cls, func: Callable[..., Any]) -> "_DocstringComponents":
        """Extract docstring parameters from a function."""
        doc = pydoc.getdoc(func)
        if not doc:
            return cls({})
        parsed = NumpyDocString(doc)
        params = {}
        for p in parsed["Parameters"]:
            name = p.name
            type_str = p.type or ""
            desc = "\n    ".join(p.desc)
            params[name] = f"{name} : {type_str}\n    {desc}"
        return cls(params)

    def __getitem__(self, key: str) -> Any:
        return self._components[key]

    def __getattr__(self, name: str) -> Any:
        if name in self._components:
            return self._components[name]
        return ""

    def __call__(self, **params: Any) -> Any:
        return self._components.get("params", {})


DocstringComponents = _DocstringComponents

_seealso_docs = {
    "scatterplot": """
scatterplot : Plot data and a linear regression model fit.
    """,
    "lineplot": """
lineplot : Show point estimates and confidence intervals using scatter plot glyphs.
    """,
    "pointplot": """
pointplot : Show point estimates and confidence intervals using lines.
    """,
    "stripplot": """
stripplot : Draw a scatterplot where one variable is categorical.
    """,
    "swarmplot": """
swarmplot : Draw a categorical scatterplot with non-overlapping points.
    """,
    "displot": """
displot : Figure-level distribution plots interface.
    """,
    "histplot": """
histplot : Plot univariate or bivariate histograms.
    """,
    "kdeplot": """
kdeplot : Plot univariate or bivariate kernel density estimates.
    """,
    "ecdfplot": """
ecdfplot : Plot empirical cumulative distribution functions.
    """,
    "rugplot": """
rugplot : Plot marginal distributions by drawing ticks.
    """,
    "violinplot": """
violinplot : Draw a combination of boxplot and kernel density estimate.
    """,
    "jointplot": """
jointplot : Draw a plot of two variables with bivariate and univariate graphs.
    """,
    "pairgrid": """
PairGrid : Subplot grid for plotting pairwise relationships.
    """,
    "pairplot": """
pairplot : Plot pairwise relationships in a dataset.
    """,
    "jointgrid": """
JointGrid : Grid for drawing a bivariate plot with marginal univariate plots.
    """,
}

_core_docs = {
    "params": _param_library._parameters["core"],
    "returns": DocstringComponents({
        "ax": """
ax : :class:`matplotlib.axes.Axes`
    Returns the Axes object with the plot drawn onto it.
        """,
        "plotter": """
plotter : object
    Returns the seaborn Plotter object.
        """,
    }),
    "seealso": DocstringComponents(_seealso_docs),
}



__all__ = [
    "DocstringTemplate",
    "ParameterLibrary",
    "DocstringDecorator",
    "DocstringInheritor",
    "DocstringValidator",
    "DocstringValidationResult",
    "LanguageSupport",
    "docstring",
    "validate_public_api",
    "DocstringComponents",
]

