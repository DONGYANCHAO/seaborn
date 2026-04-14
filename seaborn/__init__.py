"""
Seaborn: statistical data visualization
=======================================

Seaborn is a Python data visualization library based on matplotlib.
It provides a high-level interface for drawing attractive and informative
statistical graphics.

For more information, visit https://seaborn.pydata.org
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

from .rcmod import (
    set_theme,
    set,
    reset_defaults,
    reset_orig,
    axes_style,
    set_style,
    plotting_context,
    set_context,
    set_palette,
)
from .utils import (
    get_dataset_names,
    get_data_home,
    load_dataset,
    despine,
    move_legend,
)
from .palettes import (
    color_palette,
    light_palette,
    dark_palette,
    diverging_palette,
    blend_palette,
    xkcd_palette,
    crayon_palette,
    cubehelix_palette,
)
from .relational import (
    relplot,
    scatterplot,
    lineplot,
)
from .regression import (
    lmplot,
    regplot,
    residplot,
)
from .categorical import (
    catplot,
    stripplot,
    swarmplot,
    boxplot,
    violinplot,
    boxenplot,
    pointplot,
    barplot,
    countplot,
)
from .distributions import (
    displot,
    histplot,
    kdeplot,
    ecdfplot,
    rugplot,
)
from .matrix import (
    heatmap,
    clustermap,
)
from .miscplot import (
    palplot,
)
from .axisgrid import (
    FacetGrid,
    PairGrid,
    JointGrid,
    pairplot,
    jointplot,
)
from .colors import xkcd_rgb, crayons
from . import cm

import matplotlib as mpl

_orig_rc_params = mpl.rcParams.copy()

__version__ = "0.14.0.dev0"

__all__ = [
    "__version__",
    "set_theme",
    "set",
    "reset_defaults",
    "reset_orig",
    "axes_style",
    "set_style",
    "plotting_context",
    "set_context",
    "set_palette",
    "color_palette",
    "light_palette",
    "dark_palette",
    "diverging_palette",
    "blend_palette",
    "xkcd_palette",
    "crayon_palette",
    "cubehelix_palette",
    "get_dataset_names",
    "get_data_home",
    "load_dataset",
    "despine",
    "move_legend",
    "relplot",
    "scatterplot",
    "lineplot",
    "lmplot",
    "regplot",
    "residplot",
    "catplot",
    "stripplot",
    "swarmplot",
    "boxplot",
    "violinplot",
    "boxenplot",
    "pointplot",
    "barplot",
    "countplot",
    "displot",
    "histplot",
    "kdeplot",
    "ecdfplot",
    "rugplot",
    "heatmap",
    "clustermap",
    "palplot",
    "FacetGrid",
    "PairGrid",
    "JointGrid",
    "pairplot",
    "jointplot",
    "xkcd_rgb",
    "crayons",
    "cm",
]
