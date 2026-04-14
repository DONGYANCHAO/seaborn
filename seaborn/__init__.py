# Capture the original matplotlib rcParams before importing anything
# that might change them internally
import matplotlib as mpl
_orig_rc_params = mpl.rcParams.copy()

# Define the seaborn version
__version__ = "0.14.0.dev0"

# Import seaborn objects in logical order to minimize import time
# Grouping by module type to optimize cache locality

# Core utilities and configuration
from .rcmod import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
from .palettes import *  # noqa: F401,F403

# Plotting module groups
# Relational plots
from .relational import *  # noqa: F401,F403

# Regression plots
from .regression import *  # noqa: F401,F403

# Categorical plots
from .categorical import *  # noqa: F401,F403

# Distribution plots
from .distributions import *  # noqa: F401,F403

# Matrix plots
from .matrix import *  # noqa: F401,F403

# Miscellaneous plots
from .miscplot import *  # noqa: F401,F403

# Axis grid utilities
from .axisgrid import *  # noqa: F401,F403

# Widgets
from .widgets import *  # noqa: F401,F403

# Color utilities
from .colors import xkcd_rgb, crayons  # noqa: F401

# Colormap access
from . import cm  # noqa: F401
