
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Andrea Bassi and Mark Neil"
__email__ = "andreabassi@polimi.it"


from ._sim_widget import SimAnalysis, reshape
