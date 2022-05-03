
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"





from ._sim_widget import SimAnalysis, reshape
