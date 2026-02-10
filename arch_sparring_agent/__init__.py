"""Architecture Review Sparring Partner - multi-agent architecture review system."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arch-sparring-agent")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
