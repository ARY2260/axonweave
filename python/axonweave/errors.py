"""Stable error codes and exception hierarchy (AXW000-AXW101).

Every error raised by AxonWeave derives from :class:`AxonWeaveError`, carries
a stable ``AXWnnn`` code, and renders a message following the house pattern:

    CODE: what failed; expected X, got Y. Next action.

Codes never change meaning between releases; message wording may improve.

Usage errors (wrong dimensions, unknown option names, missing wiring) raise
:class:`ApiUsageError` with the ``AXW010`` prefix. It subclasses ``ValueError``
so existing ``except ValueError`` code keeps working while new code can catch
the specific class.
"""
from __future__ import annotations


class AxonWeaveError(RuntimeError):
    """Base class for all AxonWeave errors (AXW000)."""

    code = "AXW000"


class AxonWeaveWarning(UserWarning):
    """Base class for all AxonWeave warnings."""

    code = "AXW000"


class SubstrateNotInstalledError(AxonWeaveError):
    """The requested substrate is absent from the local cache (AXW001)."""

    code = "AXW001"


class DatasetIntegrityError(AxonWeaveError):
    """Checksum, manifest identity, or fingerprint mismatch (AXW002)."""

    code = "AXW002"


class SchemaError(AxonWeaveError):
    """A source data file does not match the expected schema (AXW003)."""

    code = "AXW003"


class UnsupportedDeviceError(AxonWeaveError):
    """The host framework rejected the requested device (AXW004)."""

    code = "AXW004"


class BiologicalAssumptionError(AxonWeaveError):
    """A requested model choice lacks the biological data it needs (AXW005)."""

    code = "AXW005"


class ReceptorModelError(BiologicalAssumptionError):
    """Receptor model configuration error (AXW009)."""

    code = "AXW009"


class BackendUnavailableError(AxonWeaveError):
    """An optional framework (torch, TensorFlow) is not installed (AXW006)."""

    code = "AXW006"


class ConfigurationWarning(AxonWeaveWarning):
    """A non-fatal configuration issue was detected (AXW007)."""

    code = "AXW007"


class AnnotationBuildWarning(AxonWeaveWarning):
    """Selection table construction failed during install (AXW008)."""

    code = "AXW008"


class ApiUsageError(AxonWeaveError, ValueError):
    """API misuse: bad dimensions, unknown options, missing wiring (AXW010).

    Subclasses :class:`ValueError` for backward compatibility: existing
    ``except ValueError`` handlers continue to work, while new code may catch
    :class:`ApiUsageError` specifically.
    """

    code = "AXW010"


def _require(condition: bool, message: str) -> None:
    """Raise :class:`ApiUsageError` with the ``AXW010`` prefix unless satisfied.

    House message style: ``AXW010: <what failed>; expected X, got Y``.
    The prefix is added automatically when the message does not start with it.
    """
    if not condition:
        raise ApiUsageError(message if message.startswith("AXW010:") else f"AXW010: {message}")
