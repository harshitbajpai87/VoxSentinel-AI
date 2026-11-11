"""Load detector artifacts saved by a different dependency build.

A joblib artifact is a pickle, and a pickle stores *module paths* for every
class it references. Those paths change between library releases:

* numpy >= 2 writes arrays as ``numpy._core.multiarray``; numpy < 2 wrote
  ``numpy.core.multiarray``.
* scikit-learn has moved private classes between releases.
* older scikit-learn pickles referenced ``sklearn.externals.joblib``.

Deploy with a numpy/sklearn build different from the one that trained the
artifact and ``joblib.load`` raises ``ImportError``/``ModuleNotFoundError``.
Historically that surfaced as the opaque ``could not load artifact
(ImportError)`` and the API silently dropped to "model unavailable".

This module does three things instead:

1. installs the module aliases for the known renames,
2. retries the load once with those aliases in place,
3. if it still fails, raises :class:`ArtifactLoadError` carrying the missing
   module name and the versions actually installed, so the reason is fixable
   instead of invisible.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import joblib

logger = logging.getLogger("voxsentinel.artifact_compat")

# numpy 2 renamed the compiled core; the two families are interchangeable for
# unpickling purposes, so expose whichever one the artifact asks for.
_NUMPY_CORE_SUBMODULES = (
    "multiarray",
    "numeric",
    "umath",
    "numerictypes",
    "fromnumeric",
    "shape_base",
    "_multiarray_umath",
    "_exceptions",
    "overrides",
)

# Legacy module paths that older scikit-learn pickles reference.
_LEGACY_ALIASES = {
    "sklearn.externals.joblib": "joblib",
}


class ArtifactLoadError(RuntimeError):
    """The artifact exists but cannot be unpickled in this runtime."""

    def __init__(self, detail: str, missing_module: str | None = None) -> None:
        self.detail = detail
        self.missing_module = missing_module
        super().__init__(detail)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.detail


def dependency_fingerprint() -> dict[str, str]:
    """Versions that decide whether a pickled artifact can be read here."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "sklearn", "joblib"):
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - a missing optional dep is reported, not raised
            versions[name] = "missing"
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _missing_module_name(exc: BaseException) -> str | None:
    """Pull the module name out of an ImportError, if it names one."""
    name = getattr(exc, "name", None)
    if name:
        return str(name)
    message = str(exc)
    for marker in ("No module named ", "cannot import name "):
        if marker in message:
            tail = message.split(marker, 1)[1]
            candidate = tail.split("'")[1] if "'" in tail else tail.split()[0]
            return candidate.replace(" ", "")
    return None


def _alias_numpy_core() -> list[str]:
    """Make ``numpy._core`` and ``numpy.core`` resolve to whichever exists."""
    installed: list[str] = []
    target = None
    for candidate in ("numpy._core", "numpy.core"):
        try:
            module = importlib.import_module(candidate)
        except Exception:  # noqa: BLE001 - missing is the point of the probe
            continue
        target = module
        installed.append(candidate)
        break
    if target is None:
        return []

    families = {"numpy._core": target, "numpy.core": target}
    for name, module in families.items():
        if name not in sys.modules:
            sys.modules[name] = module
    for submodule in _NUMPY_CORE_SUBMODULES:
        try:
            real = importlib.import_module(f"{installed[0]}.{submodule}")
        except Exception:  # noqa: BLE001 - not every submodule exists in every release
            continue
        for prefix in ("numpy._core", "numpy.core"):
            alias = f"{prefix}.{submodule}"
            sys.modules.setdefault(alias, real)
    return installed


# ---------------------------------------------------------------------------
# scikit-learn >= 1.6 pickles CalibratedClassifierCV base estimators inside a
# FrozenEstimator, which lives in the module ``sklearn.frozen._frozen``. Builds
# before 1.6 have no such module, so loading raises ModuleNotFoundError:
# "No module named 'sklearn.frozen'" - the exact "ImportError" this project
# used to report with no explanation. The shim below is a faithful delegating
# stand-in: it exposes the wrapped estimator's attributes and methods and makes
# fit() a no-op, which is all the unpickled calibrated model ever calls.
# ---------------------------------------------------------------------------


class _FrozenEstimatorShim:
    """Stand-in for scikit-learn's FrozenEstimator (sklearn.frozen)."""

    def __init__(self, estimator=None, freeze=None) -> None:
        self.estimator = estimator
        self.freeze = True if freeze is None else freeze

    def __getattr__(self, name: str):
        # Only called when normal attribute lookup fails; never recurse on
        # `estimator` itself, which is absent while the pickle is reconstructed.
        if name in ("fit_predict", "fit_transform"):
            raise AttributeError(f"{name} is not available for frozen estimators.")
        estimator = self.__dict__.get("estimator")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)

    def __sklearn_clone__(self):
        return self

    def fit(self, X=None, y=None, *args, **kwargs):
        return self


def _install_sklearn_frozen_shim() -> list[str]:
    """Provide sklearn.frozen._frozen when the installed scikit-learn lacks it."""
    import types

    try:
        import sklearn
    except Exception:  # noqa: BLE001 - scikit-learn is a hard dependency; be safe anyway
        return []
    if getattr(sklearn, "frozen", None) is not None:
        return []
    if "sklearn.frozen._frozen" in sys.modules:
        return []

    frozen_pkg = types.ModuleType("sklearn.frozen")
    frozen_module = types.ModuleType("sklearn.frozen._frozen")
    frozen_module.FrozenEstimator = _FrozenEstimatorShim
    frozen_module._FrozenEstimator = _FrozenEstimatorShim
    frozen_pkg.FrozenEstimator = _FrozenEstimatorShim
    frozen_pkg._frozen = frozen_module
    sys.modules["sklearn.frozen"] = frozen_pkg
    sys.modules["sklearn.frozen._frozen"] = frozen_module
    sklearn.frozen = frozen_pkg
    return ["sklearn.frozen._frozen"]


def install_import_aliases() -> dict[str, list[str]]:
    """Install compatibility aliases. Returns what was applied (for logging)."""
    applied: dict[str, list[str]] = {}
    numpy_families = _alias_numpy_core()
    if numpy_families:
        applied["numpy_core"] = numpy_families
    shimmed = _install_sklearn_frozen_shim()
    if shimmed:
        applied["sklearn_frozen_shim"] = shimmed
        logger.warning(
            "artifact was saved with scikit-learn >= 1.6 but this runtime has %s; "
            "installing the FrozenEstimator compatibility shim so it can load. "
            "Pin scikit-learn>=1.6 (requirements-core.txt) or re-save the artifact "
            "with scripts/repack_detector_artifact.py to remove the mismatch.",
            dependency_fingerprint()["sklearn"],
        )
    for legacy, current in _LEGACY_ALIASES.items():
        if legacy in sys.modules:
            continue
        try:
            sys.modules[legacy] = importlib.import_module(current)
        except Exception:  # noqa: BLE001 - alias is best effort
            continue
        applied.setdefault("legacy", []).append(legacy)
    return applied


def load_artifact(path: str | Path):
    """joblib-load ``path``, retrying once through the compatibility aliases.

    Raises :class:`ArtifactLoadError` with the real reason when the pickle
    cannot be reconstructed here, and the original exception otherwise.
    """
    artifact_path = Path(path)
    try:
        return joblib.load(artifact_path)
    except (ImportError, ModuleNotFoundError) as exc:
        missing = _missing_module_name(exc)
        applied = install_import_aliases()
        logger.warning(
            "artifact %s needs a module this runtime lacks (%s); retrying with "
            "compatibility aliases %s",
            artifact_path,
            missing or "unknown",
            applied or "none available",
        )
        try:
            return joblib.load(artifact_path)
        except (ImportError, ModuleNotFoundError):
            pass
        fingerprint = dependency_fingerprint()
        raise ArtifactLoadError(
            "could not load artifact because the pickle references "
            f"{missing or 'a module'} which is not importable in this runtime "
            f"(installed: numpy {fingerprint['numpy']}, scikit-learn "
            f"{fingerprint['sklearn']}, joblib {fingerprint['joblib']}). "
            "Install the versions the artifact was trained with, or re-save it "
            "with scripts/repack_detector_artifact.py in this environment.",
            missing_module=missing,
        ) from exc


def load_artifact_or_explain(path: str | Path):
    """load_artifact, with targeted guidance for the numpy-version failure.

    numpy >= 2 pickles its random BitGenerator by class identity, so a numpy 1.x
    runtime raises "ValueError: ... is not a known BitGenerator module" - a
    confusing message for what is really "install numpy >= 2".
    """
    try:
        return load_artifact(path)
    except ArtifactLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - translate only the known family
        text = str(exc)
        if "BitGenerator" in text or "numpy._core" in text:
            fingerprint = dependency_fingerprint()
            raise ArtifactLoadError(
                "could not load artifact because it was saved with numpy >= 2 and "
                f"this runtime has numpy {fingerprint['numpy']}. Install "
                "numpy>=2 (requirements-core.txt pins numpy==2.4.6) or re-save the "
                "artifact with scripts/repack_detector_artifact.py in this "
                "environment.",
                missing_module="numpy",
            ) from exc
        raise


def describe_artifact(path: str | Path) -> dict:
    """Non-destructive readiness report for one artifact path."""
    artifact_path = Path(path)
    report: dict = {
        "path": str(artifact_path),
        "exists": artifact_path.is_file(),
        "loadable": False,
        "error": None,
        "required_module": None,
        "dependencies": dependency_fingerprint(),
    }
    if not report["exists"]:
        return report
    try:
        load_artifact_or_explain(artifact_path)
    except ArtifactLoadError as exc:
        report["error"] = exc.detail
        report["required_module"] = exc.missing_module
    except Exception as exc:  # noqa: BLE001 - any failure means "not loadable"
        report["error"] = f"{type(exc).__name__}: {exc}"
    else:
        report["loadable"] = True
    return report
