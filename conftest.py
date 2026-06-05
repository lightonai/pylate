import pytest

collect_ignore = []

try:
    import voyager  # noqa: F401
except ImportError:
    collect_ignore.append("pylate/indexes/voyager.py")

try:
    import scann  # noqa: F401
except ImportError:
    collect_ignore.append("pylate/retrieve/xtr.py")


# Some doctests (e.g. pylate/evaluation/beir.py) download datasets from an
# external host that is regularly unreachable in CI. A network outage there is
# an infra problem, not a code regression, so we downgrade doctest failures
# caused by connection errors from FAIL to SKIP. Real doctest mismatches (wrong
# output) still fail; only network/connection errors are skipped.
_NETWORK_ERROR_MARKERS = (
    "Connection refused",
    "Connection reset",
    "Connection aborted",
    "Connection timed out",
    "Max retries exceeded",
    "Failed to establish a new connection",
    "Temporary failure in name resolution",
    "Name or service not known",
    "NewConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "timed out",
)


def _is_network_error(exc: BaseException | None) -> bool:
    """Walk the exception chain and report whether it is a network failure.

    Doctests wrap the original error in ``doctest.UnexpectedException`` whose
    ``exc_info`` holds the real exception but is not linked via ``__cause__`` /
    ``__context__``, so we unwrap that explicitly while walking the chain.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        if isinstance(exc, OSError) and exc.errno in (101, 110, 111, -2, -3):
            return True
        text = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
        if any(marker in text for marker in _NETWORK_ERROR_MARKERS):
            return True
        wrapped = getattr(exc, "exc_info", None)
        if isinstance(wrapped, tuple) and len(wrapped) == 3:
            if _is_network_error(wrapped[1]):
                return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "call"
        and report.failed
        and isinstance(item, pytest.DoctestItem)
        and call.excinfo is not None
        and _is_network_error(call.excinfo.value)
    ):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.fspath),
            0,
            f"Skipped: external host unreachable ({type(call.excinfo.value).__name__})",
        )
