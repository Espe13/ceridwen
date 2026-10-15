"""tests/agn2: a theta key the CSP does not recognise is an ERROR in this suite (a silently
ignored key once made a dust test a no-op here)."""
import pytest


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "/agn2/" in str(item.fspath):
            item.add_marker(pytest.mark.filterwarnings(
                "error:CSPBasis received unrecognized theta key"))
