import pytest

from pytaa.tools.data import validate_tickers


@pytest.mark.parametrize("tickers, raise_issue, expected", [
    ([], False, False),
    (["GOOG", None], False, False),
    (["MSFT", "FB"], False, True)
])
def test_validate_tickers(tickers, raise_issue, expected):
    """Test validate tickers function."""
    assert validate_tickers(tickers, raise_issue) == expected
