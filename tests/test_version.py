import rigma


def test_version_present():
    assert rigma.__version__.startswith("0.")
