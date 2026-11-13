from mvm.ladder import buckets, expected, parse_ladder


def test_ladder_buckets_and_expected():
    b, issues = buckets(parse_ladder("85=90,90=70,95=40"), floor=70, ceil=100)
    assert not issues
    assert abs(b.p.sum() - 1) < 1e-9
    assert [round(x, 2) for x in b.p] == [0.10, 0.20, 0.30, 0.40]
    assert 85 < expected(b) < 95


def test_ladder_flags_inconsistent_prices():
    _, issues = buckets(parse_ladder("85=70,90=75"))
    assert len(issues) == 1


def test_integer_labels():
    b, _ = buckets(parse_ladder("96=77,97=59,98=31"), floor=88, ceil=99, integer=True)
    assert list(b.label) == ["95 or lower", "96", "97", "98+"]
    assert [round(x, 2) for x in b.p] == [0.23, 0.18, 0.28, 0.31]
