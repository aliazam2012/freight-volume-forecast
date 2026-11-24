from freight_agents.volume_forecast.grounding import numbers_in, ungrounded


def test_numbers_are_normalized() -> None:
    assert numbers_in("1,855 loads, 11.40% on 2026-11-05, MASE 0.56") == {
        "1855",
        "11.4",
        "2026",
        "11",
        "5",
        "0.56",
    }


def test_ungrounded_finds_numbers_no_source_gave() -> None:
    sources = ['{"forecast": 1855, "wape_pct": 11.4}', "How busy is CHI-DAL?"]
    assert ungrounded("About 1,855 loads, error 11.4%.", sources) == set()
    assert ungrounded("About 2,000 loads, error 11.4%.", sources) == {"2000"}
