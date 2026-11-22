import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from api import create_app
from src.sql_agent.postprocess import (
    UnknownConverter,
    apply_converters,
    available,
    build_all,
    build_converters,
    validate_options,
)
from tests.conftest import FakeLLM

CFG = {
    "default_enabled": [],
    "currency": {"base_currency": "SGD", "conversion_rates": {"SGD": 1.0, "USD": 1.35, "EUR": 1.45}},
    "units": {"targets": {"length": "km", "mass": "kg", "volume": "l", "temperature": "C"}},
    "durations": {"target": "minutes"},
    "numbers": {},
    "datetime": {"dayfirst": True},
}


def one(name, options=None):
    return build_converters(CFG, [name], options)[0]


def test_every_converter_publishes_its_options():
    specs = {a["name"]: a["options"] for a in available(CFG)}
    assert [o["key"] for o in specs["units"]] == ["length", "mass", "volume", "temperature"]
    assert specs["durations"][0]["choices"] == ["seconds", "minutes", "hours", "days"]
    assert specs["datetime"][0]["choices"] == ["day first", "month first"]
    assert specs["numbers"] == []
    for opts in specs.values():
        for o in opts:
            assert o["default"] in o["choices"]


def test_units_target_option_changes_the_result():
    assert one("units").convert_value("5 km") == 5.0
    assert one("units", {"units": {"length": "mi"}}).convert_value("5 km") == pytest.approx(3.106856)
    assert one("units", {"units": {"temperature": "F"}}).convert_value("100 degC") == pytest.approx(212.0)


def test_durations_option():
    assert one("durations", {"durations": {"target": "hours"}}).convert_value("90 min") == 1.5


def test_datetime_order_option():
    assert one("datetime").convert_value("05/10/1973") == "1973-10-05"
    assert one("datetime", {"datetime": {"order": "month first"}}).convert_value("05/10/1973") == "1973-05-10"


def test_currency_rebase():
    usd = one("currency", {"currency": {"base_currency": "USD"}})
    assert usd.base == "USD"
    assert usd.convert_value("USD$ 100.00") == pytest.approx(100.0)
    assert usd.convert_value("SGD$ 135.00") == pytest.approx(100.0)  # 135 SGD = 100 USD
    assert usd.describe() == "to USD"
    assert CFG["currency"]["base_currency"] == "SGD"  # config not mutated


def test_options_do_not_leak_between_requests():
    one("units", {"units": {"length": "mi"}})
    assert one("units").convert_value("5 km") == 5.0


def test_detection_cache_key_depends_on_date_order():
    assert one("datetime").cache_key != one("datetime", {"datetime": {"order": "month first"}}).cache_key
    assert one("units").cache_key == "units"


@pytest.mark.parametrize(
    "bad",
    [{"nope": {"x": "y"}}, {"units": {"weight": "kg"}}, {"units": {"length": "parsecs"}},
     {"datetime": {"order": "sideways"}}, {"numbers": {"x": "y"}}],
)
def test_invalid_options_are_rejected(bad):
    with pytest.raises(UnknownConverter):
        validate_options(CFG, bad)


def test_valid_options_pass():
    ok = {"units": {"length": "mi"}, "currency": {"base_currency": "USD"}}
    assert validate_options(CFG, ok) == ok
    assert validate_options(CFG, None) == {}


def test_rows_use_the_chosen_option():
    df = pd.DataFrame({"d": ["5 km", "10 km"]})
    out, applied = apply_converters(df, build_converters(CFG, ["units"], {"units": {"length": "m"}}))
    assert out["d"].tolist() == [5000.0, 10000.0]
    assert applied[0].detail == "to m (length)"


# -- API ------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, shop_db):
    def make(responses):
        cfg = {
            "default_db": str(shop_db),
            "llm": {"model_name": "x", "model_provider": "groq"},
            "retrieval": {"sample_limit": 0},
            "guardrails": {"max_rows": 100},
            "agent_loop": {"max_attempts": 3},
            "uploads": {"dir": str(tmp_path / "up"), "max_mb": 1},
            "checkpoint_db": str(tmp_path / "ck.db"),
            "postprocess": CFG,
        }
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return TestClient(create_app(str(p), llm=FakeLLM(responses)))

    return make


def test_api_rejects_bad_option_with_400(client):
    with client(["x"]) as c:
        r = c.post("/chat", json={"message": "q", "conversion_options": {"units": {"length": "parsecs"}}})
    assert r.status_code == 400 and "parsecs" in r.json()["detail"]


def test_api_applies_option_end_to_end(tmp_path, client):
    import sqlite3

    db = tmp_path / "d.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (id INTEGER, dist TEXT)")
    con.executemany("INSERT INTO t VALUES (?,?)", [(1, "5 km"), (2, "10 km")])
    con.commit(); con.close()
    up = tmp_path / "x"
    with client(["SELECT id, dist FROM t ORDER BY id", "ok"]) as c:
        db_id = c.post("/upload-db", files={"file": ("d.db", db.read_bytes())}).json()["db_id"]
        r = c.post("/chat", json={
            "message": "q", "db_id": db_id, "conversions": ["units"],
            "conversion_options": {"units": {"length": "m"}},
        }).json()
    assert [row["dist"] for row in r["rows"]] == [5000.0, 10000.0]
    assert r["conversions_applied"][0]["detail"] == "to m (length)"


def test_conversions_endpoint_includes_options(client):
    with client(["x"]) as c:
        body = c.get("/conversions").json()
    by = {a["name"]: a for a in body["available"]}
    assert by["units"]["options"][0]["key"] == "length"
