import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from api import create_app
from src.sql_agent.postprocess import (
    UnknownConverter,
    apply_converters,
    available,
    build_converters,
)
from src.sql_agent.postprocess.currency import CurrencyConverter
from tests.conftest import FakeLLM

RATES = {"SGD": 1.0, "USD": 1.35}
CURRENCY_CFG = {"base_currency": "SGD", "conversion_rates": RATES}
PP_CFG = {"default_enabled": [], "currency": CURRENCY_CFG}


def convert(df):
    return apply_converters(df, [CurrencyConverter(CURRENCY_CFG)])


def test_mixed_currencies_converted():
    df = pd.DataFrame({"price": ["SGD$ 100.00", "USD$ 100.00", None]})
    out, applied = convert(df)
    assert out["price"].tolist()[:2] == [100.0, 135.0]
    assert pd.isna(out["price"].iloc[2])
    assert df["price"].iloc[1] == "USD$ 100.00"  # input untouched
    assert [(a.column, a.converter, a.detail) for a in applied] == [
        ("price", "currency", "to SGD")
    ]


def test_non_currency_columns_untouched():
    df = pd.DataFrame({"name": ["Ann", "Bob"], "n": [1, 2]})
    out, applied = convert(df)
    assert out.equals(df) and applied == []


def test_base_currency_only_is_unchanged():
    df = pd.DataFrame({"price": ["SGD$ 10.00", "SGD$ 20.00"]})
    out, applied = convert(df)
    assert out.equals(df) and applied == []


def test_detection_is_per_column():
    df = pd.DataFrame(
        {"usd_mix": ["SGD$ 1.00", "USD$ 1.00"], "sgd_only": ["SGD$ 5.00", "SGD$ 6.00"]}
    )
    out, applied = convert(df)
    assert [a.column for a in applied] == ["usd_mix"]
    assert out["sgd_only"].tolist() == ["SGD$ 5.00", "SGD$ 6.00"]


def test_registry_builds_only_enabled_and_configured():
    assert build_converters(PP_CFG, []) == []
    assert [c.name for c in build_converters(PP_CFG, ["currency"])] == ["currency"]
    assert build_converters({"default_enabled": []}, ["currency"]) == []  # unconfigured


def test_registry_rejects_unknown_name():
    with pytest.raises(UnknownConverter):
        build_converters(PP_CFG, ["currencyy"])


def test_available_lists_configured_converters():
    assert available(PP_CFG) == [
        {"name": "currency", "label": "Currency values (convert to SGD)"}
    ]
    assert available({}) == []


@pytest.fixture
def make_client(tmp_path):
    def _make(db, responses, default_enabled=None):
        cfg = {
            "default_db": str(db),
            "llm": {"model_name": "x", "model_provider": "groq"},
            "retrieval": {"sample_limit": 0},
            "guardrails": {"max_rows": 100},
            "agent_loop": {"max_attempts": 3},
            "uploads": {"dir": str(tmp_path / "up"), "max_mb": 1},
            "checkpoint_db": str(tmp_path / "ck.db"),
            "postprocess": {
                "default_enabled": default_enabled or [],
                "currency": CURRENCY_CFG,
            },
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.safe_dump(cfg))
        llm = FakeLLM(responses)
        return TestClient(create_app(str(p), llm=llm)), llm

    return _make


SQL = "SELECT id, price FROM bookings WHERE price IS NOT NULL ORDER BY id"


def test_toggle_on_converts_rows_and_summary_sees_converted_values(
    make_client, priced_db
):
    c, llm = make_client(priced_db, [SQL, "Booking 2 costs 135 SGD."])
    with c:
        r = c.post("/chat", json={"message": "prices", "conversions": ["currency"]}).json()
    assert r["conversions_applied"] == [
        {"column": "price", "converter": "currency", "detail": "to SGD"}
    ]
    assert [row["price"] for row in r["rows"]] == [100.0, 135.0]
    summary_prompt = llm.prompts[-1]
    assert "135" in summary_prompt and "USD$" not in summary_prompt
    assert "converted to SGD" in summary_prompt


def test_toggle_off_leaves_raw_strings(make_client, priced_db):
    c, llm = make_client(priced_db, [SQL, "ok"])
    with c:
        r = c.post("/chat", json={"message": "prices", "conversions": []}).json()
    assert r["conversions_applied"] == []
    assert [row["price"] for row in r["rows"]] == ["SGD$ 100.00", "USD$ 100.00"]
    assert "converted" not in llm.prompts[-1]


def test_omitted_field_uses_config_default(make_client, priced_db):
    c, _ = make_client(priced_db, [SQL, "ok"], default_enabled=["currency"])
    with c:
        r = c.post("/chat", json={"message": "prices"}).json()
    assert len(r["conversions_applied"]) == 1


def test_toggle_on_is_noop_for_db_without_currency(make_client, shop_db):
    c, llm = make_client(
        shop_db, ["SELECT name, city FROM customers ORDER BY id", "Ann, Bob."]
    )
    with c:
        r = c.post("/chat", json={"message": "names", "conversions": ["currency"]}).json()
    assert r["conversions_applied"] == [] and not r["failed"]
    assert r["rows"][0] == {"name": "Ann", "city": "Paris"}
    assert "converted" not in llm.prompts[-1]


def test_unknown_conversion_is_400(make_client, shop_db):
    c, _ = make_client(shop_db, ["x"])
    with c:
        resp = c.post("/chat", json={"message": "q", "conversions": ["nope"]})
    assert resp.status_code == 400 and "nope" in resp.json()["detail"]


def test_conversions_endpoint(make_client, shop_db):
    c, _ = make_client(shop_db, ["x"], default_enabled=["currency"])
    with c:
        body = c.get("/conversions").json()
    assert body["default"] == ["currency"]
    assert body["available"][0]["name"] == "currency"


def test_bad_default_fails_at_startup(tmp_path, shop_db):
    cfg = {
        "default_db": str(shop_db),
        "llm": {"model_name": "x", "model_provider": "groq"},
        "postprocess": {"default_enabled": ["bogus"]},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(UnknownConverter):
        create_app(str(p), llm=FakeLLM(["x"]))
