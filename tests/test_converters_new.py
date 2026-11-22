import pandas as pd
import pytest

from src.sql_agent.postprocess import apply_converters, build_all, build_converters
from src.sql_agent.postprocess.datetime_iso import DatetimeConverter
from src.sql_agent.postprocess.durations import DurationsConverter
from src.sql_agent.postprocess.numbers import NumbersConverter
from src.sql_agent.postprocess.units import UnitsConverter


def S(*values):
    return pd.Series(list(values))


# -- units ----------------------------------------------------------------
class TestUnits:
    u = UnitsConverter({})

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("5 kg", 5.0), ("12.5 lb", 5.669905), ("1000 g", 1.0), ("10 oz", 0.283495),
            ("30 cm", 0.0003), ("3 km", 3.0), ("1 mile", 1.609344), ("2 ft", 0.0006096),
            ("1 gal", 3.785412), ("500 ml", 0.5), ("2 litres", 2.0),
            ("98 °F", 36.666667), ("25 degC", 25.0), ("100 celsius", 100.0),
        ],
    )
    def test_converts(self, text, expected):
        assert self.u.convert_value(text) == pytest.approx(expected, abs=1e-5)

    def test_temperature_target_f(self):
        f = UnitsConverter({"targets": {"temperature": "F"}})
        assert f.convert_value("100 degC") == pytest.approx(212.0)
        assert f.convert_value("0 degC") == pytest.approx(32.0)

    def test_detect_requires_one_dimension(self):
        assert self.u.detect(S("5 kg", "2 lb", "1 oz"))
        assert not self.u.detect(S("5 kg", "3 cm", "2 km", "1 m", "9 lb"))  # mixed

    @pytest.mark.parametrize(
        "values",
        [("A100", "B200"), ("12345", "67890"), ("Male", "Female"), ("5 apples", "2 pears"),
         ("5 K", "6 K"), ("10 c", "20 f")],  # K/c/f deliberately not units
    )
    def test_false_positives(self, values):
        assert not self.u.detect(S(*values))

    def test_ambiguous_tokens_are_decided(self):
        assert self.u.convert_value("1 oz") == pytest.approx(0.028349523, abs=1e-6)  # mass
        assert self.u.convert_value("5 min") is None  # time belongs to durations

    def test_bad_target_fails_at_construction(self):
        with pytest.raises(ValueError):
            UnitsConverter({"targets": {"length": "kg"}})
        with pytest.raises(ValueError):
            UnitsConverter({"targets": {"length": "furlongish"}})

    def test_describe_names_the_target(self):
        assert self.u.describe("5 kg") == "to kg (mass)"


# -- durations ------------------------------------------------------------
class TestDurations:
    d = DurationsConverter({})

    @pytest.mark.parametrize(
        "text,minutes",
        [("2h 30m", 150), ("1d 4h", 1680), ("90 min", 90), ("45 mins", 45),
         ("1:30:00", 90), ("1 hr", 60), ("30 sec", 0.5)],
    )
    def test_converts(self, text, minutes):
        assert self.d.convert_value(text) == pytest.approx(minutes)

    @pytest.mark.parametrize("text", ["5 m", "10:30", "5h 3h", "abc", "2 weeks", "12"])
    def test_rejects(self, text):
        assert self.d.convert_value(text) is None

    def test_target_hours(self):
        assert DurationsConverter({"target": "hours"}).convert_value("90 min") == 1.5

    def test_bad_target(self):
        with pytest.raises(ValueError):
            DurationsConverter({"target": "fortnights"})

    def test_detect_and_false_positives(self):
        assert self.d.detect(S("2h 30m", "45 min", "1:00:00"))
        assert not self.d.detect(S("10:30", "11:45"))  # clock times
        assert not self.d.detect(S("Gold", "Silver"))


# -- numbers --------------------------------------------------------------
class TestNumbers:
    n = NumbersConverter({})

    @pytest.mark.parametrize(
        "text,expected",
        [("1,234.50", 1234.5), ("1.2K", 1200.0), ("3M", 3e6), ("2.5B", 2.5e9),
         ("(123)", -123.0), ("(1,234.50)", -1234.5), ("-1,000", -1000.0), ("$1,200", 1200.0)],
    )
    def test_converts(self, text, expected):
        assert self.n.convert_value(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["12345", "5m", "1.234,56", "abc", "1,23", "1k "])
    def test_not_numbers_needing_conversion(self, text):
        assert self.n.convert_value(text) is None

    def test_plain_digit_columns_are_never_claimed(self):
        assert not self.n.detect(S("12345", "67890", "11111"))  # ids, zip codes
        assert not self.n.detect(S("2023", "2024"))

    def test_detect(self):
        assert self.n.detect(S("1,234", "5,000", "12,345.50"))
        assert self.n.detect(S("1.2K", "3M", "500K"))


# -- datetime -------------------------------------------------------------
class TestDatetime:
    t = DatetimeConverter({"dayfirst": True})

    @pytest.mark.parametrize(
        "text,iso",
        [("05/10/1973", "1973-10-05"), ("01/01/2023 0:00", "2023-01-01T00:00:00"),
         ("25/12/2020", "2020-12-25"), ("12 Mar 2024", "2024-03-12"),
         ("March 3, 2024", "2024-03-03"), ("2023-1-5", "2023-01-05")],
    )
    def test_converts(self, text, iso):
        assert self.t.convert_value(text) == iso

    def test_month_first_config(self):
        m = DatetimeConverter({"dayfirst": False})
        assert m.convert_value("05/10/1973") == "1973-05-10"
        assert m.convert_value("12/25/2020") == "2020-12-25"

    def test_opposite_order_value_is_not_guessed(self):
        assert self.t.convert_value("12/25/2020") is None  # only month-first fits

    def test_column_with_contradicting_order_is_left_alone(self):
        assert not self.t.detect(S("05/10/1973", "12/25/2020"))

    def test_already_iso_needs_no_conversion(self):
        assert not self.t.detect(S("2023-01-01", "2024-02-02"))

    def test_detect_and_false_positives(self):
        assert self.t.detect(S("05/10/1973", "25/12/2020"))
        assert self.t.detect(S("01/01/2023 0:00", "01/01/2023 0:01"))
        assert not self.t.detect(S("April", "May"))
        assert not self.t.detect(S("12345", "67890"))
        assert not self.t.detect(S("Male", "Female"))

    def test_output_sorts_correctly_as_text(self):
        vals = ["05/10/1973", "01/02/2020", "30/11/1999"]
        iso = sorted(self.t.convert_value(v) for v in vals)
        assert iso == ["1973-10-05", "1999-11-30", "2020-02-01"]


# -- registry and interplay ----------------------------------------------
CFG = {
    "currency": {"base_currency": "SGD", "conversion_rates": {"SGD": 1.0, "USD": 1.35}},
    "units": {}, "durations": {}, "numbers": {}, "datetime": {},
}


def test_registry_orders_all_converters():
    assert [c.name for c in build_all(CFG)] == [
        "currency", "units", "durations", "numbers", "datetime"
    ]


def test_row_path_converts_each_column_with_its_converter():
    df = pd.DataFrame(
        {
            "price": ["SGD$ 1.00", "USD$ 1.00"],
            "weight": ["5 kg", "2 lb"],
            "stay": ["2h 30m", "45 min"],
            "views": ["1.2K", "3M"],
            "dob": ["05/10/1973", "25/12/2020"],
            "name": ["Ann", "Bob"],
        }
    )
    out, applied = apply_converters(df, build_all(CFG))
    assert {a.column: a.converter for a in applied} == {
        "price": "currency", "weight": "units", "stay": "durations",
        "views": "numbers", "dob": "datetime",
    }
    assert out["weight"].tolist()[0] == 5.0
    assert out["dob"].tolist() == ["1973-10-05", "2020-12-25"]
    assert out["name"].tolist() == ["Ann", "Bob"]  # untouched
    assert df["weight"].tolist() == ["5 kg", "2 lb"]  # input not mutated


def test_sql_functions_never_raise_on_bad_input():
    for conv in build_all(CFG):
        fn = conv.sql_function()
        for bad in (None, 42, 3.5, "", "gibberish", b"bytes"):
            assert fn(bad) is None or isinstance(fn(bad), (int, float, str))


def test_converters_are_fast_enough_to_run_per_row_inside_sqlite():
    """Regression: strptime-per-format made datetime ~66us/call and timed out a MIN.

    The bound is loose (50us) so it only fails on a real regression, not on noise.
    """
    import time

    samples = {
        "currency": "USD$ 723.34", "units": "3567 KM", "durations": "2h 30m",
        "numbers": "1,234.50", "datetime": "01/01/2023 0:00",
    }
    for conv in build_all(CFG):
        start = time.perf_counter()
        for _ in range(5000):
            conv.convert_value(samples[conv.name])
        per_call = (time.perf_counter() - start) / 5000
        assert per_call < 50e-6, f"{conv.name} took {per_call * 1e6:.1f}us per call"


# -- currency false positives (found on a real delivery database) ---------
def one(name):
    return build_converters(CFG, [name])[0]


def test_currency_does_not_claim_ids_that_look_like_codes():
    cur = one("currency")
    for ids in (("DRV0130", "DRV0087", "DRV0040"), ("ABC123", "XYZ999"), ("C0000", "C0007")):
        assert not cur.detect(S(*ids))
    assert cur.convert_value("DRV0130") is None


def test_currency_unknown_code_becomes_null_not_base():
    cur = one("currency")
    assert cur.convert_value("JPY$ 100") is None
    assert cur.convert_value("USD$ 100") == pytest.approx(135.0)


def test_currency_detect_needs_mostly_known_codes():
    cur = one("currency")
    assert cur.detect(S("SGD$ 1.00", "USD$ 2.00", "USD$ 3.00"))
    assert not cur.detect(S("USD$ 1.00", "JPY$ 2.00", "KRW$ 3.00", "THB$ 4.00"))
