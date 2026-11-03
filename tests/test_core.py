import copy
import json
import unittest
from pathlib import Path
from unittest.mock import Mock

from pribor_ai import core
from pribor_ai.provider import ProviderError


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "products.json"


def product(**changes):
    raw = {
        "name": "Измеритель М-1",
        "manufacturer": "Завод",
        "specs": {"Диапазон": "0–10 В", "Погрешность": "±0,5 %"},
        "description": "Прибор измеряет напряжение. Он используется для проверки электрических цепей.",
    }
    raw.update(changes)
    return core.normalize_product(raw)


def valid_plan(catalog):
    return core.demo_plan(catalog)


class InputTests(unittest.TestCase):
    def test_single_batch_and_utf8_bom(self):
        for raw in ({"name": "Прибор"}, [{"name": "Прибор"}]):
            parsed = core.parse_products("\ufeff" + json.dumps(raw, ensure_ascii=False))
            self.assertEqual([p["name"] for p in parsed], ["Прибор"])

    def test_bad_json_duplicate_keys_and_nonfinite_numbers(self):
        for raw in ('{"name":', '{"name":"A","name":"B"}',
                    '{"name":"A","specs":{"x":"1","x":"2"}}',
                    '{"name":"A","source":{"x":NaN}}',
                    '{"name":"A","source":{"x":Infinity}}'):
            with self.subTest(raw=raw), self.assertRaises(core.InputError):
                core.parse_products(raw)

    def test_json_container_and_batch_bounds(self):
        for raw in (None, 4, True, "text", [], [None], [{"name": "X"}] * 21):
            with self.subTest(raw=raw), self.assertRaises(core.InputError):
                core.parse_products(json.dumps(raw))

    def test_required_name_and_wrong_field_types(self):
        cases = [
            {}, {"name": " \n "}, {"name": None}, {"name": 1},
            {"name": "X", "manufacturer": 5}, {"name": "X", "description": []},
            {"name": "X", "specs": ["wrong"]}, {"name": "X", "specs": {"x": 1}},
            {"name": "X", "specs": {"": "1"}}, {"name": "X", "expected_specs": None},
            {"name": "X", "expected_specs": "range"}, {"name": "X", "expected_specs": [False]},
            {"name": "X", "source": []}, {"name": "X", "unexpected": "value"},
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(core.InputError):
                core.parse_products(json.dumps(raw))

    def test_falsey_nonobject_specs_are_rejected(self):
        # Regression: `raw.get("specs") or {}` used to silently accept these types.
        for value in ([], "", 0, False):
            with self.subTest(value=value), self.assertRaises(core.InputError):
                core.parse_products(json.dumps({"name": "X", "specs": value}))

    def test_null_optional_values_are_treated_as_missing(self):
        parsed = core.parse_products('{"name":"X","manufacturer":null,"description":null,"specs":null}')[0]
        card = core.generate(parsed, demo=True)
        self.assertEqual(card["manufacturer"], "данных нет")
        self.assertTrue({"manufacturer", "description", "specs"}.issubset(
            {item["field"] for item in card["missing_fields"]}))

    def test_normalized_spec_labels_cannot_silently_overwrite_values(self):
        raw = {"name": "Прибор", "specs": {"Диапазон измерений": "0–10 В", " Диапазон   измерений ": "0–20 В"}}
        with self.assertRaises(core.InputError):
            core.parse_products(json.dumps(raw, ensure_ascii=False))

    def test_missing_optional_data_is_explicit_in_card(self):
        parsed = core.parse_products('{"name":"Прибор"}')[0]
        card = core.generate(parsed, demo=True)
        self.assertEqual(card["status"], "needs_data")
        self.assertEqual(card["manufacturer"], "данных нет")
        self.assertEqual(card["commercial"]["tasks"], ["данных нет"])
        self.assertEqual({x["field"] for x in card["missing_fields"]},
                         {"manufacturer", "description", "specs", "commercial.tasks"})
        self.assertFalse(card["checks"]["description_length_ok"])
        self.assertTrue(card["warnings"])
        self.assertLess(len(card["seo_description"]), 800)

    def test_empty_and_expected_specs_are_visible_without_invention(self):
        p = product(specs={"Диапазон": None, "Питание": "  "},
                    expected_specs=["Диапазон", "Интерфейс", "Интерфейс"])
        card = core.generate(p, demo=True)
        missing = [m["field"] for m in card["missing_fields"]]
        self.assertEqual(missing.count("specs.Интерфейс"), 1)
        self.assertIn("specs.Диапазон", missing)
        self.assertIn("specs.Питание", missing)
        self.assertEqual({s["name"]: s["value"] for s in card["key_specs"]},
                         {"Диапазон": "данных нет", "Питание": "данных нет", "Интерфейс": "данных нет"})
        self.assertEqual(card["status"], "needs_data")

    def test_input_size_and_field_limits(self):
        cases = [
            {"name": "X" * 201}, {"name": "X", "manufacturer": "x" * 201},
            {"name": "X", "description": "x" * 12001},
            {"name": "X", "specs": {"x": "x" * 1001}},
            {"name": "X", "specs": {str(i): "x" for i in range(61)}},
            {"name": "X", "expected_specs": [str(i) for i in range(31)]},
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(core.InputError):
                core.parse_products(json.dumps(raw))
        with self.assertRaises(core.InputError):
            core.parse_products(" " * 1_000_001)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.product = product()
        self.catalog = core.catalog_for(self.product)
        self.plan = valid_plan(self.catalog)

    def test_plan_requires_exact_schema_and_string_lists(self):
        malformed = [None, [], {}, {**self.plan, "seo_description": "Придуманный текст"}]
        for key in self.plan:
            missing = copy.deepcopy(self.plan)
            del missing[key]
            malformed.append(missing)
            for value in (None, "d0", [1], ["d0", "d0"]):
                malformed.append({**self.plan, key: value})
        for plan in malformed:
            with self.subTest(plan=plan), self.assertRaises(core.PlanError):
                core.validate_plan(plan, self.catalog)

    def test_free_text_foreign_ids_and_numeric_claims_cannot_be_injected(self):
        for key in ("description_ids", "key_spec_ids", "purpose_ids"):
            for claim in ("Bluetooth 5.3", "Диапазон: 0–999 В", "чужой-product:d0", "d999", "s999"):
                candidate = {**self.plan, key: [claim]}
                with self.subTest(key=key, claim=claim), self.assertRaises(core.PlanError):
                    core.render_card(self.product, candidate, self.catalog, {"mode": "live"})

    def test_fact_kinds_cannot_be_mixed(self):
        for key, ids in (("key_spec_ids", ["d0"]), ("purpose_ids", ["s0"])):
            with self.subTest(key=key), self.assertRaises(core.PlanError):
                core.validate_plan({**self.plan, key: ids}, self.catalog)

    def test_queries_have_three_to_five_allowed_distinct_intents(self):
        for intents in ([], ["buy", "specs"], ["buy"] * 3,
                        ["buy", "specs", "fiction"], list(core.QUERY_TEMPLATES) + ["extra"]):
            with self.subTest(intents=intents), self.assertRaises(core.PlanError):
                core.validate_plan({**self.plan, "query_intents": intents}, self.catalog)

    def test_reused_ids_resolve_only_to_current_product(self):
        # Both products use d0/s0; sending another product's valid plan cannot carry its values.
        other = product(name="Другой прибор", specs={"Диапазон": "999999 В"},
                        description="Прибор поддерживает выдуманный-интерфейс.")
        foreign_plan = valid_plan(core.catalog_for(other))
        card = core.render_card(self.product, foreign_plan, self.catalog, {"mode": "live"})
        output = json.dumps(card, ensure_ascii=False)
        self.assertNotIn("999999", output)
        self.assertNotIn("выдуманный-интерфейс", output)
        self.assertIn("0–10 В", output)

    def test_complex_accuracy_value_is_kept_whole(self):
        value = "±1.5 °C Или 1.5 % от изм. знач. (+0.1 … +400 °C) ±2 °C Или ±2 % от изм. знач. (-30 … 0 °C)"
        p = product(specs={"Погрешность": value})
        catalog = core.catalog_for(p)
        specs = [f for f in catalog["facts"] if f["kind"] == "spec"]
        self.assertEqual(len(specs), 1)
        card = core.generate(p, demo=True)
        self.assertEqual(card["key_specs"][0]["value"], value)
        self.assertEqual(card["all_specs"]["Погрешность"], value)
        self.assertIn(value, card["seo_description"])

    def test_whole_facts_only_when_description_limit_is_reached(self):
        sentence = "А" * 900 + "."
        second = "Б" * 400 + "."
        p = product(description=sentence + " " + second, specs={})
        card = core.generate(p, demo=True)
        self.assertIn(sentence, card["seo_description"])
        self.assertNotIn("Б", card["seo_description"])
        self.assertLessEqual(len(card["seo_description"]), 1200)
        self.assertEqual(card["seo_description"], core.catalog_for(p)["intro"] + " " + sentence)

    def test_long_title_is_bounded_and_warned(self):
        card = core.generate(product(name="Очень длинное обозначение модели " * 5), demo=True)
        self.assertLessEqual(len(card["seo_title"]), 70)
        self.assertTrue(any("заголовок" in w for w in card["warnings"]))

    def test_demo_is_marked_and_never_calls_requester(self):
        requester = Mock(side_effect=AssertionError("No API in demo"))
        card = core.generate(self.product, requester=requester, demo=True)
        requester.assert_not_called()
        self.assertEqual(card["generation"]["mode"], "demo")
        self.assertIsNone(card["generation"]["model"])
        self.assertIn("без вызова LLM API", card["generation"]["note"])
        self.assertIn("**demo**", core.markdown([card]))

    def test_live_records_provider_identity_and_uses_catalog(self):
        requester = Mock(return_value={"data": self.plan, "model": "vendor/model:free", "request_id": "req-test"})
        card = core.generate(self.product, requester=requester)
        self.assertEqual(card["generation"]["mode"], "live")
        self.assertEqual(card["generation"]["model"], "vendor/model:free")
        self.assertEqual(card["generation"]["request_id"], "req-test")
        self.assertEqual(requester.call_args.args[1], self.catalog)

    def test_api_failure_is_not_converted_to_demo(self):
        failure = ProviderError("network_error", "Нет соединения")
        with self.assertRaises(ProviderError) as caught:
            core.generate(self.product, requester=Mock(side_effect=failure))
        self.assertIs(caught.exception, failure)

    def test_no_reserve_filler_when_model_selection_already_meets_minimum(self):
        product = core.parse_products(EXAMPLES.read_text(encoding="utf-8"))[0]
        catalog = core.catalog_for(product)
        plan = core.demo_plan(catalog)
        plan["description_ids"] = [f["id"] for f in catalog["facts"] if f["kind"] == "description"]
        card = core.render_card(product, plan, catalog, {"mode": "test"})
        self.assertTrue(800 <= len(card["seo_description"]) < 950)
        self.assertTrue(all(e["source_field"] == "description" for e in card["evidence"]))

    def test_two_real_inputs_meet_output_contract(self):
        products = core.parse_products(EXAMPLES.read_text(encoding="utf-8"))
        self.assertEqual(len(products), 2)
        for p in products:
            with self.subTest(name=p["name"]):
                card = core.generate(p, demo=True)
                self.assertLessEqual(len(card["seo_title"]), 70)
                self.assertTrue(800 <= len(card["seo_description"]) <= 1200)
                self.assertTrue(3 <= len(card["search_queries"]) <= 5)
                self.assertEqual(len(card["search_queries"]), len(set(card["search_queries"])))
                self.assertEqual(card["all_specs"], p["specs"])
                for spec in card["key_specs"]:
                    if spec["name"] in p["specs"]:
                        self.assertEqual(spec["value"], p["specs"][spec["name"]])
                    else:
                        self.assertIn(spec["name"], p["expected_specs"])
                        self.assertEqual(spec["value"], "данных нет")
                self.assertEqual(card["checks"]["description_characters"], len(card["seo_description"]))
                # Every included fragment has evidence and exactly reconstructs the description.
                reconstructed = core.catalog_for(p)["intro"] + " " + " ".join(e["text"] for e in card["evidence"])
                self.assertEqual(card["seo_description"], reconstructed)
                expected_missing = {"specs." + name for name in p["expected_specs"] if not p["specs"].get(name)}
                self.assertEqual({item["field"] for item in card["missing_fields"]}, expected_missing)
                self.assertEqual(card["status"], "needs_data" if expected_missing else "draft")


if __name__ == "__main__":
    unittest.main()
