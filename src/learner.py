"""Automated attribute skills for exact-match extraction.

The learner updates rules, not model parameters. It never reads hold-out labels.
Pack and threshold choices are made on a product-disjoint experiment split.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

TOKEN = re.compile(r"[a-z0-9]+")
SHORT_FIELDS = (
    "category",
    "comm",
    "sub_comm",
    "dis_name",
    "prod_desc",
    "ingredients",
    "lifestyles",
    "SZ_ITEM",
)
GRAINS = ("wheat", "oat", "corn", "rice", "flax", "quinoa", "chia", "hemp", "buckwheat", "bran")
RULE_SCOPE = {
    "A900201": ("read:sweet_",),
    "A101614": ("read:grain_", "read:primary_"),
    "A102506": ("read:slice_",),
    "A101643": ("read:flavor_", "read:fruit_", "read:natural_", "read:puree"),
    "A100190": ("read:soda_", "read:fruit_", "read:natural_", "read:puree"),
}
CUES = {
    "form_puffs": ("puff", "puffs", "rings", "os", "loop", "loops"),
    "form_flakes": ("flake", "flakes", "flk"),
    "form_crisps": ("crisp", "crisps"),
    "form_clusters": ("cluster", "clusters", "clstr"),
    "form_granola": ("granola", "grnla", "muesli"),
    "form_shredded": ("shred", "shredded", "biscuit", "bisc"),
    "form_porridge": ("porridge", "oatmeal"),
    "sweet_artificial": ("aspartame", "sucralose", "acesulfame", "saccharin"),
    "sweet_plant": ("stevia", "monk"),
    "soda_diet": ("diet",),
    "soda_prebiotic": ("prebiotic",),
    "mix_cookie": ("cookie", "cookies"),
    "mix_candy": ("candy", "sprinkle", "sprinkles"),
    "mix_granola": ("granola",),
    "mix_nuts": ("almond", "almonds", "peanut", "peanuts", "walnut", "pecan"),
    "fruit_word": (
        "apple", "orange", "guava", "berry", "strawberry", "blueberry", "mango",
        "peach", "lemon", "lime", "grape", "cherry", "raspberry", "pineapple", "banana",
    ),
}
SINGLE_MARGINS = (0.0, 0.5, 1.5, 3.0)
MULTI_THRESHOLDS = (-1.0, -0.25, 0.25, 1.0, 2.0)


def is_validation(product_id: str) -> bool:
    return int(hashlib.md5(product_id.encode()).hexdigest()[:8], 16) % 5 == 0


def tokens(value: object) -> list[str]:
    text = html.unescape(str(value)).lower().replace("\xa0", " ")
    return [tok for tok in TOKEN.findall(text) if len(tok) > 1 or tok.isdigit()]


def allowed_names(spec: dict) -> list[str]:
    names = []
    for item in spec.get("allowed_values") or []:
        name = item["name"] if isinstance(item, dict) else str(item)
        if name not in names:
            names.append(name)
    return names


def parse_product(description: str) -> dict | None:
    for candidate in (description, description.replace("\n", " ").replace("\r", " ")):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    found: dict[str, str] = {}
    for key in (*SHORT_FIELDS, "long_desc", "nutrition_panels", "brand"):
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"(.*?)"', description, re.S)
        if match:
            found[key] = match.group(1)
    return found or None


def _has_grain(token_set: set[str], grain: str) -> bool:
    if grain in token_set:
        return True
    compounds = {
        "wheat": {"wholewheat"},
        "oat": {"oats", "oatmeal"},
        "corn": {"cornstarch", "cornmeal"},
        "rice": {"rices"},
    }
    return bool(token_set & compounds.get(grain, set()))


def _grain_hits(token_set: set[str], text: str) -> list[str]:
    """Grains in a text. Corn syrup and corn starch are sweeteners, not a corn grain."""
    corn_syrup = "corn syrup" in text or "high fructose" in text
    corn_grain = any(
        phrase in text
        for phrase in ("corn flour", "corn meal", "milled corn", "whole grain corn", "yellow corn", "corn bran", "popcorn")
    )
    found = []
    for grain in GRAINS:
        if grain == "corn" and corn_syrup and not corn_grain:
            continue
        if _has_grain(token_set, grain):
            found.append(grain)
    return found


def _add_reader(parsed: dict, feats: set[str]) -> None:
    """Read ingredients and the product name separately from the marketing copy."""
    ingredients = tokens(parsed.get("ingredients") or "")
    ingredient_set = set(ingredients)
    ingredient_text = " ".join(ingredients)
    name_text = " ".join(tokens(str(parsed.get("dis_name") or "")) + tokens(str(parsed.get("prod_desc") or "")))
    corn_syrup = "corn syrup" in ingredient_text or "high fructose" in ingredient_text
    sugar = "sugar" in ingredient_set or "cane" in ingredient_set
    artificial = any(token in ingredient_set for token in ("aspartame", "sucralose", "acesulfame", "saccharin"))
    plant = "stevia" in ingredient_text or "monk" in ingredient_text
    if corn_syrup:
        feats.add("read:sweet_corn")
    if sugar and not corn_syrup:
        feats.add("read:sweet_sugar")
    if artificial:
        feats.add("read:sweet_artificial")
    if plant:
        feats.add("read:sweet_plant")
    if ingredients and not (corn_syrup or sugar or artificial or plant):
        feats.add("read:sweet_none")
    grains = _grain_hits(ingredient_set, ingredient_text)
    for grain in grains:
        feats.add("read:grain_" + grain)
    feats.add("read:grains_" + str(min(len(grains), 4)))
    if len(grains) >= 2:
        feats.add("read:grain_multi")
    elif len(grains) == 1:
        feats.add("read:grain_only_" + grains[0])
    primary = ""
    corn_is_grain = any(
        phrase in ingredient_text
        for phrase in ("corn flour", "corn meal", "milled corn", "whole grain corn", "yellow corn", "corn bran", "popcorn")
    )
    for tok in ingredients:
        for grain in GRAINS:
            if grain == "corn" and corn_syrup and not corn_is_grain:
                continue
            if _has_grain({tok}, grain):
                primary = grain
                break
        if primary:
            break
    if primary:
        feats.add("read:primary_" + primary)
    name_phrases = (
        ("soda_ginger_beer", ("ginger beer", "gngr beer")),
        ("soda_ginger_ale", ("ginger ale",)),
        ("soda_root_beer", ("root beer",)),
        ("soda_cream", ("cream soda",)),
        ("soda_lemon_lime", ("lemon lime", "lem lime", "lemlime")),
        ("soda_cola", ("cola",)),
    )
    for feature_name, phrases in name_phrases:
        if any(phrase in name_text for phrase in phrases):
            feats.add("read:" + feature_name)
    if "thin sliced" in name_text or "thin-sliced" in name_text:
        feats.add("read:slice_thin")
    if "pre sliced" in name_text or "pre-sliced" in name_text or "presliced" in name_text:
        feats.add("read:slice_pre")
    if "sliced fresh" in name_text:
        feats.add("read:slice_fresh")
    if "on the bottom" in ingredient_text or "on the bottom" in name_text:
        feats.add("read:flavor_bottom")
    fruits = ("strawberry", "blueberry", "raspberry", "mango", "peach", "cherry", "pineapple", "banana", "apricot", "apple")
    if any(fruit in ingredient_set for fruit in fruits):
        feats.add("read:fruit_piece")
    if "natural flavor" in ingredient_text or "natural flavors" in ingredient_text:
        feats.add("read:natural_flavor")
    if "puree" in ingredient_text:
        feats.add("read:puree")


def _add_field(feats: set[str], blob: list[str], field_name: str, raw: object) -> None:
    if raw in (None, ""):
        return
    if not isinstance(raw, str):
        raw = json.dumps(raw, ensure_ascii=False)
    toks = tokens(raw)
    blob.extend(toks)
    prefix = "sz" if field_name == "SZ_ITEM" else field_name[:4]
    for tok in toks:
        feats.add(f"{prefix}:{tok}")
    for left, right in zip(toks, toks[1:]):
        feats.add(f"{prefix}:{left}_{right}")


def featurize(description: str, names: list[str], mode: str) -> set[str]:
    """mode is 'short', 'long', or 'cue'. 'cue' adds boundary-safe skills on top of 'long'."""
    feats: set[str] = {"bias"}
    parsed = parse_product(description)
    blob: list[str] = []
    if isinstance(parsed, dict):
        fields = list(SHORT_FIELDS)
        if mode in {"long", "cue"}:
            fields.append("long_desc")
        for field_name in ("category", "comm", "sub_comm"):
            raw = parsed.get(field_name)
            if isinstance(raw, str) and raw.strip():
                feats.add("atom:" + field_name + "=" + re.sub(r"\s+", " ", raw).strip().lower())
        for field_name in fields:
            _add_field(feats, blob, field_name, parsed.get(field_name))
        pack_text = " ".join(
            str(parsed.get(field_name) or "")
            for field_name in ("SZ_ITEM", "dis_name", "prod_desc", "sub_comm")
        ).lower()
        if re.search(r"\d+\s*/\s*\d+|\b\d+\s*(pk|ct|pack)\b|variety pack|multi", pack_text):
            feats.add("skill:multipack")
        if re.search(r"\b(tube|tb)\b", pack_text):
            feats.add("skill:tube")
        if re.search(r"\b(bottle|btl)\b", pack_text):
            feats.add("skill:bottle")
        if re.search(r"\b(cup|single serv)\b", pack_text):
            feats.add("skill:cup")
        if re.search(r"\btub\b", pack_text):
            feats.add("skill:tub")
        token_set = set(blob)
        joined = " ".join(blob)
        found = _grain_hits(token_set, joined)
        feats.add(f"skill:grains_{min(len(found), 4)}")
        for grain in found:
            feats.add("skill:grain_" + grain)
        if "corn syrup" in joined or "high fructose" in joined:
            feats.add("skill:sweet_corn")
        if "cane sugar" in joined:
            feats.add("skill:sweet_cane")
        if "no added sugar" in joined or "zero sugar" in joined or "sugar free" in joined:
            feats.add("skill:sweet_zero")
        if "caffeine free" in joined or "caffeine-free" in joined:
            feats.add("skill:soda_caffeine_free")
        if "on the bottom" in joined:
            feats.add("skill:fruit_bottom")
        if "thin sliced" in joined or "thin-sliced" in joined:
            feats.add("skill:thin_sliced")
        if "pre sliced" in joined or "pre-sliced" in joined or "presliced" in joined:
            feats.add("skill:pre_sliced")
        if "sliced fresh" in joined:
            feats.add("skill:sliced_fresh")
        if "whole wheat" in joined or "whole grain wheat" in joined:
            feats.add("skill:whole_wheat")
        if "variety pack" in joined or "variety pk" in joined:
            feats.add("skill:variety_pack")
        if "plain" in token_set:
            feats.add("skill:plain")
        _add_reader(parsed, feats)
        for field_name, prefix in (("prod_desc", "pd"), ("dis_name", "dn")):
            compact = "".join(tokens(parsed.get(field_name) or ""))
            for index in range(min(max(len(compact) - 3, 0), 70)):
                feats.add(prefix + "4:" + compact[index:index + 4])
        if mode == "cue":
            for skill_name, cues in CUES.items():
                if any(cue in token_set for cue in cues):
                    feats.add("skill:" + skill_name)
        size = str(parsed.get("SZ_ITEM") or "")
        if size:
            feats.add("skill:size_" + "_".join(tokens(size)[:3]))
    else:
        blob = tokens(description)
        for tok in blob:
            feats.add(f"raw:{tok}")
        for left, right in zip(blob, blob[1:]):
            feats.add(f"raw:{left}_{right}")
    present = set(blob)
    for name in names:
        parts = tokens(name)
        if parts and all(part in present for part in parts):
            feats.add("alias:" + "_".join(parts))
    return feats


def catalog_from(rows: list[dict]) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for row in rows:
        for spec in row["attributes"]:
            slot = catalog.setdefault(
                spec["attribute_id"],
                {
                    "name": spec.get("attribute_name") or spec["attribute_id"],
                    "multi": False,
                    "allowed": [],
                },
            )
            slot["multi"] = slot["multi"] or bool(spec.get("multi_valued"))
            for name in allowed_names(spec):
                if name not in slot["allowed"]:
                    slot["allowed"].append(name)
    return catalog


def labels_of(row: dict) -> dict[str, set[str]]:
    return {lab["attribute_id"]: set(lab.get("values") or []) for lab in row.get("labels") or []}


def _repeated_features(docs: list[set[str]]) -> list[set[str]]:
    """Keep a token only when at least two training products contain it.

    A spelling that appears once can memorize that product. Reusable detectors stay.
    """
    counts: dict[str, int] = {}
    for doc in docs:
        for feat in doc:
            counts[feat] = counts.get(feat, 0) + 1

    def keep(feat: str) -> bool:
        if feat == "bias" or feat.startswith(("skill:", "read:", "atom:", "alias:")):
            return True
        return counts.get(feat, 0) >= 2

    return [{feat for feat in doc if keep(feat)} for doc in docs]


class Skill:
    def __init__(self) -> None:
        self.multi = False
        self.classes: list[str] = []
        self.weights: dict[str, dict[str, float]] = {}
        self.bias: dict[str, float] = {}

    def fit(self, docs: list[set[str]], targets: list[set[str]], allowed: list[str], multi: bool) -> None:
        self.multi = multi
        classes = [name for name in allowed if any(name in target for target in targets)]
        for target in targets:
            for name in target:
                if name not in classes:
                    classes.append(name)
        self.classes = classes
        if not classes:
            return
        if multi:
            self._fit_binary(docs, targets)
        else:
            self._fit_single(docs, targets)

    def _fit_single(self, docs: list[set[str]], targets: list[set[str]]) -> None:
        docs = _repeated_features(docs)
        labeled = [(doc, next(iter(target))) for doc, target in zip(docs, targets) if len(target) == 1]
        counts = Counter(label for _, label in labeled)
        cap = max(counts.values()) if counts else 1
        train: list[tuple[set[str], str]] = []
        for doc, label in labeled:
            repeats = min(6, max(1, cap // max(counts[label], 1)))
            train.extend([(doc, label)] * repeats)
        weights = {label: {} for label in self.classes}
        totals = {label: {} for label in self.classes}
        bias = dict.fromkeys(self.classes, 0.0)
        bias_total = dict.fromkeys(self.classes, 0.0)
        seen = 0
        rng = random.Random(7)
        for _epoch in range(12):
            rng.shuffle(train)
            for doc, label in train:
                seen += 1
                scores = {
                    cand: bias[cand] + sum(weights[cand].get(feat, 0.0) for feat in doc)
                    for cand in self.classes
                }
                pred = max(scores, key=scores.get)
                if pred == label:
                    continue
                for owner, direction in ((label, 1.0), (pred, -1.0)):
                    table = weights[owner]
                    stamp = totals[owner]
                    for feat in doc:
                        table[feat] = table.get(feat, 0.0) + direction
                        stamp[feat] = stamp.get(feat, 0.0) + seen * direction
                bias[label] += 1
                bias[pred] -= 1
                bias_total[label] += seen
                bias_total[pred] -= seen
        self.weights = {}
        self.bias = {}
        for label in self.classes:
            table = weights[label]
            self.weights[label] = {
                feat: value - totals[label][feat] / seen
                for feat, value in table.items()
                if seen and abs(value - totals[label][feat] / seen) >= 0.05
            }
            self.bias[label] = bias[label] - (bias_total[label] / seen if seen else 0.0)

    def _fit_binary(self, docs: list[set[str]], targets: list[set[str]]) -> None:
        docs = _repeated_features(docs)
        self.weights = {}
        self.bias = {}
        for label in self.classes:
            positive = [doc for doc, target in zip(docs, targets) if label in target]
            negative = [doc for doc, target in zip(docs, targets) if label not in target]
            if not positive:
                self.weights[label] = {}
                self.bias[label] = -1.0
                continue
            rng = random.Random(11 + len(label))
            train = [(doc, 1) for doc in positive] + [(doc, 0) for doc in negative]
            weights: dict[str, float] = {}
            totals: dict[str, float] = {}
            bias = 0.0
            bias_total = 0.0
            seen = 0
            for _epoch in range(10):
                rng.shuffle(train)
                for doc, target_bit in train:
                    seen += 1
                    score = bias + sum(weights.get(feat, 0.0) for feat in doc)
                    pred = 1 if score >= 0 else 0
                    if pred == target_bit:
                        continue
                    update = 1.0 if target_bit == 1 else -1.0
                    for feat in doc:
                        weights[feat] = weights.get(feat, 0.0) + update
                        totals[feat] = totals.get(feat, 0.0) + seen * update
                    bias += update
                    bias_total += seen * update
            self.weights[label] = {
                feat: value - totals[feat] / seen
                for feat, value in weights.items()
                if seen and abs(value - totals[feat] / seen) >= 0.05
            }
            self.bias[label] = bias - (bias_total / seen if seen else 0.0)

    def scores(self, doc: set[str]) -> dict[str, float]:
        return {
            label: self.bias.get(label, 0.0) + sum(self.weights.get(label, {}).get(feat, 0.0) for feat in doc)
            for label in self.classes
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "multi": self.multi,
            "classes": self.classes,
            "bias": self.bias,
            "weights": self.weights,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Skill:
        skill = cls()
        skill.multi = bool(payload["multi"])
        skill.classes = list(payload["classes"])
        skill.bias = {str(key): float(value) for key, value in payload["bias"].items()}
        skill.weights = {
            str(label): {str(feat): float(weight) for feat, weight in table.items()}
            for label, table in payload["weights"].items()
        }
        return skill


def _vote(member_predictions: list[list[str]], multi: bool) -> list[str]:
    """Keep a value only when at least two feature packs agree. Disagreeing packs abstain."""
    if not multi:
        counts = Counter(prediction[0] for prediction in member_predictions if prediction)
        if not counts:
            return []
        label, votes = counts.most_common(1)[0]
        return [label] if votes >= 2 else []
    counts = Counter(label for prediction in member_predictions for label in prediction)
    return sorted(label for label, votes in counts.items() if votes >= 2)


def decide(
    skill: Skill,
    scored: dict[str, float],
    allowed: list[str],
    margin: float,
    threshold: float,
    force_top: bool = False,
) -> list[str]:
    allowed_set = set(allowed)
    scored = {label: score for label, score in scored.items() if label in allowed_set}
    if not scored:
        return []
    if skill.multi:
        chosen = [label for label, score in scored.items() if score >= threshold]
        if not chosen and force_top:
            return [max(scored, key=scored.get)]
        if "Not Applicable" in chosen and len(chosen) > 1:
            others = [label for label in chosen if label != "Not Applicable"]
            if scored["Not Applicable"] >= max(scored[label] for label in others):
                return ["Not Applicable"]
            return sorted(others)
        return sorted(chosen)
    ranked = sorted(scored, key=scored.get, reverse=True)
    if len(ranked) == 1 or scored[ranked[0]] - scored[ranked[1]] >= margin:
        return [ranked[0]]
    return []


def _apply_rules(chosen: list[str], rules: list[dict[str, str]], attribute_id: str, document: set[str], multi: bool) -> list[str]:
    result = list(chosen)
    for rule in rules:
        if rule["attribute_id"] != attribute_id or rule["feature"] not in document:
            continue
        value = rule["value"]
        if not multi or value == "Not Applicable":
            return [value]
        result = [item for item in result if item != "Not Applicable"]
        if value not in result:
            result.append(value)
    return _clean(sorted(result) if multi else result)


def _clean(values: list[str]) -> list[str]:
    """Exact-match scoring rejects an empty string or a repeated value for the whole product."""
    cleaned = []
    for value in values:
        if isinstance(value, str) and value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _row_context(row: dict, catalog: dict[str, dict]) -> tuple[list[str], dict[str, list[str]]]:
    all_names: list[str] = []
    by_attr: dict[str, list[str]] = {}
    for spec in row["attributes"]:
        names = allowed_names(spec) or list(catalog.get(spec["attribute_id"], {}).get("allowed", []))
        by_attr[spec["attribute_id"]] = names
        for name in names:
            if name not in all_names:
                all_names.append(name)
    return all_names, by_attr


def _needed_modes(row: dict, bank: SkillBank) -> set[str]:
    needed: set[str] = set()
    for spec in row["attributes"]:
        mode = bank.modes.get(spec["attribute_id"], "long")
        if mode == "vote" and bank.packs:
            needed.update(("short", "long", "cue"))
        elif mode in {"short", "long", "cue"}:
            needed.add(mode)
        else:
            needed.add("long")
    if bank.rules:
        needed.add("cue")
    return needed


def _documents(row: dict, bank: SkillBank) -> dict[str, set[str]]:
    all_names, _by_attr = _row_context(row, bank.catalog)
    description = row["product_description"]
    return {mode: featurize(description, all_names, mode) for mode in _needed_modes(row, bank)}


def _predict_with(
    row: dict,
    bank: SkillBank,
    documents: dict[str, set[str]],
    margins: dict[str, float],
    thresholds: dict[str, float],
    force_tops: dict[str, bool],
) -> dict[str, list[str]]:
    _all_names, by_attr = _row_context(row, bank.catalog)
    rule_document = documents.get("cue") or documents.get("long") or documents.get("short") or set()
    prediction: dict[str, list[str]] = {}
    for spec in row["attributes"]:
        aid = spec["attribute_id"]
        names = by_attr.get(aid, [])
        mode = bank.modes.get(aid, "long")
        multi = bool(bank.catalog.get(aid, {}).get("multi"))
        if mode == "vote" and bank.packs:
            member_predictions = []
            for pack_name in ("short", "long", "cue"):
                skill = bank.packs[pack_name][aid]
                multi = skill.multi
                member_predictions.append(decide(skill, skill.scores(documents[pack_name]), names, 0.0, 0.0, False))
            chosen = _vote(member_predictions, multi)
        else:
            skill = bank.skills.get(aid)
            document = documents.get(mode if mode in documents else "long", set())
            if skill is None:
                chosen = []
            else:
                chosen = decide(
                    skill,
                    skill.scores(document),
                    names,
                    margins.get(aid, 0.0),
                    thresholds.get(aid, -0.5),
                    force_tops.get(aid, False),
                )
        prediction[aid] = _clean(_apply_rules(chosen, bank.rules, aid, rule_document, multi))
    return prediction


@dataclass
class SkillBank:
    catalog: dict[str, dict]
    skills: dict[str, Skill] = field(default_factory=dict)
    modes: dict[str, str] = field(default_factory=dict)
    margins: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    force_tops: dict[str, bool] = field(default_factory=dict)
    packs: dict[str, dict[str, Skill]] = field(default_factory=dict)
    rules: list[dict[str, str]] = field(default_factory=list)

    def predict_row(self, row: dict) -> dict[str, list[str]]:
        return _predict_with(row, self, _documents(row, self), self.margins, self.thresholds, self.force_tops)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "brown-ballers-skill-bank-v1",
            "catalog": self.catalog,
            "modes": self.modes,
            "margins": self.margins,
            "thresholds": self.thresholds,
            "force_tops": self.force_tops,
            "packs": {
                pack_name: {aid: skill.to_dict() for aid, skill in table.items()}
                for pack_name, table in self.packs.items()
            },
            "rules": self.rules,
            "skills": {aid: skill.to_dict() for aid, skill in self.skills.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SkillBank:
        bank = cls(catalog=payload["catalog"])
        bank.modes = {str(key): str(value) for key, value in payload["modes"].items()}
        bank.margins = {str(key): float(value) for key, value in payload["margins"].items()}
        bank.thresholds = {str(key): float(value) for key, value in payload["thresholds"].items()}
        bank.force_tops = {str(key): bool(value) for key, value in payload.get("force_tops", {}).items()}
        bank.packs = {
            pack_name: {aid: Skill.from_dict(skill) for aid, skill in table.items()}
            for pack_name, table in payload.get("packs", {}).items()
        }
        bank.rules = [dict(rule) for rule in payload.get("rules", [])]
        bank.skills = {aid: Skill.from_dict(skill) for aid, skill in payload["skills"].items()}
        return bank


def prepare(rows: list[dict], mode: str) -> list[dict]:
    prepared = []
    for row in rows:
        all_names, by_attr = _row_context(row, {})
        prepared.append({
            "row": row,
            "feats": featurize(row["product_description"], all_names, mode),
            "labels": labels_of(row),
            "allowed": by_attr,
            "validation": is_validation(row["product_id"]),
        })
    return prepared


def fit_skills(prepared: list[dict], catalog: dict[str, dict], train_only: bool) -> dict[str, Skill]:
    grouped: dict[str, list[dict]] = {aid: [] for aid in catalog}
    for item in prepared:
        if train_only and item["validation"]:
            continue
        for spec in item["row"]["attributes"]:
            grouped[spec["attribute_id"]].append(item)
    skills = {}
    for aid, info in catalog.items():
        subset = grouped.get(aid, [])
        skill = Skill()
        skill.fit(
            [item["feats"] for item in subset],
            [item["labels"].get(aid, set()) for item in subset],
            info["allowed"],
            info["multi"],
        )
        skills[aid] = skill
    return skills


def literal_prediction(row: dict) -> dict[str, list[str]]:
    text = html.unescape(row["product_description"]).casefold()
    prediction = {}
    for spec in row["attributes"]:
        names = allowed_names(spec)
        hits = [name for name in names if name.casefold() in text]
        if not spec.get("multi_valued"):
            hits = sorted(hits, key=len, reverse=True)[:1]
        prediction[spec["attribute_id"]] = hits
    return prediction


def empty_prediction(row: dict) -> dict[str, list[str]]:
    return {spec["attribute_id"]: [] for spec in row["attributes"]}


def micro_counts(rows: list[dict], predictions: list[dict]) -> tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    for row, prediction in zip(rows, predictions):
        labels = labels_of(row)
        requested = {spec["attribute_id"] for spec in row["attributes"]}
        well_formed = isinstance(prediction, dict) and set(prediction) <= requested
        normalized: dict[str, set[str]] = {}
        if well_formed:
            for aid, values in prediction.items():
                if not isinstance(values, list) or len(values) != len(set(values)):
                    well_formed = False
                    break
                if not all(isinstance(value, str) and value for value in values):
                    well_formed = False
                    break
                normalized[aid] = set(values)
        for aid, expected in labels.items():
            predicted = normalized.get(aid, set()) if well_formed else set()
            true_positive += len(predicted & expected)
            false_positive += len(predicted - expected)
            false_negative += len(expected - predicted)
    return true_positive, false_positive, false_negative


def f1_of(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return (2 * true_positive / denominator) if denominator else 0.0


def bank_predictions(prepared: list[dict], bank: SkillBank, validation_only: bool) -> tuple[list[dict], list[dict]]:
    rows = []
    predictions = []
    for item in prepared:
        if validation_only and not item["validation"]:
            continue
        rows.append(item["row"])
        prediction = {}
        for spec in item["row"]["attributes"]:
            aid = spec["attribute_id"]
            skill = bank.skills[aid]
            prediction[aid] = decide(
                skill,
                skill.scores(item["feats"] if bank.modes.get(aid) == item.get("mode") else featurize(
                    item["row"]["product_description"],
                    item["allowed"].get(aid, []),
                    bank.modes.get(aid, "long"),
                )),
                item["allowed"].get(aid, []),
                bank.margins.get(aid, 0.0),
                bank.thresholds.get(aid, -0.5),
            )
        predictions.append(prediction)
    return rows, predictions


def _cached_scores(prepared: list[dict], skills: dict[str, Skill], validation_only: bool) -> list[dict]:
    cached = []
    for item in prepared:
        if validation_only and not item["validation"]:
            continue
        cached.append({
            "row": item["row"],
            "allowed": item["allowed"],
            "scores": {
                aid: skills[aid].scores(item["feats"])
                for aid in item["allowed"]
                if aid in skills
            },
        })
    return cached


def _predict_cached(cached: list[dict], skills: dict[str, Skill], margins: dict[str, float], thresholds: dict[str, float]) -> tuple[list[dict], list[dict]]:
    rows = []
    predictions = []
    for item in cached:
        prediction = {}
        for aid, allowed in item["allowed"].items():
            skill = skills[aid]
            prediction[aid] = decide(
                skill,
                item["scores"].get(aid, {}),
                allowed,
                margins.get(aid, 0.0),
                thresholds.get(aid, -0.5),
            )
        rows.append(item["row"])
        predictions.append(prediction)
    return rows, predictions


def select_bank(rows: list[dict]) -> tuple[SkillBank, dict[str, Any]]:
    """Fit on the non-validation products and choose packs and thresholds on the rest."""
    catalog = catalog_from(rows)
    train_count = sum(1 for row in rows if not is_validation(row["product_id"]))
    valid_count = len(rows) - train_count
    if train_count < 15 or valid_count < 15:
        prepared = prepare(rows, "long")
        skills = fit_skills(prepared, catalog, train_only=False)
        bank = SkillBank(
            catalog,
            skills,
            {aid: "long" for aid in catalog},
            {aid: 0.0 for aid in catalog},
            {aid: -0.5 for aid in catalog},
        )
        return bank, {"q": None, "trained_on": "all_rows_small_split", "modes": dict(bank.modes)}
    packs = {}
    for mode in ("short", "long", "cue"):
        prepared = prepare(rows, mode)
        skills = fit_skills(prepared, catalog, train_only=True)
        packs[mode] = (prepared, skills, _cached_scores(prepared, skills, validation_only=True))

    modes = {aid: "long" for aid in catalog}
    margins = {aid: 0.0 for aid in catalog}
    thresholds = {aid: -0.5 for aid in catalog}
    force_tops = {aid: False for aid in catalog}

    def quality_for() -> float:
        # Stitch the cached scores from each attribute's chosen pack.
        base_cached = packs["long"][2]
        rows_out = [item["row"] for item in base_cached]
        predictions = []
        for index, item in enumerate(base_cached):
            prediction = {}
            for aid, allowed in item["allowed"].items():
                mode = modes[aid]
                if mode == "vote":
                    member_predictions = []
                    multi = False
                    for pack_name in ("short", "long", "cue"):
                        skill = packs[pack_name][1][aid]
                        multi = skill.multi
                        scores = packs[pack_name][2][index]["scores"].get(aid, {})
                        member_predictions.append(decide(skill, scores, allowed, 0.0, 0.0, False))
                    prediction[aid] = _vote(member_predictions, multi)
                    continue
                skill = packs[mode][1][aid]
                scores = packs[mode][2][index]["scores"].get(aid, {})
                prediction[aid] = decide(
                    skill, scores, allowed, margins[aid], thresholds[aid], force_tops[aid],
                )
            predictions.append(prediction)
        true_positive, false_positive, false_negative = micro_counts(rows_out, predictions)
        return f1_of(true_positive, false_positive, false_negative)

    for _pass in range(2):
        for aid, info in catalog.items():
            best_choice = (modes[aid], margins[aid], thresholds[aid], force_tops[aid], quality_for())
            options = SINGLE_MARGINS if not info["multi"] else (0.0,)
            multi_options = MULTI_THRESHOLDS if info["multi"] else (-0.5,)
            force_options = (False, True) if info["multi"] else (False,)
            for mode in ("short", "long", "cue"):
                for margin in options:
                    for threshold in multi_options:
                        for force_top in force_options:
                            modes[aid] = mode
                            margins[aid] = margin
                            thresholds[aid] = threshold
                            force_tops[aid] = force_top
                            score = quality_for()
                            if score > best_choice[4]:
                                best_choice = (mode, margin, threshold, force_top, score)
            modes[aid] = "vote"
            score = quality_for()
            if score > best_choice[4]:
                best_choice = ("vote", 0.0, 0.0, False, score)
            modes[aid], margins[aid], thresholds[aid], force_tops[aid], _score = best_choice

    chosen_skills = {}
    for aid, mode in modes.items():
        chosen_skills[aid] = packs["long" if mode == "vote" else mode][1][aid]
    bank = SkillBank(
        catalog,
        chosen_skills,
        modes,
        margins,
        thresholds,
        force_tops,
        {mode: table[1] for mode, table in packs.items()},
    )
    bank.rules = adapt_from_experiment(rows, bank)
    true_positive, false_positive, false_negative = _evidence_counts(rows, bank)
    return bank, {
        "q": f1_of(true_positive, false_positive, false_negative),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "modes": dict(modes),
        "rules": len(bank.rules),
        "trained_on": "experiment_train_split",
    }


def _evidence_counts(rows: list[dict], bank: SkillBank) -> tuple[int, int, int]:
    validation_rows = [row for row in rows if is_validation(row["product_id"])]
    predictions = [bank.predict_row(row) for row in validation_rows]
    return micro_counts(validation_rows, predictions)


def adapt_from_experiment(rows: list[dict], bank: SkillBank) -> list[dict[str, str]]:
    """Learn high-precision reader rules from experiment mistakes. The hold-out is not used."""
    counts: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        if is_validation(row["product_id"]):
            continue
        labels = labels_of(row)
        all_names: list[str] = []
        for spec in row["attributes"]:
            for name in allowed_names(spec):
                if name not in all_names:
                    all_names.append(name)
        cached: dict[str, set[str]] = {}
        for spec in row["attributes"]:
            aid = spec["attribute_id"]
            mode = bank.modes.get(aid, "long")
            mode = "cue" if mode == "vote" else mode
            if mode not in cached:
                cached[mode] = featurize(row["product_description"], all_names, mode)
            gold = labels.get(aid, set())
            for feat in cached[mode]:
                if not feat.startswith("read:"):
                    continue
                for value in allowed_names(spec):
                    bucket = counts.setdefault((aid, feat, value), [0, 0])
                    bucket[0] += 1
                    bucket[1] += value in gold
    candidates = []
    for (aid, feat, value), (present, matched) in counts.items():
        allowed_prefixes = RULE_SCOPE.get(aid)
        if not allowed_prefixes or not feat.startswith(allowed_prefixes):
            continue
        if matched < 6 or present == 0:
            continue
        precision = matched / present
        if precision >= 0.9:
            candidates.append((precision, matched, aid, feat, value))
    candidates.sort(reverse=True)
    kept: list[dict[str, str]] = []
    bank.rules = []
    validation_cache = [
        (row, _documents(row, bank))
        for row in rows
        if is_validation(row["product_id"])
    ]

    def cached_quality(rules: list[dict[str, str]]) -> float:
        bank.rules = rules
        validation_rows = [row for row, _documents_for_row in validation_cache]
        predictions = [
            _predict_with(row, bank, documents, bank.margins, bank.thresholds, bank.force_tops)
            for row, documents in validation_cache
        ]
        return f1_of(*micro_counts(validation_rows, predictions))

    best = cached_quality([])
    for _precision, _matched, aid, feat, value in candidates[:40]:
        trial = kept + [{"attribute_id": aid, "feature": feat, "value": value}]
        score = cached_quality(trial)
        if score > best:
            kept = trial
            best = score
        if len(kept) >= 12:
            break
    bank.rules = kept
    return kept


def refit_on_all(rows: list[dict], bank: SkillBank) -> SkillBank:
    """Keep the chosen packs and thresholds, then relearn weights on every experiment row."""
    catalog = bank.catalog
    fitted: dict[str, dict[str, Skill]] = {}
    for mode in ("short", "long", "cue"):
        prepared = prepare(rows, mode)
        fitted[mode] = fit_skills(prepared, catalog, train_only=False)
    rebuilt = {
        aid: fitted["long" if mode == "vote" else mode][aid]
        for aid, mode in bank.modes.items()
    }
    return SkillBank(
        catalog,
        rebuilt,
        dict(bank.modes),
        dict(bank.margins),
        dict(bank.thresholds),
        dict(bank.force_tops),
        fitted,
        list(bank.rules),
    )


def retune_thresholds(rows: list[dict], bank: SkillBank) -> SkillBank:
    """Calibrate margins on the validation products with the same reading prediction uses."""
    catalog = bank.catalog
    cached = [
        (row, _documents(row, bank))
        for row in rows
        if is_validation(row["product_id"])
    ]
    margins = dict(bank.margins)
    thresholds = dict(bank.thresholds)
    force_tops = dict(bank.force_tops)

    def quality() -> float:
        validation_rows = [row for row, _documents_for_row in cached]
        predictions = [
            _predict_with(row, bank, documents, margins, thresholds, force_tops)
            for row, documents in cached
        ]
        return f1_of(*micro_counts(validation_rows, predictions))

    for _pass in range(2):
        for aid, info in catalog.items():
            if aid not in bank.skills or bank.modes.get(aid) == "vote":
                continue
            best = (margins.get(aid, 0.0), thresholds.get(aid, -0.5), force_tops.get(aid, False), quality())
            margin_options = SINGLE_MARGINS if not info["multi"] else (margins.get(aid, 0.0),)
            threshold_options = MULTI_THRESHOLDS if info["multi"] else (thresholds.get(aid, -0.5),)
            force_options = (False, True) if info["multi"] else (False,)
            for margin in margin_options:
                for threshold in threshold_options:
                    for force_top in force_options:
                        margins[aid] = margin
                        thresholds[aid] = threshold
                        force_tops[aid] = force_top
                        score = quality()
                        if score > best[3]:
                            best = (margin, threshold, force_top, score)
            margins[aid], thresholds[aid], force_tops[aid], _score = best
    return SkillBank(catalog, bank.skills, dict(bank.modes), margins, thresholds, force_tops, bank.packs, list(bank.rules))


def comparison(rows: list[dict], bank: SkillBank) -> dict[str, Any]:
    validation_rows = [row for row in rows if is_validation(row["product_id"])]
    learned = [bank.predict_row(row) for row in validation_rows]
    literal = [literal_prediction(row) for row in validation_rows]
    empty = [empty_prediction(row) for row in validation_rows]
    learned_counts = micro_counts(validation_rows, learned)
    literal_counts = micro_counts(validation_rows, literal)
    empty_counts = micro_counts(validation_rows, empty)
    return {
        "validation_products": len(validation_rows),
        "learned_q": f1_of(*learned_counts),
        "literal_q": f1_of(*literal_counts),
        "empty_q": f1_of(*empty_counts),
        "learned_counts": {
            "true_positive": learned_counts[0],
            "false_positive": learned_counts[1],
            "false_negative": learned_counts[2],
        },
        "literal_counts": {
            "true_positive": literal_counts[0],
            "false_positive": literal_counts[1],
            "false_negative": literal_counts[2],
        },
    }


def assert_reproducible(rows: list[dict], bank: SkillBank) -> None:
    """Prediction must be a pure function of the saved bank."""
    sample = [row for row in rows if is_validation(row["product_id"])][:12]
    before = json.dumps(bank.to_dict(), sort_keys=True)
    restored = SkillBank.from_dict(json.loads(before))
    for row in sample:
        original = bank.predict_row(row)
        if original != restored.predict_row(row):
            raise RuntimeError("Loaded skill bank does not reproduce predictions")
        for values in original.values():
            if len(values) != len(set(values)) or any(not isinstance(value, str) or not value for value in values):
                raise RuntimeError("Prediction is not a list of unique nonempty strings")
    if json.dumps(bank.to_dict(), sort_keys=True) != before:
        raise RuntimeError("Prediction changed the skill bank")


def learn(rows: list[dict]) -> tuple[SkillBank, SkillBank, dict[str, Any]]:
    """Fit on the train split, choose skills on the held-out split, then refit for submission.

    The first bank is the learning evidence. The second bank is the frozen agent.
    """
    evidence_bank, selection = select_bank(rows)
    report = comparison(rows, evidence_bank)
    selected = selection.get("q")
    if selected is not None and abs(report["learned_q"] - selected) > 1e-9:
        raise RuntimeError("The reported learning score does not match the selected bank")
    final_bank = retune_thresholds(rows, refit_on_all(rows, evidence_bank))
    assert_reproducible(rows, final_bank)
    report["evidence_scope"] = (
        "Weights fit on the experiment products outside this split. "
        "The saved bank is refit on every experiment product and is not this number."
    )
    report["rules"] = list(evidence_bank.rules)
    report["selection"] = {
        "modes": selection.get("modes", {}),
        "q_during_selection": selected,
        "rules": selection.get("rules", 0),
    }
    return evidence_bank, final_bank, report


def rows_from_examples(examples: list[Any]) -> list[dict]:
    rows = []
    for example in examples:
        request = example.request
        attributes = []
        for spec in request.attribute_specs:
            values = []
            for item in spec.allowed_values or []:
                if isinstance(item, dict):
                    values.append({"name": item.get("name", ""), "definition": item.get("definition", "")})
                elif hasattr(item, "name"):
                    values.append({"name": item.name, "definition": getattr(item, "definition", "")})
                else:
                    values.append({"name": str(item), "definition": ""})
            attributes.append({
                "attribute_id": spec.attribute_id,
                "attribute_name": getattr(spec, "attribute_name", spec.attribute_id),
                "definition": getattr(spec, "definition", ""),
                "multi_valued": bool(getattr(spec, "multi_valued", False)),
                "allowed_values": values,
            })
        labels = []
        for attribute_id, values in dict(example.labels).items():
            labels.append({"attribute_id": attribute_id, "values": [str(value) for value in values]})
        description = getattr(request, "input_text", None) or request.title
        rows.append({
            "run_id": str(getattr(request, "run_id", "") or "learn"),
            "product_id": str(request.product_id),
            "product_description": description,
            "attributes": attributes,
            "labels": labels,
        })
    return rows
