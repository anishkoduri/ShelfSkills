"""Automated attribute skills for exact-match extraction.

The learner updates skill weights and which readings to keep. It does not update foundation-model parameters.
Pack and threshold choices are made on a product-disjoint experiment split.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import re
import unicodedata
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

    def fit(
        self,
        docs: list[set[str]],
        targets: list[set[str]],
        allowed: list[str],
        multi: bool,
        record_checkpoints: bool = False,
        epochs: int | None = None,
    ) -> None:
        self.multi = multi
        self.checkpoints = []
        classes = [name for name in allowed if any(name in target for target in targets)]
        for target in targets:
            for name in target:
                if name not in classes:
                    classes.append(name)
        self.classes = classes
        if not classes:
            return
        if multi:
            self._fit_binary(docs, targets, record_checkpoints, epochs)
        else:
            self._fit_single(docs, targets, record_checkpoints, epochs)

    def _fit_single(
        self,
        docs: list[set[str]],
        targets: list[set[str]],
        record_checkpoints: bool = False,
        epochs: int | None = None,
    ) -> None:
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
        for _epoch in range(12 if epochs is None else epochs):
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
            if record_checkpoints and seen:
                self.checkpoints.append(self._averaged_single(weights, totals, bias, bias_total, seen))
        self.weights, self.bias = self._averaged_single(weights, totals, bias, bias_total, seen)

    def _averaged_single(self, weights, totals, bias, bias_total, seen) -> tuple[dict, dict]:
        averaged_weights = {}
        averaged_bias = {}
        for label in self.classes:
            table = weights[label]
            averaged_weights[label] = {
                feat: value - totals[label][feat] / seen
                for feat, value in table.items()
                if seen and abs(value - totals[label][feat] / seen) >= 0.05
            }
            averaged_bias[label] = bias[label] - (bias_total[label] / seen if seen else 0.0)
        return averaged_weights, averaged_bias

    def _fit_binary(
        self,
        docs: list[set[str]],
        targets: list[set[str]],
        record_checkpoints: bool = False,
        epochs: int | None = None,
    ) -> None:
        docs = _repeated_features(docs)
        self.weights = {}
        self.bias = {}
        per_label: dict[str, list[tuple[dict[str, float], float]]] = {}
        for label in self.classes:
            positive = [doc for doc, target in zip(docs, targets) if label in target]
            negative = [doc for doc, target in zip(docs, targets) if label not in target]
            if not positive:
                self.weights[label] = {}
                self.bias[label] = -1.0
                if record_checkpoints:
                    per_label[label] = [({}, -1.0) for _epoch in range(10 if epochs is None else epochs)]
                continue
            rng = random.Random(11 + len(label))
            train = [(doc, 1) for doc in positive] + [(doc, 0) for doc in negative]
            weights: dict[str, float] = {}
            totals: dict[str, float] = {}
            bias = 0.0
            bias_total = 0.0
            seen = 0
            epoch_snapshots: list[tuple[dict[str, float], float]] = []
            for _epoch in range(10 if epochs is None else epochs):
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
                if record_checkpoints and seen:
                    epoch_snapshots.append(self._averaged_binary(weights, totals, bias, bias_total, seen))
            frozen = self._averaged_binary(weights, totals, bias, bias_total, seen)
            self.weights[label] = frozen[0]
            self.bias[label] = frozen[1]
            if record_checkpoints:
                per_label[label] = epoch_snapshots
        if record_checkpoints and per_label:
            width = max(len(snapshots) for snapshots in per_label.values())
            for epoch_index in range(width):
                averaged_weights = {}
                averaged_bias = {}
                for label, snapshots in per_label.items():
                    table, bias_value = snapshots[min(epoch_index, len(snapshots) - 1)]
                    averaged_weights[label] = table
                    averaged_bias[label] = bias_value
                self.checkpoints.append((averaged_weights, averaged_bias))

    def _averaged_binary(self, weights, totals, bias, bias_total, seen) -> tuple[dict[str, float], float]:
        averaged = {
            feat: value - totals[feat] / seen
            for feat, value in weights.items()
            if seen and abs(value - totals[feat] / seen) >= 0.05
        }
        return averaged, bias - (bias_total / seen if seen else 0.0)

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


def _title_text(description: str) -> str:
    """Product name only. Marketing copy and ingredients are a different reading."""
    parsed = parse_product(description)
    if not isinstance(parsed, dict):
        return " ".join(tokens(description))
    return " ".join(tokens(str(parsed.get("dis_name") or "")) + tokens(str(parsed.get("prod_desc") or "")))


def _apply_title(
    chosen: list[str],
    rules: list[dict[str, str]],
    attribute_id: str,
    title_text: str,
    allowed: list[str],
    multi: bool,
) -> list[str]:
    """Use a value written in the product name when that reading was reliable in training."""
    if multi or not rules or not title_text:
        return chosen
    allowed_set = set(allowed)
    matched = None
    for rule in rules:
        if rule["attribute_id"] != attribute_id or rule["value"] not in allowed_set:
            continue
        if rule["phrase"] in title_text:
            matched = rule["value"]
    return [matched] if matched else chosen


_DIET_SWEETENERS = ("aspartame", "sucralose", "acesulfame", "saccharin", "stevia", "monk")


def _soda_diet_cue(description: str) -> bool:
    """Diet in the product name, or an artificial or plant sweetener in the ingredients."""
    parsed = parse_product(description)
    if not isinstance(parsed, dict):
        return False
    name = set(tokens(str(parsed.get("dis_name") or "")) + tokens(str(parsed.get("prod_desc") or "")))
    if "diet" in name:
        return True
    ingredients = set(tokens(str(parsed.get("ingredients") or "")))
    return any(token in ingredients for token in _DIET_SWEETENERS)


_SEED_NEXT = {"loaf", "thick", "sandwich", "lf", "grain", "harvest", "grn"}
_SEED_INGREDIENTS = (("flax", "seeds"), ("caraway", "seeds"), ("chia", "seeds"))
_MIX_FRUIT = {
    "strawberry", "strawberries", "blueberry", "blueberries", "raspberry", "raspberries",
    "peach", "peaches", "mango", "apple", "apples", "cherry", "cherries", "banana", "bananas",
    "apricot", "apricots", "pineapple", "cranberry", "cranberries", "fruit",
}
_MIX_PIECE = {"piece", "pieces", "bit", "bits", "chunk", "chunks"}
_FRUIT_PLURAL = {
    "strawberries", "blueberries", "raspberries", "blackberries", "peaches", "cherries",
    "bananas", "apples", "apricots", "mangoes", "mangos", "pineapples", "pears",
}
_FRUIT_SINGULAR = {
    "strawberry", "blueberry", "raspberry", "blackberry", "peach", "cherry", "banana",
    "apple", "apricot", "mango", "pineapple", "pear", "fruit",
}
_PIECE_WORD = {"pieces", "piece", "diced", "chunks", "chunk", "slices", "sliced", "compote", "preparation"}
_HEAD_BLOCK = {"puree", "purees", "juice", "juices", "flavor", "flavors", "flavored", "flavour", "nectar", "concentrate", "concentrated", "extract", "pectin"}


def _parsed(description: str) -> dict | None:
    parsed = parse_product(description)
    return parsed if isinstance(parsed, dict) else None


def _name_token_lists(parsed: dict) -> list[list[str]]:
    return [tokens(parsed.get(field) or "") for field in ("dis_name", "prod_desc")]


def _has_phrase(token_lists: list[list[str]], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    for toks in token_lists:
        if any(tuple(toks[index:index + width]) == phrase for index in range(len(toks) - width + 1)):
            return True
    return False


def _bakery_seed_cue(description: str) -> bool:
    """Seeded loaf names, or flax, caraway, and chia seeds in the ingredient list."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    for toks in _name_token_lists(parsed):
        if any(left == "seeded" and right in _SEED_NEXT for left, right in zip(toks, toks[1:])):
            return True
    pairs = set(zip(tokens(parsed.get("ingredients") or ""), tokens(parsed.get("ingredients") or "")[1:]))
    return any(pair in pairs for pair in _SEED_INGREDIENTS)


def _bakery_garlic_cue(description: str) -> bool:
    """Garlic written in the product name. Ingredient powder and dried garlic do not count."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    name = {token for toks in _name_token_lists(parsed) for token in toks}
    return "garlic" in name


def _mix_fruit_piece_cue(description: str) -> bool:
    """Fruit pieces, dried fruit, or fruit on the bottom in the short product text."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    for field in ("ingredients", "dis_name", "prod_desc"):
        toks = tokens(parsed.get(field) or "")
        if _has_phrase([toks], ("on", "the", "bottom")) and any(token in _MIX_FRUIT for token in toks):
            return True
        for left, right in zip(toks, toks[1:]):
            if left in _MIX_FRUIT and right in _MIX_PIECE:
                return True
            if left == "dried" and right in _MIX_FRUIT:
                return True
    return False


def _ingredient_heads(ingredients: str) -> list[list[str]]:
    text = html.unescape(str(ingredients or ""))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    parts: list[str] = []
    depth = 0
    buffer: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
            buffer.append(char)
        elif char == ")":
            depth = max(0, depth - 1)
            buffer.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
    if buffer:
        parts.append("".join(buffer))
    return [tokens(part.split("(", 1)[0]) for part in parts]


def _top_fruit_piece_cue(description: str) -> bool:
    """A top-level fruit piece or plural fruit, not a puree, flavor, or bottom layer."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    if _flavor_bottom_cue(description):
        return False
    short = _name_token_lists(parsed) + [tokens(parsed.get("ingredients") or "")]
    if _has_phrase(short, ("on", "the", "bottom")):
        return False
    for head in _ingredient_heads(str(parsed.get("ingredients") or "")):
        if not head or any(token in _HEAD_BLOCK for token in head):
            continue
        if any(token in _FRUIT_PLURAL for token in head):
            return True
        if any(token in _FRUIT_SINGULAR for token in head) and any(token in _PIECE_WORD for token in head):
            return True
    return False


def _blob(parsed: dict, *fields: str) -> str:
    parts = [html.unescape(str(parsed.get(field) or "")).lower().replace("\xa0", " ") for field in fields]
    return " ".join(parts)


def _name_token(description: str, token: str) -> bool:
    parsed = _parsed(description)
    if parsed is None:
        return False
    return any(token in toks for toks in _name_token_lists(parsed))


_SODA_FRUIT = {
    "mango", "mngo", "raspberry", "berry", "peach", "pineapple", "guava", "strawberry",
    "blueberry", "banana", "grape", "apple", "apricot", "cranberry", "pomegranate",
    "watermelon", "pnapl",
}
_SODA_FRUIT_BLOCK = {"spice", "spiced", "sage", "herb", "herbal", "red", "crimson", "crmsn"}


def _soda_fruit_cue(description: str) -> bool:
    """A fruit in the soda name. Ginger ale, spice, and red soda are not fruit soda."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    toks = [token for group in _name_token_lists(parsed) for token in group]
    names = set(toks)
    if not names & _SODA_FRUIT:
        return False
    if names & _SODA_FRUIT_BLOCK:
        return False
    return not any(left in {"ginger", "gngr", "gingery"} and right == "ale" for left, right in zip(toks, toks[1:]))


def _loaf_sourdough_cue(description: str) -> bool:
    """Sourdough in the product name. The ingredient line is a different bread."""
    return _name_token(description, "sourdough")


def _roll_brioche_cue(description: str) -> bool:
    return _name_token(description, "brioche")


def _bakery_oat_cue(description: str) -> bool:
    """The token oat in the product name. Oatmeal is not that token."""
    return _name_token(description, "oat")


def _yogurt_drinkable_cue(description: str) -> bool:
    """The word drinkable in the product name."""
    return _name_token(description, "drinkable")


def _yogurt_french_cue(description: str) -> bool:
    """The word french in the product name."""
    return _name_token(description, "french")


def _sweetener_artificial_cue(description: str) -> bool:
    """A named artificial sweetener in the ingredients, with no stevia or monk fruit."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    ingredients = set(tokens(parsed.get("ingredients") or ""))
    if not (_ARTIFICIAL_SWEETENERS & ingredients):
        return False
    name = {token for toks in _name_token_lists(parsed) for token in toks}
    return not (_PLANT_SWEETENERS & (ingredients | name))


def _cereal_name_wheat_cue(description: str) -> bool:
    """The shelf abbreviation wht, unless the name also says rice."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    name = {token for toks in _name_token_lists(parsed) for token in toks}
    return "wht" in name and "rice" not in name


def _cereal_name_multiple_cue(description: str) -> bool:
    """The name says multigrain, including the shelf abbreviation multgr."""
    return _name_token(description, "multigrain") or _name_token(description, "multgr")


def _yogurt_probiotic_cue(description: str) -> bool:
    """The word probiotic. Culture names alone are a different product."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    for field in ("dis_name", "prod_desc", "ingredients", "long_desc"):
        if any(token.startswith("probiotic") for token in tokens(parsed.get(field) or "")):
            return True
    return False


_CEREAL_ALIAS = {
    "oat": "oat",
    "oats": "oat",
    "oatmeal": "oat",
    "corn": "corn",
    "cornmeal": "corn",
    "wheat": "wheat",
    "rice": "rice",
    "flax": "flax",
    "flaxseed": "flax",
    "quinoa": "quinoa",
    "chia": "chia",
    "hemp": "hemp",
    "buckwheat": "buckwheat",
    "bran": "bran",
}
_CEREAL_SKIP_NEXT = {"syrup", "starch", "fiber", "fibre", "malt", "dextrin", "maltodextrin", "bran"}
_CEREAL_LABEL = {"oat": "Oat", "corn": "Corn", "wheat": "Wheat", "rice": "Rice"}
_CEREAL_VALUES = (
    "Multiple", "Oat", "Corn", "Wheat", "Rice", "Specialty", "Not Applicable",
    "Flaxseed", "Bran", "Quinoa", "Chia", "Hemp", "Buckwheat",
)
_ARTIFICIAL_SWEETENERS = {"aspartame", "sucralose", "acesulfame", "saccharin"}
_PLANT_SWEETENERS = {"stevia", "monk"}


def _cereal_only_label(description: str) -> str:
    """One real grain in the ingredient head. Syrup, starch, fiber, and the 2% tail do not count."""
    parsed = _parsed(description)
    if parsed is None:
        return ""
    ingredients = tokens(parsed.get("ingredients") or "")
    if not ingredients:
        return ""
    cut = len(ingredients)
    for index, token in enumerate(ingredients):
        nxt = ingredients[index + 1] if index + 1 < len(ingredients) else ""
        if (token == "contains" and nxt.isdigit()) or (token == "less" and nxt == "than") or (token == "or" and nxt == "less"):
            cut = index
            break
    found: list[str] = []
    head = ingredients[:cut]
    for index, token in enumerate(head):
        if token == "cornstarch":
            continue
        grain = _CEREAL_ALIAS.get(token)
        if not grain:
            continue
        nxt = head[index + 1] if index + 1 < len(head) else ""
        if nxt in _CEREAL_SKIP_NEXT:
            continue
        if grain not in found:
            found.append(grain)
    if len(found) != 1:
        return ""
    return _CEREAL_LABEL.get(found[0], "")


_BOTTOM_PHRASES = (
    "on the bottom",
    "bottom layer",
    "fruit layer",
    "tucked beneath",
    "beneath the surface",
    "waiting beneath",
    "compote tucked",
)
_GRANOLA_WORDS = ("granola", "grnla", "grnl", "crumble", "cluster", "clusters")


def _flavor_bottom_cue(description: str) -> bool:
    """A fruit layer under the yogurt, not fruit mixed through it."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    text = _blob(parsed, "dis_name", "prod_desc", "ingredients", "long_desc")
    if any(phrase in text for phrase in _BOTTOM_PHRASES):
        return True
    # Hyphens split, so fruit-on-the-bottom is the same phrase.
    toks = tokens(text)
    return any(tuple(toks[index:index + 3]) == ("on", "the", "bottom") for index in range(len(toks) - 2))


def _soda_listed_caffeine_cue(description: str) -> bool:
    """Caffeine in the ingredient list, and the list does not say caffeine free."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    ingredients = _blob(parsed, "ingredients")
    if "caffeine free" in ingredients or "caffeine-free" in ingredients:
        return False
    return "caffeine" in tokens(ingredients)


def _soda_zs_cue(description: str) -> bool:
    """The shelf-tag token zs. The word zero by itself is a different soda."""
    return _name_token(description, "zs")


def _mix_drop_granola_cue(description: str) -> bool:
    """No granola word in the name, ingredients, or description."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    text = _blob(parsed, "dis_name", "prod_desc", "ingredients", "long_desc")
    return not any(word in tokens(text) or word in text for word in _GRANOLA_WORDS)


def _yogurt_fatfree_cue(description: str) -> bool:
    """Nonfat or fat-free in the product name. Nonfat milk in the ingredients is a different product."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    text = " ".join(token for toks in _name_token_lists(parsed) for token in toks)
    return any(phrase in text for phrase in ("nonfat", "non fat", "fat free", "fatfree"))


def _yogurt_lowfat_cue(description: str) -> bool:
    """Low or reduced fat, or 1% or 2% milk. Whole milk and '1% or less' do not count."""
    parsed = _parsed(description)
    if parsed is None:
        return False
    name = _blob(parsed, "dis_name", "prod_desc")
    ingredients = _blob(parsed, "ingredients")
    if any(phrase in name for phrase in ("lowfat", "low-fat", "low fat", "reduced fat", "reduced-fat", "2 pcnt")):
        return True
    if re.search(r"\b[12]\s*%", name):
        return True
    if any(phrase in ingredients for phrase in ("reduced fat milk", "reduced-fat milk", "lowfat milk", "low-fat milk", "low fat milk")):
        return True
    return any("milk" in match.group(1) for match in re.finditer(r"\b[12]\s*%\s*(?!or\b)(.{0,24})", ingredients))


def _rule_cues(description: str) -> set[str]:
    found: set[str] = set()
    if _soda_diet_cue(description):
        found.add("read:soda_diet_or_sweetener")
    if _bakery_seed_cue(description):
        found.add("read:bakery_seed")
    if _bakery_garlic_cue(description):
        found.add("read:bakery_garlic")
    if _mix_fruit_piece_cue(description):
        found.add("read:mix_fruit_piece")
    if _top_fruit_piece_cue(description):
        found.add("read:top_fruit_piece")
    if _yogurt_drinkable_cue(description):
        found.add("read:yogurt_drinkable")
    if _yogurt_french_cue(description):
        found.add("read:yogurt_french")
    if _sweetener_artificial_cue(description):
        found.add("read:sweetener_artificial")
    if _cereal_name_wheat_cue(description):
        found.add("read:cereal_name_wheat")
    if _cereal_name_multiple_cue(description):
        found.add("read:cereal_name_multiple")
    if _yogurt_probiotic_cue(description):
        found.add("read:yogurt_probiotic")
    if _yogurt_fatfree_cue(description):
        found.add("read:yogurt_fatfree")
    if _yogurt_lowfat_cue(description):
        found.add("read:yogurt_lowfat")
    cereal_only = _cereal_only_label(description)
    if cereal_only:
        found.add("read:cereal_only_" + cereal_only.lower())
    if _loaf_sourdough_cue(description):
        found.add("read:loaf_sourdough")
    if _roll_brioche_cue(description):
        found.add("read:roll_brioche")
    if _bakery_oat_cue(description):
        found.add("read:bakery_oat")
    if _soda_fruit_cue(description):
        found.add("read:soda_fruit_name")
    if _flavor_bottom_cue(description):
        found.add("read:flavor_bottom")
    if _soda_listed_caffeine_cue(description):
        found.add("read:soda_listed_caffeine")
    if _soda_zs_cue(description):
        found.add("read:soda_zs")
    if _mix_drop_granola_cue(description):
        found.add("read:mix_no_granola")
    return found


def _apply_rules(chosen: list[str], rules: list[dict[str, str]], attribute_id: str, document: set[str], multi: bool) -> list[str]:
    result = list(chosen)
    for rule in rules:
        if rule["attribute_id"] != attribute_id or rule["feature"] not in document:
            continue
        when = rule.get("when", "")
        if when and when not in result:
            continue
        drops = []
        if rule.get("drop"):
            drops.append(rule["drop"])
        extra = rule.get("drops") or []
        drops.extend([extra] if isinstance(extra, str) else extra)
        if drops:
            blocked = set(drops)
            result = [item for item in result if item not in blocked]
        value = rule.get("value", "")
        if not value:
            continue
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
    rule_document = set(documents.get("cue") or documents.get("long") or documents.get("short") or set())
    rule_document.update(_rule_cues(row["product_description"]))
    title_text = _title_text(row["product_description"]) if bank.title_rules else ""
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
        prediction[aid] = _clean(
            _apply_title(
                _apply_rules(chosen, bank.rules, aid, rule_document, multi),
                bank.title_rules,
                aid,
                title_text,
                names,
                multi,
            )
        )
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
    title_rules: list[dict[str, str]] = field(default_factory=list)
    epoch_counts: dict[str, int] = field(default_factory=dict)

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
            "title_rules": self.title_rules,
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
        bank.title_rules = [dict(rule) for rule in payload.get("title_rules", [])]
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


def fit_skills(
    prepared: list[dict],
    catalog: dict[str, dict],
    train_only: bool,
    epoch_counts: dict[str, int] | None = None,
) -> dict[str, Skill]:
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
            epochs=epoch_counts.get(aid) if epoch_counts else None,
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


def choose_epochs(rows: list[dict], bank: SkillBank) -> dict[str, int]:
    """Keep an earlier training pass when that pass is more exact on the validation split.

    The pass is chosen on experiment products the weights did not see. A pass is
    kept only when the exact-match score rises.
    """
    validation = [(row, _documents(row, bank)) for row in rows if is_validation(row["product_id"])]
    validation_rows = [row for row, _documents_for_row in validation]

    def predict_all() -> list[dict]:
        return [
            _predict_with(row, bank, documents, bank.margins, bank.thresholds, bank.force_tops)
            for row, documents in validation
        ]

    base = predict_all()
    base_q = f1_of(*micro_counts(validation_rows, base))
    prepared: dict[str, list[dict]] = {}
    options = []
    for aid, info in bank.catalog.items():
        mode = bank.modes.get(aid, "long")
        if mode == "vote" or aid not in bank.skills:
            continue
        if mode not in prepared:
            prepared[mode] = prepare(rows, mode)
        subset = [
            item
            for item in prepared[mode]
            if not item["validation"] and any(spec["attribute_id"] == aid for spec in item["row"]["attributes"])
        ]
        trial_skill = Skill()
        trial_skill.fit(
            [item["feats"] for item in subset],
            [item["labels"].get(aid, set()) for item in subset],
            info["allowed"],
            info["multi"],
            record_checkpoints=True,
        )
        saved = bank.skills[aid]
        bank.skills[aid] = trial_skill
        reproduced = all(
            trial.get(aid) == original.get(aid)
            for trial, original in zip(predict_all(), base)
        )
        best: tuple[float, int, dict, dict] | None = None
        if reproduced:
            for epoch, (weights, bias) in enumerate(trial_skill.checkpoints, start=1):
                trial_skill.weights = weights
                trial_skill.bias = bias
                gained = f1_of(*micro_counts(validation_rows, predict_all())) - base_q
                if best is None or gained > best[0]:
                    best = (gained, epoch, weights, bias)
        bank.skills[aid] = saved
        if best is not None and best[0] > 1e-12:
            options.append((best[0], aid, best[1], best[2], best[3]))
    options.sort(reverse=True)
    current_q = base_q
    chosen: dict[str, int] = {}
    for _gain, aid, epoch, weights, bias in options:
        skill = bank.skills[aid]
        saved_weights, saved_bias = skill.weights, skill.bias
        skill.weights = weights
        skill.bias = bias
        trial_q = f1_of(*micro_counts(validation_rows, predict_all()))
        if trial_q > current_q + 1e-12:
            current_q = trial_q
            chosen[aid] = epoch
            skill.weights = {label: dict(table) for label, table in weights.items()}
            skill.bias = dict(bias)
        else:
            skill.weights = saved_weights
            skill.bias = saved_bias
    bank.epoch_counts = chosen
    return chosen


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
    bank.epoch_counts = choose_epochs(rows, bank)
    bank.title_rules = adapt_title_rules(rows, bank)
    adapt_soda_diet(rows, bank)
    adapt_candidate_rules(rows, bank)
    true_positive, false_positive, false_negative = _evidence_counts(rows, bank)
    return bank, {
        "q": f1_of(true_positive, false_positive, false_negative),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "modes": dict(modes),
        "rules": len(bank.rules),
        "title_rules": len(bank.title_rules),
        "epochs": dict(bank.epoch_counts),
        "trained_on": "experiment_train_split",
    }


def _evidence_counts(rows: list[dict], bank: SkillBank) -> tuple[int, int, int]:
    validation_rows = [row for row in rows if is_validation(row["product_id"])]
    predictions = [bank.predict_row(row) for row in validation_rows]
    return micro_counts(validation_rows, predictions)


def adapt_from_experiment(rows: list[dict], bank: SkillBank) -> list[dict[str, str]]:
    """Learn high-precision reader rules from mistakes on the experiment train split."""
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


def adapt_soda_diet(rows: list[dict], bank: SkillBank) -> None:
    """Append Diet and drop Cane when the name or sweetener cue was reliable in training.

    The rule is added only when validation exact-match rises. Zero-sugar wording is not the cue.
    """
    support = 0
    correct = 0
    for row in rows:
        if is_validation(row["product_id"]):
            continue
        if not any(spec["attribute_id"] == "A100196" for spec in row["attributes"]):
            continue
        if not _soda_diet_cue(row["product_description"]):
            continue
        support += 1
        correct += "Diet" in labels_of(row).get("A100196", set())
    if support < 6 or correct / support < 0.9:
        return
    rule = {
        "attribute_id": "A100196",
        "feature": "read:soda_diet_or_sweetener",
        "value": "Diet",
        "drop": "Cane & Real Sugar",
    }
    validation_cache = [
        (row, _documents(row, bank))
        for row in rows
        if is_validation(row["product_id"])
    ]

    def quality(rules: list[dict[str, str]]) -> float:
        bank.rules = rules
        validation_rows = [row for row, _documents_for_row in validation_cache]
        predictions = [
            _predict_with(row, bank, documents, bank.margins, bank.thresholds, bank.force_tops)
            for row, documents in validation_cache
        ]
        return f1_of(*micro_counts(validation_rows, predictions))

    base = list(bank.rules)
    base_score = quality(base)
    trial = base + [rule]
    if quality(trial) > base_score:
        bank.rules = trial
    else:
        bank.rules = base


def _train_precision(rows: list[dict], attribute_id: str, cue, label: str) -> tuple[int, int]:
    support = 0
    correct = 0
    for row in rows:
        if is_validation(row["product_id"]):
            continue
        if not any(spec["attribute_id"] == attribute_id for spec in row["attributes"]):
            continue
        if not cue(row["product_description"]):
            continue
        support += 1
        correct += label in labels_of(row).get(attribute_id, set())
    return support, correct


def adapt_candidate_rules(rows: list[dict], bank: SkillBank) -> None:
    """Try candidate readings on the experiment split.

    A bundle is kept when validation micro-F1 rises. An exact training reading is
    kept when validation micro-F1 does not fall.
    """
    bundles: list[list[dict[str, str]]] = []
    seed_support, seed_correct = _train_precision(rows, "A101571", _bakery_seed_cue, "Seeds")
    garlic_support, garlic_correct = _train_precision(rows, "A101571", _bakery_garlic_cue, "Garlic")
    seed_rule = {"attribute_id": "A101571", "feature": "read:bakery_seed", "value": "Seeds"}
    garlic_rule = {"attribute_id": "A101571", "feature": "read:bakery_garlic", "value": "Garlic"}
    pair: list[dict[str, str]] = []
    if seed_support >= 4 and seed_correct / seed_support >= 0.9:
        pair.append(seed_rule)
    if garlic_support >= 4 and garlic_correct / garlic_support >= 0.9:
        pair.append(garlic_rule)
    if len(pair) == 2:
        bundles.append(pair)
    bundles.extend([[rule] for rule in pair])
    bundles.append([{
        "attribute_id": "A101643",
        "feature": "read:top_fruit_piece",
        "value": "Fruit Mixed In",
        "when": "Fruit Flavored",
    }])
    mix_support, mix_correct = _train_precision(rows, "A101644", _mix_fruit_piece_cue, "Fruit")
    if mix_support >= 5 and mix_correct == mix_support:
        bundles.append([{
            "attribute_id": "A101644",
            "feature": "read:mix_fruit_piece",
            "value": "Fruit",
        }])
    probiotic_support, probiotic_correct = _train_precision(rows, "A100334", _yogurt_probiotic_cue, "Probiotic & Functional")
    if probiotic_support >= 6 and probiotic_correct == probiotic_support:
        bundles.append([{
            "attribute_id": "A100334",
            "feature": "read:yogurt_probiotic",
            "value": "Probiotic & Functional",
        }])
    lowfat_rule = {
        "attribute_id": "A102531",
        "feature": "read:yogurt_lowfat",
        "value": "Low & Reduced Fat",
    }
    lowfat_support, lowfat_correct = _train_precision(rows, "A102531", _yogurt_lowfat_cue, "Low & Reduced Fat")
    lowfat_ready = lowfat_support >= 6 and lowfat_correct == lowfat_support
    validation_cache = [
        (row, _documents(row, bank))
        for row in rows
        if is_validation(row["product_id"])
    ]

    def quality(rules: list[dict[str, str]]) -> float:
        bank.rules = rules
        validation_rows = [row for row, _documents_for_row in validation_cache]
        predictions = [
            _predict_with(row, bank, documents, bank.margins, bank.thresholds, bank.force_tops)
            for row, documents in validation_cache
        ]
        return f1_of(*micro_counts(validation_rows, predictions))

    kept = list(bank.rules)
    best = quality(kept)
    for _pass in range(2):
        for bundle in bundles:
            extra = [rule for rule in bundle if rule not in kept]
            if not extra:
                continue
            trial = kept + extra
            score = quality(trial)
            if score > best:
                kept = trial
                best = score
    steady = []
    if lowfat_ready:
        steady.append(lowfat_rule)
    for attribute_id, cue, feature, value, minimum in (
        ("A102220", _loaf_sourdough_cue, "read:loaf_sourdough", "Sourdough", 0),
        ("A102246", _roll_brioche_cue, "read:roll_brioche", "Brioche", 2),
        ("A101571", _bakery_oat_cue, "read:bakery_oat", "Oats", 4),
        ("A100190", _soda_fruit_cue, "read:soda_fruit_name", "Fruit", 6),
        ("A101643", _flavor_bottom_cue, "read:flavor_bottom", "Fruit on the Bottom", 6),
        ("A100196", _soda_zs_cue, "read:soda_zs", "Diet", 2),
        ("A100334", _yogurt_drinkable_cue, "read:yogurt_drinkable", "Drinkable", 4),
        ("A100334", _yogurt_french_cue, "read:yogurt_french", "French", 6),
        ("A102531", _yogurt_fatfree_cue, "read:yogurt_fatfree", "Fat Free", 6),
    ):
        support, correct = _train_precision(rows, attribute_id, cue, value)
        if support >= minimum and correct == support:
            steady.append({
                "attribute_id": attribute_id,
                "feature": feature,
                "value": value,
            })
    for label in ("Oat", "Corn", "Wheat", "Rice"):
        support = correct = 0
        for row in rows:
            if is_validation(row["product_id"]):
                continue
            if not any(spec["attribute_id"] == "A101614" for spec in row["attributes"]):
                continue
            if _cereal_only_label(row["product_description"]) != label:
                continue
            support += 1
            correct += labels_of(row).get("A101614", set()) == {label}
        if support >= 4 and correct == support:
            steady.append({
                "attribute_id": "A101614",
                "feature": "read:cereal_only_" + label.lower(),
                "value": label,
                "drop": "Multiple",
            })
    for attribute_id, cue, feature, label in (
        ("A100196", _soda_listed_caffeine_cue, "read:soda_listed_caffeine", "Caffeine Free"),
        ("A101644", _mix_drop_granola_cue, "read:mix_no_granola", "Granola"),
    ):
        support = correct = 0
        for row in rows:
            if is_validation(row["product_id"]):
                continue
            if not any(spec["attribute_id"] == attribute_id for spec in row["attributes"]):
                continue
            if not cue(row["product_description"]):
                continue
            support += 1
            correct += label not in labels_of(row).get(attribute_id, set())
        if support >= 6 and correct == support:
            steady.append({
                "attribute_id": attribute_id,
                "feature": feature,
                "drop": label,
            })
    exact_rules = (
        ("A900201", _sweetener_artificial_cue, "read:sweetener_artificial", "Artificial Sweetener", 6, None),
        ("A101614", _cereal_name_wheat_cue, "read:cereal_name_wheat", "Wheat", 2, [value for value in _CEREAL_VALUES if value != "Wheat"]),
        ("A101614", _cereal_name_multiple_cue, "read:cereal_name_multiple", "Multiple", 1, [value for value in _CEREAL_VALUES if value != "Multiple"]),
    )
    for attribute_id, cue, feature, value, minimum, drops in exact_rules:
        support = correct = 0
        for row in rows:
            if is_validation(row["product_id"]):
                continue
            if not any(spec["attribute_id"] == attribute_id for spec in row["attributes"]):
                continue
            if not cue(row["product_description"]):
                continue
            support += 1
            correct += labels_of(row).get(attribute_id, set()) == {value}
        if support >= minimum and correct == support:
            rule = {
                "attribute_id": attribute_id,
                "feature": feature,
                "value": value,
            }
            if drops:
                rule["drops"] = drops
            steady.append(rule)
    for rule in steady:
        if rule in kept:
            continue
        trial = kept + [rule]
        score = quality(trial)
        if score >= best:
            kept = trial
            best = score
    bank.rules = kept


def adapt_title_rules(rows: list[dict], bank: SkillBank) -> list[dict[str, str]]:
    """Keep a name phrase only when that attribute's phrases raise the validation score.

    The phrase is learned from the train split. An attribute is added only when the
    validation products get more exact matches.
    """
    counts: dict[tuple[str, str, str], list[int]] = {}
    for row in rows:
        if is_validation(row["product_id"]):
            continue
        text = _title_text(row["product_description"])
        labels = labels_of(row)
        for spec in row["attributes"]:
            aid = spec["attribute_id"]
            if bank.catalog.get(aid, {}).get("multi"):
                continue
            gold = labels.get(aid, set())
            for name in allowed_names(spec):
                phrase = " ".join(tokens(name))
                if len(phrase) < 4 or phrase not in text:
                    continue
                hit = counts.setdefault((aid, phrase, name), [0, 0])
                hit[0] += 1
                hit[1] += name in gold
    grouped: dict[str, list[dict[str, str]]] = {}
    for (aid, phrase, name), (seen, correct) in counts.items():
        if seen < 6 or not seen or correct / seen < 0.9:
            continue
        grouped.setdefault(aid, []).append({"attribute_id": aid, "phrase": phrase, "value": name})

    validation_cache = [
        (row, _documents(row, bank))
        for row in rows
        if is_validation(row["product_id"])
    ]

    def quality(title_rules: list[dict[str, str]]) -> float:
        bank.title_rules = title_rules
        validation_rows = [row for row, _documents_for_row in validation_cache]
        predictions = [
            _predict_with(row, bank, documents, bank.margins, bank.thresholds, bank.force_tops)
            for row, documents in validation_cache
        ]
        return f1_of(*micro_counts(validation_rows, predictions))

    kept: list[dict[str, str]] = []
    best = quality([])
    for _pass in range(2):
        for aid in list(grouped):
            extra = [rule for rule in grouped[aid] if rule not in kept]
            if not extra:
                continue
            trial = kept + extra
            score = quality(trial)
            if score > best:
                kept = trial
                best = score
    bank.title_rules = kept
    return kept


def refit_on_all(rows: list[dict], bank: SkillBank) -> SkillBank:
    """Keep the chosen packs and thresholds, then relearn weights on every experiment row."""
    catalog = bank.catalog
    fitted: dict[str, dict[str, Skill]] = {}
    for mode in ("short", "long", "cue"):
        prepared = prepare(rows, mode)
        fitted[mode] = fit_skills(prepared, catalog, train_only=False, epoch_counts=bank.epoch_counts)
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
        list(bank.title_rules),
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
    return SkillBank(
        catalog,
        bank.skills,
        dict(bank.modes),
        margins,
        thresholds,
        force_tops,
        bank.packs,
        list(bank.rules),
        list(bank.title_rules),
    )


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
    """Fit on the train split, choose skills on the validation split, then refit for submission.

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
    report["rules"] = list(evidence_bank.rules) + [
        {"attribute_id": rule["attribute_id"], "feature": "title:" + rule["phrase"], "value": rule["value"]}
        for rule in evidence_bank.title_rules
    ]
    report["selection"] = {
        "modes": selection.get("modes", {}),
        "q_during_selection": selected,
        "rules": selection.get("rules", 0),
        "title_rules": selection.get("title_rules", 0),
        "epochs": selection.get("epochs", {}),
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
