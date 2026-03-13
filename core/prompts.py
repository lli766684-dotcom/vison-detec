from pathlib import Path
from typing import Dict, List, Optional

from utils.io_utils import load_json

DEFAULT_PROMPT_CFG = {
    "object_consistency": {
        "expected_templates": [
            "a real photo of {product_name_with_article}",
            "a product photo of {product_name_with_article}",
            "a real photo of a damaged {product_name}",
            "a close-up photo of {product_name_with_article}",
            "a real photo of fresh {product_name}",
        ],
        "negative_templates": [
            "a real photo of {negative_object_with_article}",
            "a product photo of {negative_object_with_article}",
            "a real photo of a damaged {negative_object}",
        ],
        "generic_negative_templates": [
            "a cartoon image",
            "an illustration",
            "a poster",
            "a screenshot",
            "a product advertisement",
        ],
        "neutral_templates": [
            "a real photo of fruit",
            "a close-up product photo",
            "a real object on a table",
        ],
        "attribute_expected_templates": [
            "a real photo of {product_name} with {color} color",
            "a real photo of {brand} {product_name}",
            "a product photo of {product_name} in {package_form}",
            "a real photo of {product_name} with spec {spec}",
        ],
        "attribute_negative_templates": [
            "a real photo of a different brand {product_name}",
            "a real photo of {product_name} with different color",
            "a real photo of {product_name} in different package",
        ],
        "damage_templates": [
            "a close-up photo of damaged {product_name}",
            "a close-up photo of rotten {product_name}",
            "a close-up photo of broken package of {product_name}",
        ],
        "normal_state_templates": [
            "a normal intact {product_name}",
            "a fresh and undamaged {product_name}",
        ],
    },
    "confusion_objects": {
        "fruit": ["banana", "pear", "orange", "peach", "grape", "mango", "mixed fruits"]
    },
}


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def to_article_phrase(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    return f"{'an' if t[0].lower() in 'aeiou' else 'a'} {t}"


def load_prompt_config(path: Optional[str]) -> Dict:
    if path:
        p = Path(path)
    else:
        p = Path(__file__).resolve().parents[1] / "config" / "prompt_templates.json"
    data = load_json(p, default={})
    cfg = dict(DEFAULT_PROMPT_CFG)
    cfg.update(data or {})
    return cfg


def is_damage_case(req: Dict) -> bool:
    text = " ".join(
        [
            str(req.get("refund_reason", "")),
            str(req.get("product_desc", "")),
            " ".join(req.get("expected_labels") or []),
        ]
    ).lower()
    keywords = ["damage", "broken", "defect", "scratch", "rotten", "mold", "bad", "quality issue"]
    return any(k in text for k in keywords)


def build_object_prompts(req: Dict, prompt_cfg: Dict) -> Dict[str, List[str]]:
    ocfg = prompt_cfg.get("object_consistency", {})
    product_name = req["product_name"].strip()
    product_name_with_article = to_article_phrase(product_name)
    expected = [
        t.format(product_name=product_name, product_name_with_article=product_name_with_article)
        for t in ocfg.get("expected_templates", [])
    ]

    negatives_seed = list(req.get("negative_labels") or [])
    category = req.get("category", "")
    confusion_pool = (prompt_cfg.get("confusion_objects", {}) or {}).get(category, [])
    for x in confusion_pool:
        if x.lower() != product_name.lower():
            negatives_seed.append(x)
    negatives_seed = dedupe_keep_order([str(x).strip() for x in negatives_seed if str(x).strip()])

    negatives = []
    for n in negatives_seed:
        n_with_article = to_article_phrase(n)
        for t in ocfg.get("negative_templates", []):
            negatives.append(t.format(negative_object=n, negative_object_with_article=n_with_article))
    negatives.extend(ocfg.get("generic_negative_templates", []))

    return {
        "expected": dedupe_keep_order(expected),
        "negative": dedupe_keep_order(negatives),
        "neutral": dedupe_keep_order(ocfg.get("neutral_templates", [])),
    }


def build_attribute_prompts(req: Dict, prompt_cfg: Dict) -> Dict[str, List[str]]:
    ocfg = prompt_cfg.get("object_consistency", {})
    product_name = req["product_name"]
    expected_templates = ocfg.get("attribute_expected_templates", [])
    negative_templates = ocfg.get("attribute_negative_templates", [])
    attributes = ["brand", "color", "package_form", "spec"]
    by_attribute = {}
    expected_all, negative_all = [], []

    for attr in attributes:
        value = str(req.get(attr, "")).strip()
        if not value:
            continue
        slots = {
            "product_name": product_name,
            "brand": req.get("brand", "") or "generic",
            "color": req.get("color", "") or "natural",
            "package_form": req.get("package_form", "") or "standard package",
            "spec": req.get("spec", "") or "regular",
        }
        per_expected = dedupe_keep_order([t.format(**slots) for t in expected_templates])
        per_negative = dedupe_keep_order([t.format(**slots) for t in negative_templates])
        by_attribute[attr] = {"expected": per_expected, "negative": per_negative}
        expected_all.extend(per_expected)
        negative_all.extend(per_negative)

    return {"expected": dedupe_keep_order(expected_all), "negative": dedupe_keep_order(negative_all), "by_attribute": by_attribute}


def build_damage_prompts(req: Dict, prompt_cfg: Dict) -> Dict[str, List[str]]:
    ocfg = prompt_cfg.get("object_consistency", {})
    product_name = req["product_name"]
    if is_damage_case(req):
        expected = [t.format(product_name=product_name) for t in ocfg.get("damage_templates", [])]
        neutral = [t.format(product_name=product_name) for t in ocfg.get("normal_state_templates", [])]
    else:
        expected = [t.format(product_name=product_name) for t in ocfg.get("normal_state_templates", [])]
        neutral = [t.format(product_name=product_name) for t in ocfg.get("damage_templates", [])]
    return {"expected": dedupe_keep_order(expected), "neutral": dedupe_keep_order(neutral)}


def build_all_prompts(req: Dict, prompt_cfg: Dict) -> Dict:
    obj = build_object_prompts(req, prompt_cfg)
    attr = build_attribute_prompts(req, prompt_cfg)
    dmg = build_damage_prompts(req, prompt_cfg)

    candidates = []
    for txt in obj["expected"]:
        candidates.append((txt, "expected", "object"))
    for txt in obj["negative"]:
        candidates.append((txt, "negative", "object"))
    for txt in obj["neutral"]:
        candidates.append((txt, "neutral", "object"))
    for attr_name, attr_prompts in attr.get("by_attribute", {}).items():
        for txt in attr_prompts.get("expected", []):
            candidates.append((txt, "expected", f"attribute:{attr_name}"))
        for txt in attr_prompts.get("negative", []):
            candidates.append((txt, "negative", f"attribute:{attr_name}"))
    for txt in dmg["expected"]:
        candidates.append((txt, "expected", "damage"))
    for txt in dmg["neutral"]:
        candidates.append((txt, "neutral", "damage"))

    return {"object_prompts": obj, "attribute_prompts": attr, "damage_prompts": dmg, "candidates": candidates}

