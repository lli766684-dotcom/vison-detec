from typing import Dict, List, Tuple

import numpy as np

from core.prompts import build_all_prompts, is_damage_case
from utils.math_utils import clamp01, cosine


def feature_extract(req: Dict, prep: Dict, backend) -> Dict:
    order_reference_vectors = {}
    refund_vectors = {}
    for p, img in prep["order_reference_images_processed"].items():
        order_reference_vectors[p] = backend.image_embedding(img)
    for p, img in prep["refund_images_processed"].items():
        refund_vectors[p] = backend.image_embedding(img)
    return {"order_reference_vectors": order_reference_vectors, "refund_vectors": refund_vectors}


def _status_by_signals(label_group: str, sim: float, thresholds: Dict) -> str:
    sim_cfg = thresholds["similarity"]
    if label_group == "negative":
        return "fail"
    if label_group == "expected" and sim >= float(sim_cfg["sim_high"]):
        return "pass"
    if label_group == "expected" and sim >= float(sim_cfg["sim_mid"]):
        return "warn"
    if label_group == "neutral" and sim >= float(sim_cfg["sim_high"]):
        return "warn"
    if sim < float(sim_cfg["sim_low"]):
        return "fail"
    return "warn"


def _score_by_signals(label_group: str, text_score: float, sim: float, thresholds: Dict) -> float:
    base = 0.5 * text_score + 0.5 * sim
    base += float(thresholds["similarity"]["group_prior"].get(label_group, 0.0))
    return clamp01(base)


def score_single_image(
    img_path: str,
    rvec: np.ndarray,
    order_reference_vectors: Dict[str, np.ndarray],
    text_mat: np.ndarray,
    texts: List[str],
    groups: List[str],
    sources: List[str],
    thresholds: Dict,
) -> Dict:
    best_ref, best_ref_sim = None, -1.0
    for ref_path, ref_vec in order_reference_vectors.items():
        sim = cosine(rvec, ref_vec)
        if sim > best_ref_sim:
            best_ref_sim = sim
            best_ref = ref_path

    sims = text_mat @ rvec
    top_idx = int(np.argmax(sims))
    second_idx = int(np.argsort(sims)[-2]) if len(sims) > 1 else top_idx
    top_score, second_score = float(sims[top_idx]), float(sims[second_idx])
    top_label, second_label = texts[top_idx], texts[second_idx]
    top_group, top_source = groups[top_idx], sources[top_idx]

    status = _status_by_signals(top_group, best_ref_sim, thresholds)
    score = _score_by_signals(top_group, top_score, best_ref_sim, thresholds)
    if top_group == "expected" and abs(top_score - second_score) <= float(thresholds["similarity"]["conflict_margin"]):
        status = "warn"

    return {
        "image": img_path,
        "status": status,
        "score": score,
        "top_label": top_label,
        "top_score": top_score,
        "second_label": second_label,
        "second_score": second_score,
        "label_group": top_group,
        "candidate_group": top_group,
        "prompt_source": top_source,
        "max_reference_similarity": float(best_ref_sim),
        "best_matched_reference_image": best_ref,
        "preliminary_object_judgement": "consistent" if status == "pass" else "suspicious" if status == "warn" else "inconsistent",
    }


def aggregate_image_results(per_image: List[Dict], compliance: Dict, thresholds: Dict) -> Tuple[str, float, Dict, Dict, str, float]:
    status_counts = {"pass": 0, "warn": 0, "fail": 0}
    group_counts = {"expected": 0, "neutral": 0, "negative": 0}
    best_ref_img, best_ref_sim = None, -1.0
    for row in per_image:
        status_counts[row["status"]] += 1
        group_counts[row["label_group"]] += 1
        if row["max_reference_similarity"] > best_ref_sim:
            best_ref_sim = row["max_reference_similarity"]
            best_ref_img = row["best_matched_reference_image"]

    n = max(1, len(per_image))
    fail_ratio = status_counts["fail"] / n
    warn_ratio = status_counts["warn"] / n
    has_expected = group_counts["expected"] > 0
    has_negative = group_counts["negative"] > 0
    agg_cfg = thresholds["aggregation"]

    if fail_ratio >= float(agg_cfg["agg_fail_ratio"]):
        agg_status = "fail"
    elif has_expected and has_negative:
        agg_status = "warn"
    elif warn_ratio >= float(agg_cfg["agg_warn_ratio"]):
        agg_status = "warn"
    elif status_counts["pass"] > 0 and status_counts["warn"] == 0 and status_counts["fail"] == 0:
        agg_status = "pass"
    elif status_counts["pass"] > 0 and status_counts["warn"] > 0:
        agg_status = "warn"
    else:
        agg_status = "fail"

    comp = compliance.get("evidence_compliance", "complete")
    if comp == "insufficient":
        agg_status = "fail"
    elif comp == "partial" and agg_status == "pass":
        agg_status = "warn"

    agg_score = float(np.mean([r["score"] for r in per_image])) if per_image else 0.0
    if comp == "partial":
        agg_score = clamp01(agg_score - 0.08)
    elif comp == "insufficient":
        agg_score = clamp01(agg_score - 0.18)
    return agg_status, agg_score, status_counts, group_counts, best_ref_img, best_ref_sim


def infer_category_match(group_counts: Dict[str, int]) -> str:
    if group_counts["expected"] >= max(group_counts["negative"], group_counts["neutral"]):
        return "matched"
    if group_counts["negative"] > group_counts["expected"]:
        return "mismatched"
    return "uncertain"


def infer_attribute_match(req: Dict, per_image: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for attr in ["brand", "color", "package_form", "spec"]:
        if not str(req.get(attr, "")).strip():
            out[attr] = {"status": "unknown", "top_prompt": None, "top_score": 0.0}
            continue
        attr_rows = [r for r in per_image if r.get("prompt_source") == f"attribute:{attr}"]
        if not attr_rows:
            out[attr] = {"status": "unknown", "top_prompt": None, "top_score": 0.0}
            continue
        top_row = max(attr_rows, key=lambda r: r.get("top_score", 0.0))
        g = top_row.get("label_group")
        status = "matched" if g == "expected" else "mismatched" if g == "negative" else "unknown"
        out[attr] = {"status": status, "top_prompt": top_row.get("top_label"), "top_score": round(float(top_row.get("top_score", 0.0)), 4)}
    return out


def infer_damage_evidence(req: Dict, per_image: List[Dict]) -> Dict:
    damage_case = is_damage_case(req)
    damage_rows = [r for r in per_image if r.get("prompt_source") == "damage"]
    damage_images = [{"image": r["image"], "score": round(float(r["top_score"]), 4)} for r in sorted(damage_rows, key=lambda x: x.get("top_score", 0.0), reverse=True)]
    damage_score = float(np.mean([r["top_score"] for r in damage_rows])) if damage_rows else 0.0
    damage_expected = any(r.get("label_group") == "expected" for r in damage_rows)
    if not damage_case:
        return {"damage_evidence": "not_required", "damage_score": round(damage_score, 4), "damage_images": damage_images, "damage_reason": "退款原因未指向损坏类案件，本次不强制要求损坏证据。"}
    if damage_expected:
        return {"damage_evidence": "present", "damage_score": round(damage_score, 4), "damage_images": damage_images, "damage_reason": "已检测到与损坏描述一致的图像线索。"}
    if damage_rows:
        return {"damage_evidence": "weak", "damage_score": round(damage_score, 4), "damage_images": damage_images, "damage_reason": "存在疑似损坏线索，但强度不足，建议人工复核。"}
    return {"damage_evidence": "absent", "damage_score": 0.0, "damage_images": [], "damage_reason": "未检测到明确匹配损坏描述的图像证据。"}


def _build_reason(status: str) -> str:
    if status == "pass":
        return "商品对象与订单参考图基本一致，当前证据可作为有效候选。"
    if status == "warn":
        return "商品对象存在部分一致或证据不完整，建议结合其他维度继续复核。"
    return "商品对象与订单参考图不一致，或证据质量/完整性不足，不能直接作为有效依据。"


def object_consistency(req: Dict, feats: Dict, backend, prompt_cfg: Dict, compliance: Dict, thresholds: Dict) -> Dict:
    all_prompts = build_all_prompts(req, prompt_cfg)
    candidates = all_prompts["candidates"]
    texts = [x[0] for x in candidates]
    groups = [x[1] for x in candidates]
    sources = [x[2] for x in candidates]
    if not texts:
        raise ValueError("No prompts generated for object consistency")

    text_mat = backend.text_embeddings(texts)
    per_image = [
        score_single_image(img_path, rvec, feats["order_reference_vectors"], text_mat, texts, groups, sources, thresholds)
        for img_path, rvec in feats["refund_vectors"].items()
    ]
    agg_status, agg_score, status_counts, group_counts, best_ref_img, best_ref_sim = aggregate_image_results(per_image, compliance, thresholds)
    damage_info = infer_damage_evidence(req, per_image)
    return {
        "status": agg_status,
        "score": round(agg_score, 4),
        "reason": _build_reason(agg_status),
        "category_match": infer_category_match(group_counts),
        "attribute_match": infer_attribute_match(req, per_image),
        "damage_evidence": damage_info["damage_evidence"],
        "damage_score": damage_info["damage_score"],
        "damage_images": damage_info["damage_images"],
        "damage_reason": damage_info["damage_reason"],
        "best_matched_reference_image": best_ref_img,
        "per_image_results": per_image,
        "raw_signals": {
            "status_counts": status_counts,
            "group_counts": group_counts,
            "max_reference_similarity": round(float(best_ref_sim), 4),
            "prompt_groups": {"object": all_prompts["object_prompts"], "attribute": all_prompts["attribute_prompts"], "damage": all_prompts["damage_prompts"]},
        },
    }

