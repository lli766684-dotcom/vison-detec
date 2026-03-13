from pathlib import Path
from typing import Dict, List, Optional


# 数量限制配置
SHOT_TYPE_LIMITS = {
    "damage_closeup": {"min": 1, "max": 3},
    "damage_with_order_link": {"min": 1, "max": 3},
    "overview": {"min": 0, "max": 3},
    "merchant_label_closeup": {"min": 0, "max": 2},
    "auxiliary": {"min": 0, "max": 2},
}


def detect_shot_type_by_filename(path_str: str) -> str:
    """
    根据文件名关键词推断 shot_type（已废弃，仅保留兼容）
    
    注意：当前系统不再使用此函数，shot_type 完全由前端标注决定。
    """
    n = Path(path_str).stem.lower()
    damage_keys = ["damage", "broken", "detail", "close", "defect", "scratch", "rot", "mold"]
    merchant_keys = ["marker", "label", "logo", "shop", "merchant", "package", "sticker", "tag", "order", "link"]
    aux_keys = ["count", "quantity", "scene", "env", "table", "box", "all"]

    has_damage = any(k in n for k in damage_keys)
    has_merchant = any(k in n for k in merchant_keys)
    has_aux = any(k in n for k in aux_keys)

    # 统一使用新名称
    if has_damage and has_merchant:
        return "damage_with_order_link"
    if has_damage:
        return "damage_closeup"
    if has_merchant:
        return "merchant_label_closeup"
    if has_aux:
        return "auxiliary"
    return "overview"


def resolve_shot_type_by_path(path: str, req: Dict) -> str:
    """
    解析 shot_type：从 refund_image_meta 字典读取前端标注
    
    核心逻辑：真正的图类型来源是 req["refund_image_meta"][path]["shot_type"]
    不再使用文件名推断。
    
    如果缺少 shot_type，直接报错（纯新逻辑，不再兼容旧格式）
    """
    meta_dict = req.get("refund_image_meta", {})
    
    meta = meta_dict.get(path)
    if meta is None:
        raise ValueError(f"退款图[{path}] 在 refund_image_meta 中不存在")
    
    shot_type = meta.get("shot_type")
    
    if shot_type and isinstance(shot_type, str) and shot_type.strip():
        return shot_type.strip()
    
    # 纯新逻辑：缺少 shot_type 直接报错
    raise ValueError(f"退款图[{path}] 缺少 shot_type")


def resolve_shot_type_by_index(idx: int, req: Dict) -> str:
    """
    兼容函数：通过索引解析 shot_type
    
    内部调用 resolve_shot_type_by_path
    """
    refund_images = req.get("refund_images", [])
    if idx >= len(refund_images):
        raise ValueError(f"退款图索引越界：idx={idx}, len={len(refund_images)}")
    
    path = refund_images[idx]
    return resolve_shot_type_by_path(path, req)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _resolve_required_min(req: Dict, rules_cfg: Dict) -> List[str]:
    # 新版必传：damage_closeup + damage_with_order_link
    return list(
        req.get("required_shot_types_min")
        or rules_cfg.get("required_shot_types_min")
        or req.get("required_shot_types")
        or ["damage_closeup", "damage_with_order_link"]
    )


def _resolve_recommended(req: Dict, rules_cfg: Dict) -> List[str]:
    # 新版推荐：overview + merchant_label_closeup + auxiliary
    return list(
        req.get("recommended_shot_types")
        or rules_cfg.get("recommended_shot_types")
        or ["overview", "merchant_label_closeup", "auxiliary"]
    )


def check_shot_type_consistency(mm_results: List[Dict], req: Dict) -> Dict:
    """
    检查声明类型 vs 模型预测类型一致性
    
    Args:
        mm_results: 多模态分析结果列表
        req: 标准化请求对象
    
    Returns:
        一致性检查结果
    """
    if not mm_results:
        return {
            "checked": False,
            "reason": "无多模态分析结果",
            "mismatches": [],
            "mismatch_count": 0,
        }
    
    mismatches = []
    for mm in mm_results:
        path = mm.get("path", "")
        declared = mm.get("declared_shot_type", "")
        predicted = mm.get("predicted_shot_type", "")
        
        if declared and predicted and declared != predicted:
            mismatches.append({
                "path": path,
                "declared_shot_type": declared,
                "predicted_shot_type": predicted,
            })
    
    return {
        "checked": True,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "all_match": len(mismatches) == 0,
    }


def evidence_compliance(req: Dict, prep: Dict, thresholds: Dict) -> Dict:
    """
    证据合规性检查（纯新逻辑版）
    
    核心逻辑：
    1. 从 refund_image_meta 读取前端标注的 shot_type
    2. 统计每类图数量
    3. 检查必传是否满足
    4. 检查推荐是否覆盖
    5. 检查数量是否超限
    6. 收集 auxiliary 备注
    7. 给出最终合规等级
    """
    refund_images = req["refund_images"]
    refund_image_meta = req.get("refund_image_meta", {})
    min_refund_images = int(thresholds["image"]["min_refund_images"])
    rules_cfg = thresholds.get("evidence_rules", {})
    frontend_block_on_missing_required = bool(rules_cfg.get("frontend_block_on_missing_required", True))

    required_min = _resolve_required_min(req, rules_cfg)
    recommended = _resolve_recommended(req, rules_cfg)

    # 1. 从 refund_image_meta 读取 shot_type，统计每类图数量
    shot_type_counts = {}
    for shot_type in SHOT_TYPE_LIMITS.keys():
        shot_type_counts[shot_type] = 0
    
    for idx in range(len(refund_images)):
        shot_type = resolve_shot_type_by_index(idx, req)
        if shot_type and shot_type in shot_type_counts:
            shot_type_counts[shot_type] += 1
    
    # 2. 检查必传是否满足
    missing_required = []
    for shot_type in required_min:
        min_count = SHOT_TYPE_LIMITS.get(shot_type, {}).get("min", 1)
        if shot_type_counts.get(shot_type, 0) < min_count:
            missing_required.append(shot_type)
    
    required_passed = len(missing_required) == 0

    # 3. 检查推荐是否覆盖
    recommended_missing = []
    for shot_type in recommended:
        if shot_type_counts.get(shot_type, 0) == 0:
            recommended_missing.append(shot_type)
    
    recommended_hit = len(recommended) - len(recommended_missing)

    # 4. 检查数量是否超限
    shot_type_over_limit = []
    for shot_type, count in shot_type_counts.items():
        max_count = SHOT_TYPE_LIMITS.get(shot_type, {}).get("max", 999)
        if count > max_count:
            shot_type_over_limit.append(shot_type)

    # 5. 收集 auxiliary 备注（适配字典格式）
    auxiliary_notes = []
    for path in refund_images:
        meta = refund_image_meta.get(path, {})
        shot_type = resolve_shot_type_by_path(path, req)
        note = meta.get("note", "")
        if shot_type == "auxiliary" and note:
            auxiliary_notes.append(note)

    # 6. 收集图像质量问题
    issues = []
    for p in refund_images:
        flags = prep["image_quality_flags"].get(p, {})
        if flags.get("blurry"):
            issues.append(f"清晰度不足: {p}")
        if flags.get("dark"):
            issues.append(f"亮度过低: {p}")
    if len(refund_images) < min_refund_images:
        issues.append("退款图数量不足")

    # 7. 给出最终合规等级
    # - insufficient: 缺少任一必传图
    # - partial: 必传满足，但推荐没全满足 或 图片有质量问题 或 某类超限
    # - complete: 必传全满足 + 推荐都满足 + 无质量问题 + 无超限
    if not required_passed:
        compliance = "insufficient"
    elif recommended_missing or issues or shot_type_over_limit:
        compliance = "partial"
    else:
        compliance = "complete"

    # 兼容旧字段
    shot_completeness = "insufficient" if not required_passed else ("partial" if recommended_missing else "complete")
    detected_set = set(shot_type for shot_type in shot_type_counts if shot_type_counts[shot_type] > 0)

    return {
        # 核心输出字段（新文档要求）
        "required_passed": required_passed,
        "missing_required_shots": sorted(_dedupe_keep_order(missing_required)),
        "recommended_coverage": {
            "hit": recommended_hit,
            "total": len(recommended),
            "missing": sorted(_dedupe_keep_order(recommended_missing)),
        },
        "shot_type_counts": shot_type_counts,
        "shot_type_over_limit": shot_type_over_limit,
        "auxiliary_notes": auxiliary_notes,
        "compliance_level": compliance,
        
        # 兼容旧字段
        "evidence_compliance": compliance,
        "shot_completeness": shot_completeness,
        "detected_shot_types": sorted(list(detected_set)),
        "compliance_issues": _dedupe_keep_order(issues),
        "required_shot_types_min": required_min,
        "recommended_shot_types": recommended,
        "frontend_block_on_missing_required": frontend_block_on_missing_required,
        "evidence_rules_result": {
            "required_passed": required_passed,
            "required_shot_types_min": required_min,
            "required_missing": sorted(_dedupe_keep_order(missing_required)),
            "recommended_shot_types": recommended,
            "recommended_coverage": {
                "hit": recommended_hit,
                "total": len(recommended),
                "missing": sorted(_dedupe_keep_order(recommended_missing)),
            },
            "policy": {
                "frontend_block_on_missing_required": frontend_block_on_missing_required,
            },
            "compliance_level": compliance,
        },
    }
