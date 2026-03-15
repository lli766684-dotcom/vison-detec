"""
core/evidence.py

证据输出模块
负责：生成结构化证据和面向人的解释文本

新主链：ingest -> preprocess -> compliance -> multimodal_analysis -> fusion -> evidence
旧链（可选对照）：feature_extract -> object_consistency
"""

from typing import Dict, List, Optional


# shot_type 中文标签映射
SHOT_TYPE_LABELS = {
    "damage_closeup": "商品损坏细节图",
    "damage_with_order_link": "商品与订单/商家关联图",
    "overview": "商品整体图",
    "merchant_label_closeup": "商家标签/包装近景图",
    "auxiliary": "其他辅助图",
}


def _get_shot_type_label(shot_type: str) -> str:
    return SHOT_TYPE_LABELS.get(shot_type, shot_type)


def evidence_writer(
    comp: Dict, 
    obj: Optional[Dict], 
    fusion: Dict, 
    mm_results: List[Dict] = None
) -> List[str]:
    """
    生成证据说明文本
    
    Args:
        comp: compliance 模块输出
        obj: 可选，object_consistency 模块输出（旧链）
        fusion: fusion 模块输出
        mm_results: multimodal_analysis 模块输出
    
    Returns:
        证据说明文本列表
    """
    lines = []
    
    # ==================== 多模态分析结果 ====================
    if mm_results and len(mm_results) > 0:
        lines.append("【多模态图像理解】")
        
        # 统计汇总
        damage_detected = any(mm.get("contains_damage", False) for mm in mm_results if mm.get("analysis_success", True))
        linkage_detected = any(mm.get("contains_order_or_merchant_linkage", False) for mm in mm_results if mm.get("analysis_success", True))
        real_scene_count = sum(1 for mm in mm_results if mm.get("analysis_success", True) and mm.get("is_real_scene_likely", True))
        total_valid = sum(1 for mm in mm_results if mm.get("analysis_success", True))
        
        # 损坏证据说明
        if damage_detected:
            damage_confs = [mm.get("damage_confidence", 0) for mm in mm_results if mm.get("contains_damage") and mm.get("analysis_success", True)]
            avg_conf = sum(damage_confs) / len(damage_confs) if damage_confs else 0
            if avg_conf > 0.7:
                lines.append(f"  ✅ 检测到明确损坏证据（平均置信度={avg_conf:.2f}）")
            else:
                lines.append(f"  ⚠️ 检测到损坏证据，但置信度中等（平均置信度={avg_conf:.2f}）")
        else:
            lines.append("  ❌ 未检测到明确损坏证据")
        
        # 关联证据说明
        if linkage_detected:
            linkage_confs = [mm.get("linkage_confidence", 0) for mm in mm_results if mm.get("contains_order_or_merchant_linkage") and mm.get("analysis_success", True)]
            avg_conf = sum(linkage_confs) / len(linkage_confs) if linkage_confs else 0
            if avg_conf > 0.7:
                lines.append(f"  ✅ 检测到订单/商家关联证据（平均置信度={avg_conf:.2f}）")
            else:
                lines.append(f"  ⚠️ 检测到关联证据，但置信度中等（平均置信度={avg_conf:.2f}）")
        else:
            lines.append("  ⚠️ 未检测到订单/商家关联证据，建议补充包装/标签近景图")
        
        # 场景真实性说明
        if total_valid > 0:
            real_ratio = real_scene_count / total_valid
            if real_ratio >= 0.8:
                lines.append(f"  ✅ 图像更像真实拍摄场景")
            elif real_ratio >= 0.5:
                lines.append(f"  ⚠️ 部分图像场景真实性存疑")
            else:
                lines.append(f"  ❌ 多数图像疑似非实拍场景，建议人工复核")
        
        # 商品匹配度说明
        product_matches = [mm.get("product_match_confidence", 0.5) for mm in mm_results if mm.get("analysis_success", True)]
        if product_matches:
            avg_match = sum(product_matches) / len(product_matches)
            if avg_match > 0.7:
                lines.append(f"  ✅ 商品与订单主体参考图匹配较高（平均匹配度={avg_match:.2f}）")
            elif avg_match > 0.4:
                lines.append(f"  ⚠️ 商品与参考图匹配中等（平均匹配度={avg_match:.2f}）")
            else:
                lines.append(f"  ❌ 商品与参考图匹配较低（平均匹配度={avg_match:.2f}），可能存在不一致")
        
        lines.append("")
        
        # 每张图详情
        lines.append("【各图分析详情】")
        for idx, mm in enumerate(mm_results):
            if not mm.get("analysis_success", True):
                lines.append(f"  图{idx+1}：分析失败 - {mm.get('error', '未知错误')}")
                continue
            
            # 类型判断
            declared = mm.get("declared_shot_type", "-")
            predicted = mm.get("predicted_shot_type", "-")
            type_match = declared == predicted
            type_icon = "✅" if type_match else "⚠️"
            lines.append(f"  图{idx+1} [{declared}]：模型判断={predicted} {type_icon}")
            
            # 内容检测摘要
            flags = []
            if mm.get("contains_damage"):
                flags.append(f"损坏({mm.get('damage_confidence', 0):.0%})")
            if mm.get("contains_order_or_merchant_linkage"):
                flags.append(f"关联({mm.get('linkage_confidence', 0):.0%})")
            if not mm.get("is_real_scene_likely"):
                flags.append("非实拍")
            
            if flags:
                lines.append(f"    检测：{', '.join(flags)}")
            else:
                lines.append(f"    检测：无特殊标记")
            
            # 摘要
            summary = mm.get("summary", "")
            if summary:
                lines.append(f"    说明：{summary[:50]}..." if len(summary) > 50 else f"    说明：{summary}")
            
            # 新字段：按类型展示特殊信息
            predicted_type = mm.get("predicted_shot_type", "")
            
            # damage_closeup 和 damage_with_order_link：损坏描述
            if predicted_type in ["damage_closeup", "damage_with_order_link"]:
                damage_desc = mm.get("damage_description", "")
                if damage_desc:
                    lines.append(f"    损坏描述：{damage_desc[:60]}..." if len(damage_desc) > 60 else f"    损坏描述：{damage_desc}")
            
            # damage_with_order_link：关联证据描述
            if predicted_type == "damage_with_order_link":
                linkage_desc = mm.get("linkage_description", "")
                if linkage_desc:
                    lines.append(f"    关联证据：{linkage_desc[:60]}..." if len(linkage_desc) > 60 else f"    关联证据：{linkage_desc}")
                
                # 关联物匹配度
                ref_linkage = mm.get("reference_linkage_confidence", 0)
                if ref_linkage > 0:
                    lines.append(f"    关联物匹配度：{ref_linkage:.0%}")
            
            # merchant_label_closeup：标签可读性
            if predicted_type == "merchant_label_closeup":
                readability = mm.get("label_readability", "未知")
                readability_icon = {"清晰": "✅", "模糊": "⚠️", "不可读": "❌"}.get(readability, "❓")
                lines.append(f"    标签可读性：{readability_icon} {readability}")
                
                # 关联物匹配度
                ref_linkage = mm.get("reference_linkage_confidence", 0)
                if ref_linkage > 0:
                    lines.append(f"    与参考图匹配度：{ref_linkage:.0%}")
            
            # overview：整体视图
            if predicted_type == "overview":
                shows_overall = mm.get("shows_overall_product_view", False)
                icon = "✅" if shows_overall else "⚠️"
                lines.append(f"    展示商品全貌：{icon} {'是' if shows_overall else '否'}")
            
            # auxiliary：声明支持情况
            if predicted_type == "auxiliary":
                claim_level = mm.get("claim_support_level", "unknown")
                claim_icon = {"supported": "✅", "partially_supported": "⚠️", "not_supported": "❌"}.get(claim_level, "❓")
                claim_labels = {"supported": "完全支持", "partially_supported": "部分支持", "not_supported": "不支持"}
                lines.append(f"    声明支持：{claim_icon} {claim_labels.get(claim_level, claim_level)}")
                
                claim_analysis = mm.get("claim_analysis", "")
                if claim_analysis:
                    lines.append(f"    分析：{claim_analysis[:60]}..." if len(claim_analysis) > 60 else f"    分析：{claim_analysis}")
        
        lines.append("")
    
    # ==================== 必传图状态 ====================
    lines.append("【必传图状态】")
    required_passed = comp.get("required_passed", True)
    shot_type_counts = comp.get("shot_type_counts", {})
    required_shot_types = comp.get("required_shot_types_min", ["damage_closeup", "damage_with_order_link"])
    
    for shot_type in required_shot_types:
        label = _get_shot_type_label(shot_type)
        count = shot_type_counts.get(shot_type, 0)
        
        # 获取限制配置
        from core.compliance import SHOT_TYPE_LIMITS
        limit_cfg = SHOT_TYPE_LIMITS.get(shot_type, {"min": 1, "max": 3})
        
        if count >= limit_cfg.get("min", 1):
            lines.append(f"  ✅ {label}：已上传 {count} 张")
        else:
            lines.append(f"  ❌ {label}：未上传（最少需要 {limit_cfg.get('min', 1)} 张）")
    
    if required_passed:
        lines.append("  ➜ 必传状态：已满足")
    else:
        missing = comp.get("missing_required_shots", [])
        missing_labels = [_get_shot_type_label(s) for s in missing]
        lines.append(f"  ➜ 必传状态：缺少 {', '.join(missing_labels)}")
    
    lines.append("")
    
    # ==================== 推荐补充图状态 ====================
    lines.append("【推荐补充图状态】")
    recommended = comp.get("recommended_shot_types", ["overview", "merchant_label_closeup", "auxiliary"])
    recommended_coverage = comp.get("recommended_coverage", {})
    
    for shot_type in recommended:
        label = _get_shot_type_label(shot_type)
        count = shot_type_counts.get(shot_type, 0)
        from core.compliance import SHOT_TYPE_LIMITS
        limit_cfg = SHOT_TYPE_LIMITS.get(shot_type, {"max": 2})
        max_count = limit_cfg.get("max", 2)
        
        if count > 0:
            lines.append(f"  ✅ {label}：已上传 {count}/{max_count} 张")
        else:
            lines.append(f"  ⚪ {label}：未上传（可选，最多 {max_count} 张）")
    
    hit = recommended_coverage.get("hit", 0)
    total = recommended_coverage.get("total", 3)
    lines.append(f"  ➜ 推荐覆盖：{hit}/{total} 类")
    
    lines.append("")
    
    # ==================== 辅助图说明 ====================
    auxiliary_notes = comp.get("auxiliary_notes", [])
    if auxiliary_notes:
        lines.append("【辅助图说明】")
        # auxiliary_notes 现在是字符串列表
        for note in auxiliary_notes:
            lines.append(f"  • {note}")
        lines.append("")
    
    # ==================== 图像质量问题 ====================
    issues = comp.get("compliance_issues", [])
    if issues:
        lines.append("【图像质量问题】")
        for issue in issues:
            lines.append(f"  ⚠️ {issue}")
        lines.append("")
    
    # ==================== 超限警告 ====================
    shot_type_over_limit = comp.get("shot_type_over_limit", [])
    if shot_type_over_limit:
        lines.append("【数量超限警告】")
        for item in shot_type_over_limit:
            label = _get_shot_type_label(item["shot_type"])
            lines.append(f"  ⚠️ {label}：已上传 {item['count']} 张，超出上限 {item['max']} 张")
        lines.append("")
    
    # ==================== 旧链对照结果（如有） ====================
    if obj:
        lines.append("【旧链对照结果（CLIP/Embedding）】")
        status_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(obj.get("status", "unknown"), "❓")
        lines.append(f"  {status_icon} 状态：{obj.get('status', 'unknown').upper()}（置信度={obj.get('score', 0):.2f}）")
        lines.append(f"  类目匹配：{obj.get('category_match', 'unknown')}")
        
        attr_match = obj.get("attribute_match", {})
        if attr_match:
            attr_items = []
            for k, v in attr_match.items():
                status = v.get("status", "unknown") if isinstance(v, dict) else str(v)
                attr_items.append(f"{k}={status}")
            lines.append(f"  属性匹配：{', '.join(attr_items)}")
        
        damage_evidence = obj.get("damage_evidence", "unknown")
        damage_score = obj.get("damage_score", 0.0)
        damage_icon = {"present": "✅", "weak": "⚠️", "absent": "❌", "not_required": "⚪"}.get(damage_evidence, "❓")
        lines.append(f"  {damage_icon} 损坏证据：{damage_evidence}（score={damage_score:.2f}）")
        
        lines.append("")
    
    # ==================== 融合结论 ====================
    lines.append("【融合结论】")
    
    # 融合原因
    fusion_reason = fusion.get("fusion_reason", [])
    if fusion_reason:
        lines.append("  判定依据：")
        for reason in fusion_reason:
            lines.append(f"    • {reason}")
    
    risk_icon = {"low": "✅", "medium": "⚠️", "high": "❌"}.get(fusion.get("image_risk_level", "high"), "❓")
    lines.append(f"  {risk_icon} 风险等级：{fusion.get('image_risk_level', 'high').upper()}")
    lines.append(f"  置信度：{fusion.get('confidence_score', 0):.2f}")
    lines.append(f"  最终判定：{fusion.get('object_match', 'fail').upper()}")
    
    return lines


def build_object_evidence(
    comp: Dict, 
    obj: Optional[Dict], 
    evidence_text: List[str],
    mm_results: List[Dict] = None,
    fusion: Dict = None
) -> Dict:
    """
    构建结构化证据输出
    
    Args:
        comp: compliance 模块输出
        obj: 可选，object_consistency 模块输出（旧链）
        evidence_text: 证据说明文本
        mm_results: multimodal_analysis 模块输出
        fusion: fusion 模块输出
    
    Returns:
        结构化证据字典
    """
    result = {
        "evidence_compliance": comp.get("evidence_compliance"),
        "shot_completeness": comp.get("shot_completeness"),
        "detected_shot_types": comp.get("detected_shot_types"),
        "missing_required_shots": comp.get("missing_required_shots"),
        "required_passed": comp.get("required_passed"),
        "required_shot_types_min": comp.get("required_shot_types_min", []),
        "recommended_shot_types": comp.get("recommended_shot_types", []),
        "recommended_coverage": comp.get("recommended_coverage", {}),
        "evidence_rules_result": comp.get("evidence_rules_result", {}),
        "evidence_text": evidence_text,
    }
    
    # 多模态证据
    if mm_results:
        damage_images = [mm["path"] for mm in mm_results if mm.get("contains_damage") and mm.get("analysis_success", True)]
        linkage_images = [mm["path"] for mm in mm_results if mm.get("contains_order_or_merchant_linkage") and mm.get("analysis_success", True)]
        
        result["multimodal_evidence"] = {
            "damage_images": damage_images,
            "linkage_images": linkage_images,
            "total_analyzed": len(mm_results),
            "analysis_success_count": sum(1 for mm in mm_results if mm.get("analysis_success", True)),
        }
    
    # 融合结果
    if fusion:
        result["fusion_summary"] = {
            "object_match": fusion.get("object_match"),
            "confidence_score": fusion.get("confidence_score"),
            "image_risk_level": fusion.get("image_risk_level"),
            "fusion_reason": fusion.get("fusion_reason", []),
        }
    
    # 旧链结果（如有）
    if obj:
        result["legacy_object_consistency"] = {
            "status": obj.get("status"),
            "score": obj.get("score"),
            "category_match": obj.get("category_match"),
            "attribute_match": obj.get("attribute_match"),
            "damage_evidence": obj.get("damage_evidence"),
            "damage_score": obj.get("damage_score", 0.0),
            "damage_images": obj.get("damage_images", []),
            "damage_reason": obj.get("damage_reason", ""),
            "best_matched_reference_image": obj.get("best_matched_reference_image"),
            "per_image_results": obj.get("per_image_results"),
            "reason": obj.get("reason"),
        }
    
    return result