"""
core/fusion.py

融合层模块
负责：综合 compliance、multimodal_analysis 和 ai_fake_detector 结果，输出最终判定

新主链：ingest -> preprocess -> compliance -> multimodal_analysis -> ai_fake_detector -> fusion -> evidence
旧链（可选对照）：feature_extract -> object_consistency
"""

from typing import Dict, List, Optional

from utils.math_utils import clamp01


def fusion_engine(
    comp: Dict,
    thresholds: Dict,
    mm_results: List[Dict] = None,
    ai_results: List[Dict] = None,  # 新增：AI 检测结果
    obj: Dict = None,  # 可选：旧链 object_consistency 结果，用于对照
) -> Dict:
    """
    融合引擎 - 综合多源信号输出最终判定
    
    Args:
        comp: compliance 模块输出（证据合规性）
        thresholds: 阈值配置
        mm_results: multimodal_analysis 模块输出（多模态分析结果）
        ai_results: ai_fake_detector 模块输出（AI 生成/真实性检测结果）
        obj: 可选，object_consistency 模块输出（旧链对照）
    
    Returns:
        最终判定结果，包含：
        - object_match: 匹配状态 (pass/warn/fail)
        - confidence_score: 置信度 (0.0-1.0)
        - image_risk_level: 风险等级 (low/medium/high)
        - mm_adjustment: 多模态调整幅度
        - ai_adjustment: AI 检测调整幅度
        - fusion_reason: 融合原因列表
        - legacy_object_match: 旧链对照结果（如有）
    """
    fusion_reason = []
    
    # ==================== 基础状态计算（来自 comp） ====================
    evidence_compliance = comp.get("evidence_compliance", "insufficient")
    required_passed = comp.get("required_passed", False)
    
    # 基于 evidence_compliance 确定基础状态和置信度
    if evidence_compliance == "complete":
        base_status = "pass"
        base_confidence = 0.75
        fusion_reason.append("证据完整合规")
    elif evidence_compliance == "partial":
        base_status = "warn"
        base_confidence = 0.55
        fusion_reason.append("证据部分合规")
    else:  # insufficient
        base_status = "fail"
        base_confidence = 0.35
        fusion_reason.append("证据不完整")
    
    # 必传图未满足直接降级
    if not required_passed:
        base_status = "fail"
        base_confidence = min(base_confidence, 0.30)
        missing = comp.get("missing_required_shots", [])
        if missing:
            fusion_reason.append(f"缺少必传图: {', '.join(missing)}")
    
    # ==================== 多模态信号融合 ====================
    mm_adjustment = 0.0
    mm_signals = {
        "damage_detected": False,
        "damage_confidence_avg": 0.0,
        "linkage_detected": False,
        "linkage_confidence_avg": 0.0,
        "real_scene_avg": 0.0,
        "product_match_avg": 0.0,
        "reference_linkage_avg": 0.0,
        "type_consistency_rate": 1.0,
        # 新增字段
        "claim_supported_rate": 1.0,  # auxiliary 类型：声明支持率
        "label_readable_rate": 1.0,   # merchant_label_closeup 类型：标签可读率
        "overall_view_rate": 1.0,     # overview 类型：整体视图率
    }
    
    if mm_results and len(mm_results) > 0:
        valid_results = [mm for mm in mm_results if mm.get("analysis_success", True)]
        
        if valid_results:
            # 损坏检测
            damage_detected = any(mm.get("contains_damage", False) for mm in valid_results)
            damage_confs = [mm.get("damage_confidence", 0) for mm in valid_results if mm.get("contains_damage")]
            mm_signals["damage_detected"] = damage_detected
            mm_signals["damage_confidence_avg"] = sum(damage_confs) / len(damage_confs) if damage_confs else 0
            
            # 关联证据检测
            linkage_detected = any(mm.get("contains_order_or_merchant_linkage", False) for mm in valid_results)
            linkage_confs = [mm.get("linkage_confidence", 0) for mm in valid_results if mm.get("contains_order_or_merchant_linkage")]
            mm_signals["linkage_detected"] = linkage_detected
            mm_signals["linkage_confidence_avg"] = sum(linkage_confs) / len(linkage_confs) if linkage_confs else 0
            
            # 实景判断平均
            real_scene_count = sum(1 for mm in valid_results if mm.get("is_real_scene_likely", True))
            mm_signals["real_scene_avg"] = real_scene_count / len(valid_results)
            
            # 商品匹配度平均
            product_matches = [mm.get("product_match_confidence", 0.5) for mm in valid_results]
            mm_signals["product_match_avg"] = sum(product_matches) / len(product_matches) if product_matches else 0.5
            
            # 类型一致性
            type_match_count = sum(
                1 for mm in valid_results 
                if mm.get("declared_shot_type") == mm.get("predicted_shot_type")
            )
            mm_signals["type_consistency_rate"] = type_match_count / len(valid_results) if valid_results else 1.0
            
            # 关联物匹配度平均（damage_with_order_link 和 merchant_label_closeup 类型）
            ref_linkage_matches = [
                mm.get("reference_linkage_confidence", 0) 
                for mm in valid_results 
                if mm.get("predicted_shot_type") in ["damage_with_order_link", "merchant_label_closeup"]
            ]
            mm_signals["reference_linkage_avg"] = sum(ref_linkage_matches) / len(ref_linkage_matches) if ref_linkage_matches else 0
            
            # auxiliary 类型：声明支持率
            auxiliary_results = [mm for mm in valid_results if mm.get("predicted_shot_type") == "auxiliary"]
            if auxiliary_results:
                supported_count = sum(
                    1 for mm in auxiliary_results 
                    if mm.get("claim_support_level") == "supported"
                )
                partial_count = sum(
                    1 for mm in auxiliary_results 
                    if mm.get("claim_support_level") == "partially_supported"
                )
                # 完全支持计1，部分支持计0.5
                mm_signals["claim_supported_rate"] = (supported_count + 0.5 * partial_count) / len(auxiliary_results)
            
            # merchant_label_closeup 类型：标签可读率
            label_results = [mm for mm in valid_results if mm.get("predicted_shot_type") == "merchant_label_closeup"]
            if label_results:
                readable_count = sum(
                    1 for mm in label_results 
                    if mm.get("label_readability") == "清晰"
                )
                mm_signals["label_readable_rate"] = readable_count / len(label_results)
            
            # overview 类型：整体视图率
            overview_results = [mm for mm in valid_results if mm.get("predicted_shot_type") == "overview"]
            if overview_results:
                overall_count = sum(
                    1 for mm in overview_results 
                    if mm.get("shows_overall_product_view", False)
                )
                mm_signals["overall_view_rate"] = overall_count / len(overview_results)
        
        # ==================== 多模态调整逻辑 ====================
        
        # 1. 损坏证据调整
        if mm_signals["damage_detected"] and mm_signals["damage_confidence_avg"] > 0.7:
            mm_adjustment += 0.02
            fusion_reason.append("检测到明确损坏证据")
        elif mm_signals["damage_detected"] and mm_signals["damage_confidence_avg"] > 0.5:
            mm_adjustment += 0.01
            fusion_reason.append("检测到损坏证据（中等置信度）")
        elif not mm_signals["damage_detected"]:
            # 未检测到损坏证据，小幅减分
            mm_adjustment -= 0.02
            fusion_reason.append("未检测到明确损坏证据")
        
        # 2. 关联证据调整
        if mm_signals["linkage_detected"] and mm_signals["linkage_confidence_avg"] > 0.7:
            mm_adjustment += 0.02
            fusion_reason.append("检测到订单/商家关联证据")
        elif not mm_signals["linkage_detected"]:
            # 未检测到关联证据，小幅减分
            mm_adjustment -= 0.01
            fusion_reason.append("未检测到订单/商家关联证据")
        
        # 3. 实景判断调整
        if mm_signals["real_scene_avg"] < 0.5:
            mm_adjustment -= 0.03
            fusion_reason.append("部分图像疑似非实拍")
        elif mm_signals["real_scene_avg"] >= 0.8:
            mm_adjustment += 0.01
            fusion_reason.append("图像更像实拍场景")
        
        # 4. 商品匹配度调整
        if mm_signals["product_match_avg"] > 0.7:
            mm_adjustment += 0.02
            fusion_reason.append("商品与参考图匹配较高")
        elif mm_signals["product_match_avg"] < 0.3:
            mm_adjustment -= 0.03
            fusion_reason.append("商品与参考图匹配较低")
        
        # 5. 类型一致性调整
        if mm_signals["type_consistency_rate"] < 0.5:
            mm_adjustment -= 0.02
            fusion_reason.append("部分图像类型与声明不一致")
        
        # 6. auxiliary 类型：声明支持率调整
        if mm_signals["claim_supported_rate"] < 0.5:
            mm_adjustment -= 0.02
            fusion_reason.append("辅助图声明支持率较低")
        elif mm_signals["claim_supported_rate"] >= 0.8:
            mm_adjustment += 0.01
            fusion_reason.append("辅助图声明得到支持")
        
        # 7. merchant_label_closeup 类型：标签可读率调整
        if mm_signals["label_readable_rate"] < 0.5:
            mm_adjustment -= 0.01
            fusion_reason.append("标签可读性较低")
        
        # 8. overview 类型：整体视图率调整
        if mm_signals["overall_view_rate"] < 0.5:
            mm_adjustment -= 0.01
            fusion_reason.append("部分整体图未展示商品全貌")
        
        # 9. 关联物匹配度调整（有参考图时）
        if mm_signals["reference_linkage_avg"] > 0.7:
            mm_adjustment += 0.02
            fusion_reason.append("关联证据与参考图匹配较高")
        elif mm_signals["reference_linkage_avg"] > 0 and mm_signals["reference_linkage_avg"] < 0.3:
            mm_adjustment -= 0.02
            fusion_reason.append("关联证据与参考图匹配较低")
        
        # 限制多模态调整幅度（最多 ±0.10）
        mm_adjustment = max(-0.10, min(0.10, mm_adjustment))
    
    # ==================== AI 生成/真实性检测信号融合 ====================
    ai_adjustment = 0.0
    ai_signals = {
        "total_analyzed": 0,
        "success_count": 0,
        "ai_suspect_count": 0,
        "ai_suspect_rate": 0.0,
        "ai_score_avg": 0.0,
        "high_risk_count": 0,
        "key_image_high_risk_count": 0,
    }
    
    if ai_results and len(ai_results) > 0:
        valid_ai = [x for x in ai_results if x.get("ai_analysis_success", True)]
        ai_signals["total_analyzed"] = len(ai_results)
        ai_signals["success_count"] = len(valid_ai)
        
        if valid_ai:
            # 疑似 AI 生成的图像
            suspect_list = [x for x in valid_ai if x.get("is_ai_generated_suspected", False)]
            ai_scores = [x.get("ai_gen_score", 0.0) for x in valid_ai]
            
            ai_signals["ai_suspect_count"] = len(suspect_list)
            ai_signals["ai_suspect_rate"] = len(suspect_list) / len(valid_ai)
            ai_signals["ai_score_avg"] = sum(ai_scores) / len(ai_scores)
            
            # 高风险图像
            high_risk = [x for x in valid_ai if x.get("forgery_risk_level") == "high"]
            ai_signals["high_risk_count"] = len(high_risk)
            
            # 核心证据图（damage_closeup, damage_with_order_link）高风险数
            key_types = {"damage_closeup", "damage_with_order_link"}
            key_high_risk = [
                x for x in high_risk
                if x.get("declared_shot_type") in key_types
            ]
            ai_signals["key_image_high_risk_count"] = len(key_high_risk)
        
        # ==================== AI 检测调整逻辑 ====================
        
        # 1. 核心证据图存在高风险 AIGC 嫌疑
        if ai_signals["key_image_high_risk_count"] >= 2:
            ai_adjustment -= 0.08
            fusion_reason.append("核心退款证据图均存在较强AI生成嫌疑")
        elif ai_signals["key_image_high_risk_count"] == 1:
            ai_adjustment -= 0.05
            fusion_reason.append("关键退款图存在AI生成嫌疑")
        
        # 2. 整体可疑比例高
        if ai_signals["ai_suspect_rate"] >= 0.7:
            ai_adjustment -= 0.04
            fusion_reason.append("多数退款图存在AI生成嫌疑")
        elif ai_signals["ai_suspect_rate"] >= 0.4:
            ai_adjustment -= 0.02
            fusion_reason.append("部分退款图存在AI生成嫌疑")
        
        # 3. 平均分过高
        if ai_signals["ai_score_avg"] >= 0.8:
            ai_adjustment -= 0.04
            fusion_reason.append("图像整体生成痕迹较强")
        elif ai_signals["ai_score_avg"] >= 0.6:
            ai_adjustment -= 0.02
            fusion_reason.append("图像存在一定生成痕迹")
        
        # 4. 如果 AI 检测大量失败，不直接判死，提示人工复核
        if ai_signals["total_analyzed"] > 0 and ai_signals["success_count"] / ai_signals["total_analyzed"] < 0.5:
            fusion_reason.append("AI真伪检测成功率较低，建议人工复核")
        
        # 限制 AI 调整幅度（最多 -0.15）
        ai_adjustment = max(-0.15, min(0.0, ai_adjustment))
    
    # ==================== 计算最终置信度 ====================
    confidence = clamp01(base_confidence + mm_adjustment + ai_adjustment)
    
    # ==================== 确定最终状态 ====================
    # 根据置信度调整状态
    if confidence >= 0.65:
        final_status = "pass"
    elif confidence >= 0.45:
        final_status = "warn"
    else:
        final_status = "fail"
    
    # 但如果必传图未满足，强制 fail
    if not required_passed:
        final_status = "fail"
    
    # ==================== 映射风险等级 ====================
    risk_mapping = thresholds.get("risk", {}).get("status_to_risk", {
        "pass": "low",
        "warn": "medium",
        "fail": "high"
    })
    risk = risk_mapping.get(final_status, "high")
    
    # ==================== 构建输出 ====================
    result = {
        "object_match": final_status,
        "confidence_score": round(confidence, 4),
        "image_risk_level": risk,
        "fusion_reason": fusion_reason,
    }
    
    # 记录多模态调整
    if mm_results:
        result["mm_adjustment"] = round(mm_adjustment, 4)
        result["mm_signals"] = mm_signals
    
    # 记录 AI 检测调整
    if ai_results:
        result["ai_adjustment"] = round(ai_adjustment, 4)
        result["ai_signals"] = ai_signals
    
    # 记录旧链对照结果（如有）
    if obj:
        result["legacy_object_match"] = {
            "status": obj.get("status"),
            "score": obj.get("score"),
        }
    
    return result
