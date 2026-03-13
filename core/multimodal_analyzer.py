"""
core/multimodal_analyzer.py

多模态模型分析器模块
负责：构造 API 请求、调用多模态模型、解析返回、输出统一结构

定位：大模型负责"看懂图"，规则层负责"怎么判"
"""

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

from core.multimodal_prompts import (
    SYSTEM_PROMPT,
    SHOT_TYPE_DEFINITIONS,
    build_single_image_prompt,
    build_reference_comparison_prompt,
    parse_model_output,
    validate_single_image_output,
    validate_reference_comparison_output,
)


# 允许的 shot_type 枚举
VALID_SHOT_TYPES = set(SHOT_TYPE_DEFINITIONS.keys())

# 允许的 claim_support_level 枚举
VALID_CLAIM_SUPPORT_LEVELS = {"supported", "partially_supported", "not_supported"}

# 允许的 label_readability 枚举
VALID_LABEL_READABILITY = {"清晰", "模糊", "遮挡", "过曝", "不可读"}


def _encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_image_media_type(image_path: str) -> str:
    """获取图片的 MIME 类型"""
    suffix = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return media_types.get(suffix, "image/jpeg")


def _build_analysis_prompt(
    product_name: str,
    refund_reason: str,
    declared_shot_type: str,
    has_product_reference: bool = False,
    has_linkage_reference: bool = False,
    user_claim: str = "",
) -> str:
    """
    构建分析提示词 - 调用 multimodal_prompts.py
    
    如果有参考图，使用 reference_comparison prompt
    否则使用 single_image prompt
    """
    if has_product_reference or has_linkage_reference:
        return build_reference_comparison_prompt(
            declared_shot_type=declared_shot_type,
            product_name=product_name,
            refund_reason=refund_reason,
            has_product_reference=has_product_reference,
            has_linkage_reference=has_linkage_reference,
            user_claim=user_claim,
        )
    else:
        return build_single_image_prompt(
            declared_shot_type=declared_shot_type,
            product_name=product_name,
            refund_reason=refund_reason,
            user_claim=user_claim,
        )


def _build_messages(
    prompt: str,
    refund_image_base64: str,
    refund_image_media_type: str,
    product_reference_images: List[Dict],
    order_link_reference_images: List[Dict],
) -> List[Dict]:
    """构建 API 请求的 messages"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ]
        }
    ]
    
    # 添加退款图
    messages[0]["content"].append({
        "type": "image_url",
        "image_url": {
            "url": f"data:{refund_image_media_type};base64,{refund_image_base64}"
        }
    })
    
    # 添加商品主体参考图（最多2张）
    for ref_img in product_reference_images[:2]:
        ref_base64 = ref_img.get("base64")
        ref_media_type = ref_img.get("media_type")
        if ref_base64 and ref_media_type:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{ref_media_type};base64,{ref_base64}"
                }
            })
    
    # 添加包装/标签参考图（最多2张）
    for ref_img in order_link_reference_images[:2]:
        ref_base64 = ref_img.get("base64")
        ref_media_type = ref_img.get("media_type")
        if ref_base64 and ref_media_type:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{ref_media_type};base64,{ref_base64}"
                }
            })
    
    return messages


def _call_multimodal_api(
    messages: List[Dict],
    api_config: Dict,
) -> str:
    """调用多模态 API"""
    provider = api_config.get("provider", "openai_compatible")
    api_base = api_config.get("api_base", "")
    api_key = api_config.get("api_key", "")
    model_name = api_config.get("model_name", "gpt-4o-mini")
    timeout = api_config.get("timeout_sec", 60)
    
    if not api_base or not api_key:
        raise ValueError("多模态 API 配置不完整：缺少 api_base 或 api_key")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 2000,  # 增加 token 限制，避免返回被截断
        "temperature": 0.1,
    }
    
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    
    if response.status_code != 200:
        raise ValueError(f"API 调用失败：{response.status_code} - {response.text}")
    
    result = response.json()
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # 调试：检查是否有 finish_reason 为 "length"（表示被截断）
    finish_reason = result.get("choices", [{}])[0].get("finish_reason", "")
    if finish_reason == "length":
        print(f"[WARNING] API 返回被截断（finish_reason=length），考虑增加 max_tokens")
    
    return content


def _parse_api_response(content: str) -> Dict:
    """解析 API 返回的 JSON，支持截断 JSON 的容错处理"""
    import re
    
    if not content or not content.strip():
        raise ValueError("API 返回空内容")
    
    # 0. 清理内容：移除可能的 markdown 代码块标记
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # 1. 尝试直接解析
    try:
        result = json.loads(cleaned)
        return result
    except json.JSONDecodeError:
        pass
    
    # 2. 尝试提取完整 JSON 块（贪婪匹配最后一个完整的 }）
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # 3. 尝试修复截断的 JSON
    truncated = cleaned
    
    # 尝试找到最后一个完整的字段
    complete_field_patterns = [
        (r'"predicted_shot_type"\s*:\s*"([^"]+)"', "predicted_shot_type"),
        (r'"contains_product"\s*:\s*(true|false)', "contains_product"),
        (r'"contains_damage"\s*:\s*(true|false)', "contains_damage"),
        (r'"damage_confidence"\s*:\s*([\d.]+)', "damage_confidence"),
        (r'"contains_order_or_merchant_linkage"\s*:\s*(true|false)', "contains_order_or_merchant_linkage"),
        (r'"linkage_confidence"\s*:\s*([\d.]+)', "linkage_confidence"),
        (r'"is_real_scene_likely"\s*:\s*(true|false)', "is_real_scene_likely"),
        (r'"real_scene_confidence"\s*:\s*([\d.]+)', "real_scene_confidence"),
        (r'"product_match_confidence"\s*:\s*([\d.]+)', "product_match_confidence"),
        (r'"reference_linkage_confidence"\s*:\s*([\d.]+)', "reference_linkage_confidence"),
        (r'"claim_support_confidence"\s*:\s*([\d.]+)', "claim_support_confidence"),
        (r'"shows_overall_product_view"\s*:\s*(true|false)', "shows_overall_product_view"),
        (r'"overall_view_confidence"\s*:\s*([\d.]+)', "overall_view_confidence"),
    ]
    
    result = {}
    
    # 提取所有能匹配的字段
    for pattern, field_name in complete_field_patterns:
        match = re.search(pattern, cleaned)
        if match:
            if field_name in ["predicted_shot_type"]:
                result[field_name] = match.group(1)
            elif field_name in ["contains_product", "contains_damage", "contains_order_or_merchant_linkage", "is_real_scene_likely", "shows_overall_product_view"]:
                result[field_name] = match.group(1).lower() == "true"
            else:
                try:
                    result[field_name] = float(match.group(1))
                except ValueError:
                    pass
    
    # 提取 claim_support_level
    claim_match = re.search(r'"claim_support_level"\s*:\s*"([^"]+)"', cleaned)
    if claim_match:
        result["claim_support_level"] = claim_match.group(1)
    
    # 提取 label_readability
    label_match = re.search(r'"label_readability"\s*:\s*"([^"]+)"', cleaned)
    if label_match:
        result["label_readability"] = label_match.group(1)
    
    # 提取字符串字段
    string_fields = ["summary", "damage_description", "linkage_description", "real_scene_risk_reason", 
                     "claim_support_reason", "observed_auxiliary_evidence", "current_condition_description", "user_claim"]
    for field in string_fields:
        field_match = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)', cleaned)
        if field_match:
            result[field] = field_match.group(1)
    
    # 如果提取到了有效字段，返回结果
    if result:
        return result
    
    # 4. 尝试补全括号后解析
    open_braces = truncated.count('{') - truncated.count('}')
    if open_braces > 0:
        truncated = truncated + '}' * open_braces
    
    try:
        result = json.loads(truncated)
        return result
    except json.JSONDecodeError:
        pass
    
    # 5. 最后尝试：截断到最后一个完整的键值对
    last_brace = truncated.rfind('}')
    
    if last_brace > 0:
        try_truncated = truncated[:last_brace + 1]
        open_braces = try_truncated.count('{') - try_truncated.count('}')
        if open_braces > 0:
            try_truncated = try_truncated + '}' * open_braces
        try:
            result = json.loads(try_truncated)
            return result
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"无法解析 API 返回的 JSON（可能被截断）：{content[:500]}")


def _validate_analysis_result(result: Dict, declared_shot_type: str = "auxiliary") -> Dict:
    """验证并补全分析结果 - 根据不同的 shot_type 使用不同的验证逻辑"""
    
    # 确保 predicted_shot_type 有效
    predicted = result.get("predicted_shot_type", "auxiliary")
    if predicted not in VALID_SHOT_TYPES:
        predicted = "auxiliary"
    
    result["predicted_shot_type"] = predicted
    
    # ==================== 通用字段（所有类型） ====================
    # 布尔值字段
    bool_fields = [
        "contains_product",
        "is_real_scene_likely",
    ]
    for field in bool_fields:
        if field not in result:
            result[field] = False
        elif not isinstance(result[field], bool):
            result[field] = bool(result[field])
    
    # 置信度字段
    confidence_fields = [
        "real_scene_confidence",
    ]
    for field in confidence_fields:
        if field not in result:
            result[field] = 0.5
        else:
            try:
                val = float(result[field])
                result[field] = max(0.0, min(1.0, val))
            except (TypeError, ValueError):
                result[field] = 0.5
    
    # 字符串字段
    if "real_scene_risk_reason" not in result:
        result["real_scene_risk_reason"] = ""
    elif not isinstance(result["real_scene_risk_reason"], str):
        result["real_scene_risk_reason"] = str(result["real_scene_risk_reason"]) if result["real_scene_risk_reason"] else ""
    
    if "summary" not in result or not result["summary"]:
        result["summary"] = "模型未提供摘要"
    
    # ==================== 根据类型验证专属字段 ====================
    
    if predicted == "damage_closeup":
        # damage_closeup 专属字段
        result = _validate_damage_closeup_fields(result)
    
    elif predicted == "damage_with_order_link":
        # damage_with_order_link 专属字段
        result = _validate_damage_with_order_link_fields(result)
    
    elif predicted == "overview":
        # overview 专属字段
        result = _validate_overview_fields(result)
    
    elif predicted == "merchant_label_closeup":
        # merchant_label_closeup 专属字段
        result = _validate_merchant_label_closeup_fields(result)
    
    elif predicted == "auxiliary":
        # auxiliary 专属字段
        result = _validate_auxiliary_fields(result)
    
    return result


def _validate_damage_closeup_fields(result: Dict) -> Dict:
    """验证 damage_closeup 专属字段"""
    # 布尔值
    if "contains_damage" not in result:
        result["contains_damage"] = False
    elif not isinstance(result["contains_damage"], bool):
        result["contains_damage"] = bool(result["contains_damage"])
    
    # 置信度
    for field in ["damage_confidence", "product_match_confidence"]:
        if field not in result:
            result[field] = 0.0
        else:
            try:
                val = float(result[field])
                result[field] = max(0.0, min(1.0, val))
            except (TypeError, ValueError):
                result[field] = 0.0
    
    # 字符串
    if "damage_description" not in result:
        result["damage_description"] = ""
    elif not isinstance(result["damage_description"], str):
        result["damage_description"] = str(result["damage_description"]) if result["damage_description"] else ""
    
    return result


def _validate_damage_with_order_link_fields(result: Dict) -> Dict:
    """验证 damage_with_order_link 专属字段"""
    # 布尔值
    for field in ["contains_damage", "contains_order_or_merchant_linkage"]:
        if field not in result:
            result[field] = False
        elif not isinstance(result[field], bool):
            result[field] = bool(result[field])
    
    # 置信度
    for field in ["damage_confidence", "linkage_confidence", "product_match_confidence", "reference_linkage_confidence"]:
        if field not in result:
            result[field] = 0.0
        else:
            try:
                val = float(result[field])
                result[field] = max(0.0, min(1.0, val))
            except (TypeError, ValueError):
                result[field] = 0.0
    
    # 字符串
    for field in ["damage_description", "linkage_description"]:
        if field not in result:
            result[field] = ""
        elif not isinstance(result[field], str):
            result[field] = str(result[field]) if result[field] else ""
    
    return result


def _validate_overview_fields(result: Dict) -> Dict:
    """验证 overview 专属字段"""
    # 布尔值
    if "shows_overall_product_view" not in result:
        result["shows_overall_product_view"] = False
    elif not isinstance(result["shows_overall_product_view"], bool):
        result["shows_overall_product_view"] = bool(result["shows_overall_product_view"])
    
    # 置信度
    for field in ["overall_view_confidence", "product_match_confidence"]:
        if field not in result:
            result[field] = 0.0
        else:
            try:
                val = float(result[field])
                result[field] = max(0.0, min(1.0, val))
            except (TypeError, ValueError):
                result[field] = 0.0
    
    # 字符串
    if "current_condition_description" not in result:
        result["current_condition_description"] = ""
    elif not isinstance(result["current_condition_description"], str):
        result["current_condition_description"] = str(result["current_condition_description"]) if result["current_condition_description"] else ""
    
    return result


def _validate_merchant_label_closeup_fields(result: Dict) -> Dict:
    """验证 merchant_label_closeup 专属字段"""
    # 布尔值
    if "contains_order_or_merchant_linkage" not in result:
        result["contains_order_or_merchant_linkage"] = False
    elif not isinstance(result["contains_order_or_merchant_linkage"], bool):
        result["contains_order_or_merchant_linkage"] = bool(result["contains_order_or_merchant_linkage"])
    
    # 置信度
    for field in ["linkage_confidence", "reference_linkage_confidence"]:
        if field not in result:
            result[field] = 0.0
        else:
            try:
                val = float(result[field])
                result[field] = max(0.0, min(1.0, val))
            except (TypeError, ValueError):
                result[field] = 0.0
    
    # 字符串
    if "linkage_description" not in result:
        result["linkage_description"] = ""
    elif not isinstance(result["linkage_description"], str):
        result["linkage_description"] = str(result["linkage_description"]) if result["linkage_description"] else ""
    
    # label_readability 枚举值
    if "label_readability" not in result:
        result["label_readability"] = "不可读"
    elif result["label_readability"] not in VALID_LABEL_READABILITY:
        result["label_readability"] = "不可读"
    
    return result


def _validate_auxiliary_fields(result: Dict) -> Dict:
    """验证 auxiliary 专属字段"""
    # 字符串
    for field in ["user_claim", "claim_support_reason", "observed_auxiliary_evidence"]:
        if field not in result:
            result[field] = ""
        elif not isinstance(result[field], str):
            result[field] = str(result[field]) if result[field] else ""
    
    # claim_support_level 枚举值
    if "claim_support_level" not in result:
        result["claim_support_level"] = "not_supported"
    elif result["claim_support_level"] not in VALID_CLAIM_SUPPORT_LEVELS:
        result["claim_support_level"] = "not_supported"
    
    # 置信度
    if "claim_support_confidence" not in result:
        result["claim_support_confidence"] = 0.0
    else:
        try:
            val = float(result["claim_support_confidence"])
            result["claim_support_confidence"] = max(0.0, min(1.0, val))
        except (TypeError, ValueError):
            result["claim_support_confidence"] = 0.0
    
    return result


def analyze_refund_image_with_mm(
    image_path: str,
    declared_shot_type: str,
    product_name: str,
    refund_reason: str,
    product_reference_paths: List[str],
    order_link_reference_paths: List[str],
    api_config: Dict,
    user_claim: str = "",
) -> Dict:
    """
    使用多模态模型分析单张退款图
    
    Args:
        image_path: 退款图路径
        declared_shot_type: 前端声明的图片类型
        product_name: 商品名称
        refund_reason: 退款原因
        product_reference_paths: 商品主体参考图路径列表
        order_link_reference_paths: 包装/标签参考图路径列表
        api_config: API 配置（provider, api_base, api_key, model_name, timeout_sec）
        user_claim: 用户声明（仅 auxiliary 类型需要）
    
    Returns:
        结构化分析结果
    """
    # 编码退款图
    refund_image_base64 = _encode_image_to_base64(image_path)
    refund_image_media_type = _get_image_media_type(image_path)
    
    # 编码参考图
    product_reference_images = []
    for ref_path in product_reference_paths[:2]:
        try:
            ref_base64 = _encode_image_to_base64(ref_path)
            ref_media_type = _get_image_media_type(ref_path)
            product_reference_images.append({
                "base64": ref_base64,
                "media_type": ref_media_type,
                "path": ref_path,
            })
        except Exception:
            pass
    
    order_link_reference_images = []
    for ref_path in order_link_reference_paths[:2]:
        try:
            ref_base64 = _encode_image_to_base64(ref_path)
            ref_media_type = _get_image_media_type(ref_path)
            order_link_reference_images.append({
                "base64": ref_base64,
                "media_type": ref_media_type,
                "path": ref_path,
            })
        except Exception:
            pass
    
    # 构建提示词
    has_product_reference = len(product_reference_images) > 0
    has_linkage_reference = len(order_link_reference_images) > 0
    
    prompt = _build_analysis_prompt(
        product_name=product_name,
        refund_reason=refund_reason,
        declared_shot_type=declared_shot_type,
        has_product_reference=has_product_reference,
        has_linkage_reference=has_linkage_reference,
        user_claim=user_claim,
    )
    
    # 构建消息
    messages = _build_messages(
        prompt=prompt,
        refund_image_base64=refund_image_base64,
        refund_image_media_type=refund_image_media_type,
        product_reference_images=product_reference_images,
        order_link_reference_images=order_link_reference_images,
    )
    
    # 调用 API
    content = _call_multimodal_api(messages, api_config)
    
    # 解析返回
    result = _parse_api_response(content)
    
    # 验证并补全结果
    result = _validate_analysis_result(result, declared_shot_type)
    
    # 添加元信息
    result["path"] = image_path
    result["declared_shot_type"] = declared_shot_type
    
    return result


def multimodal_analyze_batch(
    refund_images: List[str],
    refund_image_meta: Dict[str, Dict],
    product_name: str,
    refund_reason: str,
    order_reference_images: List[str],
    order_reference_meta: Dict[str, Dict],
    api_config: Dict,
) -> List[Dict]:
    """
    批量分析退款图
    
    Args:
        refund_images: 退款图路径列表
        refund_image_meta: 退款图元信息字典（按路径索引，包含 shot_type 和 user_claim）
        product_name: 商品名称
        refund_reason: 退款原因
        order_reference_images: 参考图路径列表
        order_reference_meta: 参考图元信息字典（按路径索引）
        api_config: API 配置
    
    Returns:
        每张退款图的分析结果列表
    """
    # 分离参考图类型
    product_reference_paths = []
    order_link_reference_paths = []
    
    for path in order_reference_images:
        meta = order_reference_meta.get(path, {})
        ref_type = meta.get("ref_type", "product_reference")
        if ref_type == "product_reference":
            product_reference_paths.append(path)
        elif ref_type == "order_link_reference":
            order_link_reference_paths.append(path)
    
    results = []
    for image_path in refund_images:
        # 获取声明的 shot_type 和 user_claim（从字典中按路径查询）
        meta = refund_image_meta.get(image_path, {})
        declared_shot_type = meta.get("shot_type") or "auxiliary"
        user_claim = meta.get("user_claim", "")  # auxiliary 类型的用户声明
        
        try:
            result = analyze_refund_image_with_mm(
                image_path=image_path,
                declared_shot_type=declared_shot_type,
                product_name=product_name,
                refund_reason=refund_reason,
                product_reference_paths=product_reference_paths,
                order_link_reference_paths=order_link_reference_paths,
                api_config=api_config,
                user_claim=user_claim,
            )
            result["analysis_success"] = True
            result["error"] = None
        except Exception as e:
            result = _get_default_error_result(image_path, declared_shot_type, str(e))
        
        results.append(result)
    
    return results


def _get_default_error_result(image_path: str, declared_shot_type: str, error_msg: str) -> Dict:
    """生成分析失败时的默认结果"""
    return {
        "path": image_path,
        "declared_shot_type": declared_shot_type,
        "predicted_shot_type": "auxiliary",
        "analysis_success": False,
        "error": error_msg,
        # 通用字段
        "contains_product": False,
        "is_real_scene_likely": True,
        "real_scene_confidence": 0.5,
        "real_scene_risk_reason": "",
        "summary": f"分析失败：{error_msg}",
        # damage_closeup / damage_with_order_link 字段
        "contains_damage": False,
        "damage_confidence": 0.0,
        "damage_description": "",
        # damage_with_order_link / merchant_label_closeup 字段
        "contains_order_or_merchant_linkage": False,
        "linkage_confidence": 0.0,
        "linkage_description": "",
        # 参考图匹配字段
        "product_match_confidence": 0.0,
        "reference_linkage_confidence": 0.0,
        # overview 字段
        "shows_overall_product_view": False,
        "overall_view_confidence": 0.0,
        "current_condition_description": "",
        # merchant_label_closeup 字段
        "label_readability": "不可读",
        # auxiliary 字段
        "user_claim": "",
        "claim_support_level": "not_supported",
        "claim_support_confidence": 0.0,
        "claim_support_reason": "",
        "observed_auxiliary_evidence": "",
    }