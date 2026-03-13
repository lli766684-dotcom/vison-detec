"""
多模态分析专用 Prompt 模块

职责：
- 定义模型身份
- 5 套独立的 prompt 和 schema（按图像类型）
- 输出 schema 约束
"""

from typing import Dict, List, Optional
import json


# ==================== System Prompt ====================

SYSTEM_PROMPT = """你是一个专业的电商退款图像证据分析器。

你的职责是：
1. 分析退款图片的内容和类型
2. 检测是否存在损坏证据
3. 判断是否存在订单/商家关联证据
4. 评估图片真实性（是否像实拍）
5. 与参考图进行对比分析

你的限制：
- 只做结构化图像理解，输出 JSON 格式结果
- 不直接输出最终退款裁决
- 不做主观判断，只描述观察到的内容
- 置信度基于视觉证据，不凭空猜测"""


# ==================== 图像类型定义 ====================

SHOT_TYPE_DEFINITIONS = {
    "damage_closeup": "商品损坏细节图 - 展示腐坏、裂痕、破损、渗漏、变形等问题部位的近景图",
    "damage_with_order_link": "商品与订单/商家关联图 - 同时包含问题商品和订单/商家来源信息（包装、标签、贴纸、面单、小票等）",
    "overview": "商品整体图 - 展示商品整体外观的图片",
    "merchant_label_closeup": "商家标签/包装近景图 - 展示包装标签、店铺标识、贴纸、面单等商家关联信息的近景图",
    "auxiliary": "其他辅助图 - 运输外箱、数量情况、整体摆放环境等辅助说明图",
}


# ==================== 5 套独立 Schema ====================

# 1. damage_closeup（损坏细节图）
DAMAGE_CLOSEUP_SCHEMA = {
    "predicted_shot_type": "string - 图像类型，固定为 damage_closeup",
    "contains_product": "boolean - 是否包含商品主体（商品本身或其一部分）",
    "contains_damage": "boolean - 是否包含损坏的商品",
    "damage_confidence": "float - 损坏检测置信度，范围 0.0-1.0",
    "damage_description": "string - 损坏情况的详细描述（如：发霉、破损、渗漏等）",
    "is_real_scene_likely": "boolean - 是否像实景拍摄（非网图、非截图、非海报）",
    "real_scene_confidence": "float - 实拍置信度，范围 0.0-1.0",
    "real_scene_risk_reason": "string - 如果怀疑非实拍，说明原因（如：光线不自然、背景像合成、边缘有痕迹等）",
    "product_match_confidence": "float - 与商品主体参考图的匹配度，范围 0.0-1.0（无参考图时填 0）",
    "summary": "string - 详细描述图片内容，包括商品状态、损坏情况等，不超过150字",
}

# 2. damage_with_order_link（关联图）
DAMAGE_WITH_ORDER_LINK_SCHEMA = {
    "predicted_shot_type": "string - 图像类型，固定为 damage_with_order_link",
    "contains_product": "boolean - 是否包含商品主体（商品本身或其一部分）",
    "contains_damage": "boolean - 是否包含损坏的商品",
    "damage_confidence": "float - 损坏检测置信度，范围 0.0-1.0",
    "damage_description": "string - 损坏情况的详细描述",
    "contains_order_or_merchant_linkage": "boolean - 是否包含订单/商家关联证据（包装、标签、贴纸、面单、小票、订单号等）",
    "linkage_confidence": "float - 关联证据置信度，范围 0.0-1.0",
    "linkage_description": "string - 关联证据的详细描述（如：快递面单、商家标签、订单小票等）",
    "is_real_scene_likely": "boolean - 是否像实景拍摄",
    "real_scene_confidence": "float - 实拍置信度，范围 0.0-1.0",
    "real_scene_risk_reason": "string - 如果怀疑非实拍，说明原因",
    "product_match_confidence": "float - 与商品主体参考图的匹配度，范围 0.0-1.0",
    "reference_linkage_confidence": "float - 与包装/标签参考图的关联度，范围 0.0-1.0",
    "summary": "string - 详细描述图片内容，不超过150字",
}

# 3. overview（整体图）
OVERVIEW_SCHEMA = {
    "predicted_shot_type": "string - 图像类型，固定为 overview",
    "shows_overall_product_view": "boolean - 是否展示商品的整体视图（能看到商品全貌）",
    "overall_view_confidence": "float - 整体视图置信度，范围 0.0-1.0",
    "contains_product": "boolean - 是否包含商品主体",
    "current_condition_description": "string - 商品当前状态的描述（如：完好、有损坏痕迹、包装完整等）",
    "is_real_scene_likely": "boolean - 是否像实景拍摄",
    "real_scene_confidence": "float - 实拍置信度，范围 0.0-1.0",
    "real_scene_risk_reason": "string - 如果怀疑非实拍，说明原因",
    "product_match_confidence": "float - 与商品主体参考图的匹配度，范围 0.0-1.0",
    "summary": "string - 详细描述图片内容，不超过150字",
}

# 4. merchant_label_closeup（商家标签近景图）
MERCHANT_LABEL_CLOSEUP_SCHEMA = {
    "predicted_shot_type": "string - 图像类型，固定为 merchant_label_closeup",
    "contains_order_or_merchant_linkage": "boolean - 是否包含订单/商家关联证据（标签、贴纸、面单等）",
    "linkage_confidence": "float - 关联证据置信度，范围 0.0-1.0",
    "linkage_description": "string - 关联证据的详细描述",
    "label_readability": "string - 标签可读性，枚举值：清晰 / 模糊 / 遮挡 / 过曝 / 不可读",
    "is_real_scene_likely": "boolean - 是否像实景拍摄（标签是否真实贴在实物上）",
    "real_scene_confidence": "float - 实拍置信度，范围 0.0-1.0",
    "real_scene_risk_reason": "string - 如果怀疑非实拍，说明原因",
    "reference_linkage_confidence": "float - 与包装/标签参考图的关联度，范围 0.0-1.0",
    "summary": "string - 详细描述图片内容，不超过150字",
}

# 5. auxiliary（辅助图）
AUXILIARY_SCHEMA = {
    "predicted_shot_type": "string - 图像类型，固定为 auxiliary",
    "user_claim": "string - 用户声称这张图要证明的内容",
    "claim_support_level": "string - 图片对用户声明的支持程度，枚举值：supported / partially_supported / not_supported",
    "claim_support_confidence": "float - 支持程度置信度，范围 0.0-1.0",
    "claim_support_reason": "string - 支持或不支持的原因说明",
    "observed_auxiliary_evidence": "string - 图片中观察到的辅助证据描述",
    "is_real_scene_likely": "boolean - 是否像实景拍摄",
    "real_scene_confidence": "float - 实拍置信度，范围 0.0-1.0",
    "real_scene_risk_reason": "string - 如果怀疑非实拍，说明原因",
    "summary": "string - 详细描述图片内容，不超过150字",
}


# ==================== 5 套独立 Prompt 构建函数 ====================

def build_damage_closeup_prompt(
    product_name: str = "",
    refund_reason: str = "",
    has_product_reference: bool = False,
) -> str:
    """构建 damage_closeup 类型的 prompt"""
    
    ref_instruction = ""
    if has_product_reference:
        ref_instruction = """
【参考图说明】
- 商品主体参考图：订单商品的正常外观参考，用于判断退款图中的商品是否与订单商品一致
- 请对比退款图中的商品与参考图中的商品外观，判断一致性
"""
    
    return f"""请分析这张退款图片，返回结构化的 JSON 结果。

【图像类型】
这是一张「商品损坏细节图」，用于展示腐坏、裂痕、破损、渗漏、变形等问题部位。

【商品信息】
- 商品名称：{product_name or '未知'}
- 退款原因：{refund_reason or '未知'}
{ref_instruction}
【分析任务】
请重点分析以下内容：
1. 判断是否包含商品主体（contains_product）- 商品本身或其一部分
2. 检测是否包含损坏（contains_damage, damage_confidence）
3. 详细描述损坏情况（damage_description）- 如发霉、破损、渗漏、变形等
4. 判断是否像实景拍摄（is_real_scene_likely, real_scene_confidence）
5. 如果怀疑非实拍，说明原因（real_scene_risk_reason）
6. 如有参考图，判断商品一致性（product_match_confidence）
7. 详细描述图片内容（summary）

【重要说明】
- contains_product：如果图片中能看到商品（如蓝莓、苹果等），即使损坏也应为 true
- 不要检测关联物，这不是关联图的任务
- real_scene_confidence：0 表示肯定是假图，1 表示肯定是实拍
- real_scene_risk_reason：只有当 is_real_scene_likely 为 false 或 real_scene_confidence < 0.7 时才需要填写

【输出格式】
请严格返回以下 JSON 格式，不要添加任何其他内容：
{{
  "predicted_shot_type": "damage_closeup",
  "contains_product": true或false,
  "contains_damage": true或false,
  "damage_confidence": 0.0到1.0之间的数值,
  "damage_description": "损坏情况的详细描述",
  "is_real_scene_likely": true或false,
  "real_scene_confidence": 0.0到1.0之间的数值,
  "real_scene_risk_reason": "如果怀疑非实拍，说明原因，否则填空字符串",
  "product_match_confidence": 0.0到1.0之间的数值,
  "summary": "详细描述图片内容"
}}"""


def build_damage_with_order_link_prompt(
    product_name: str = "",
    refund_reason: str = "",
    has_product_reference: bool = False,
    has_linkage_reference: bool = False,
) -> str:
    """构建 damage_with_order_link 类型的 prompt"""
    
    ref_instructions = []
    if has_product_reference:
        ref_instructions.append("- 商品主体参考图：订单商品的正常外观参考，用于判断退款图中的商品是否与订单商品一致")
    if has_linkage_reference:
        ref_instructions.append("- 包装/标签参考图：订单/商家侧的包装、标签、贴纸、面单等参考，用于判断关联证据是否匹配")
    
    ref_section = "\n".join(ref_instructions) if ref_instructions else "无参考图"
    
    return f"""请分析这张退款图片，返回结构化的 JSON 结果。

【图像类型】
这是一张「商品与订单/商家关联图」，需要同时包含问题商品和订单/商家来源信息。

【商品信息】
- 商品名称：{product_name or '未知'}
- 退款原因：{refund_reason or '未知'}

【参考图说明】
{ref_section}

【分析任务】
请重点分析以下内容：
1. 判断是否包含商品主体（contains_product）
2. 检测是否包含损坏（contains_damage, damage_confidence, damage_description）
3. 检测是否包含订单/商家关联证据（contains_order_or_merchant_linkage, linkage_confidence）
   - 关联证据包括：包装、标签、贴纸、面单、小票、订单号、商家标识等
4. 详细描述关联证据（linkage_description）
5. 判断是否像实景拍摄（is_real_scene_likely, real_scene_confidence, real_scene_risk_reason）
6. 与参考图对比（product_match_confidence, reference_linkage_confidence）
7. 详细描述图片内容（summary）

【重要说明】
- contains_product：如果图片中能看到商品，即使损坏也应为 true
- 关联证据：任何能证明商品与订单/商家有关的信息
- real_scene_risk_reason：只有当怀疑非实拍时才需要填写

【输出格式】
请严格返回以下 JSON 格式，不要添加任何其他内容：
{{
  "predicted_shot_type": "damage_with_order_link",
  "contains_product": true或false,
  "contains_damage": true或false,
  "damage_confidence": 0.0到1.0之间的数值,
  "damage_description": "损坏情况的详细描述",
  "contains_order_or_merchant_linkage": true或false,
  "linkage_confidence": 0.0到1.0之间的数值,
  "linkage_description": "关联证据的详细描述",
  "is_real_scene_likely": true或false,
  "real_scene_confidence": 0.0到1.0之间的数值,
  "real_scene_risk_reason": "如果怀疑非实拍，说明原因",
  "product_match_confidence": 0.0到1.0之间的数值,
  "reference_linkage_confidence": 0.0到1.0之间的数值,
  "summary": "详细描述图片内容"
}}"""


def build_overview_prompt(
    product_name: str = "",
    refund_reason: str = "",
    has_product_reference: bool = False,
) -> str:
    """构建 overview 类型的 prompt"""
    
    ref_instruction = ""
    if has_product_reference:
        ref_instruction = """
【参考图说明】
- 商品主体参考图：订单商品的正常外观参考，用于判断退款图中的商品是否与订单商品一致
"""
    
    return f"""请分析这张退款图片，返回结构化的 JSON 结果。

【图像类型】
这是一张「商品整体图」，用于展示商品的整体外观。

【商品信息】
- 商品名称：{product_name or '未知'}
- 退款原因：{refund_reason or '未知'}
{ref_instruction}
【分析任务】
请重点分析以下内容：
1. 判断是否展示商品的整体视图（shows_overall_product_view, overall_view_confidence）
   - 整体视图：能看到商品全貌，而非局部特写
2. 判断是否包含商品主体（contains_product）
3. 描述商品当前状态（current_condition_description）- 如完好、有损坏痕迹、包装完整等
4. 判断是否像实景拍摄（is_real_scene_likely, real_scene_confidence, real_scene_risk_reason）
5. 如有参考图，判断商品一致性（product_match_confidence）
6. 详细描述图片内容（summary）

【重要说明】
- overview 的重点是：整体性 + 实拍性 + 商品一致性
- 不是检测损坏细节，而是看商品整体状态
- real_scene_risk_reason：只有当怀疑非实拍时才需要填写

【输出格式】
请严格返回以下 JSON 格式，不要添加任何其他内容：
{{
  "predicted_shot_type": "overview",
  "shows_overall_product_view": true或false,
  "overall_view_confidence": 0.0到1.0之间的数值,
  "contains_product": true或false,
  "current_condition_description": "商品当前状态的描述",
  "is_real_scene_likely": true或false,
  "real_scene_confidence": 0.0到1.0之间的数值,
  "real_scene_risk_reason": "如果怀疑非实拍，说明原因",
  "product_match_confidence": 0.0到1.0之间的数值,
  "summary": "详细描述图片内容"
}}"""


def build_merchant_label_closeup_prompt(
    product_name: str = "",
    refund_reason: str = "",
    has_linkage_reference: bool = False,
) -> str:
    """构建 merchant_label_closeup 类型的 prompt"""
    
    ref_instruction = ""
    if has_linkage_reference:
        ref_instruction = """
【参考图说明】
- 包装/标签参考图：订单/商家侧的包装、标签、贴纸、面单等参考，用于判断标签是否匹配
"""
    
    return f"""请分析这张退款图片，返回结构化的 JSON 结果。

【图像类型】
这是一张「商家标签/包装近景图」，用于展示包装标签、店铺标识、贴纸、面单等商家关联信息。

【商品信息】
- 商品名称：{product_name or '未知'}
- 退款原因：{refund_reason or '未知'}
{ref_instruction}
【分析任务】
请重点分析以下内容：
1. 判断是否包含订单/商家关联证据（contains_order_or_merchant_linkage, linkage_confidence）
   - 关联证据包括：标签、贴纸、面单、小票、商家标识等
2. 详细描述关联证据（linkage_description）
3. 判断标签可读性（label_readability）- 枚举值：清晰 / 模糊 / 遮挡 / 过曝 / 不可读
4. 判断是否像实景拍摄（is_real_scene_likely, real_scene_confidence, real_scene_risk_reason）
   - 标签是否真实贴在实物上，而非后期合成
5. 如有参考图，判断关联证据匹配度（reference_linkage_confidence）
6. 详细描述图片内容（summary）

【重要说明】
- merchant_label_closeup 的重点是：标签真实性 + 可读性 + 关联性
- 不需要检测商品主体，可能只有标签没有商品
- label_readability 影响证据价值：模糊/遮挡的标签证据价值较低
- real_scene_risk_reason：只有当怀疑非实拍时才需要填写

【输出格式】
请严格返回以下 JSON 格式，不要添加任何其他内容：
{{
  "predicted_shot_type": "merchant_label_closeup",
  "contains_order_or_merchant_linkage": true或false,
  "linkage_confidence": 0.0到1.0之间的数值,
  "linkage_description": "关联证据的详细描述",
  "label_readability": "清晰或模糊或遮挡或过曝或不可读",
  "is_real_scene_likely": true或false,
  "real_scene_confidence": 0.0到1.0之间的数值,
  "real_scene_risk_reason": "如果怀疑非实拍，说明原因",
  "reference_linkage_confidence": 0.0到1.0之间的数值,
  "summary": "详细描述图片内容"
}}"""


def build_auxiliary_prompt(
    product_name: str = "",
    refund_reason: str = "",
    user_claim: str = "",
) -> str:
    """构建 auxiliary 类型的 prompt"""
    
    return f"""请分析这张退款图片，返回结构化的 JSON 结果。

【图像类型】
这是一张「辅助说明图」，用户声称这张图可以证明某些情况。

【商品信息】
- 商品名称：{product_name or '未知'}
- 退款原因：{refund_reason or '未知'}

【用户声明】
用户声称这张图要证明：{user_claim or '未提供说明'}

【分析任务】
请重点分析以下内容：
1. 判断图片内容是否支持用户的声明（claim_support_level）
   - supported：图片内容明确支持用户声明
   - partially_supported：图片内容部分支持用户声明
   - not_supported：图片内容不支持用户声明
2. 支持程度置信度（claim_support_confidence）
3. 支持或不支持的原因（claim_support_reason）
4. 描述图片中观察到的辅助证据（observed_auxiliary_evidence）
5. 判断是否像实景拍摄（is_real_scene_likely, real_scene_confidence, real_scene_risk_reason）
6. 详细描述图片内容（summary）

【重要说明】
- auxiliary 的重点是：声明-图像一致性校验
- 不需要检测商品主体，可能只拍外箱、漏液痕迹、融化冰袋等
- claim_support_level 是三级判断，不是简单的 true/false
- real_scene_risk_reason：只有当怀疑非实拍时才需要填写

【输出格式】
请严格返回以下 JSON 格式，不要添加任何其他内容：
{{
  "predicted_shot_type": "auxiliary",
  "user_claim": "{user_claim or '未提供说明'}",
  "claim_support_level": "supported或partially_supported或not_supported",
  "claim_support_confidence": 0.0到1.0之间的数值,
  "claim_support_reason": "支持或不支持的原因说明",
  "observed_auxiliary_evidence": "图片中观察到的辅助证据描述",
  "is_real_scene_likely": true或false,
  "real_scene_confidence": 0.0到1.0之间的数值,
  "real_scene_risk_reason": "如果怀疑非实拍，说明原因",
  "summary": "详细描述图片内容"
}}"""


# ==================== 统一入口函数 ====================

def build_single_image_prompt(
    declared_shot_type: str = None,
    product_name: str = "",
    refund_reason: str = "",
    user_claim: str = "",
) -> str:
    """
    构建单图分析 prompt（无参考图）
    
    Args:
        declared_shot_type: 前端声明的图像类型
        product_name: 商品名称
        refund_reason: 退款原因
        user_claim: 用户声明（仅 auxiliary 类型需要）
    
    Returns:
        完整的 prompt 字符串
    """
    if declared_shot_type == "damage_closeup":
        return build_damage_closeup_prompt(product_name, refund_reason, has_product_reference=False)
    elif declared_shot_type == "damage_with_order_link":
        return build_damage_with_order_link_prompt(product_name, refund_reason, False, False)
    elif declared_shot_type == "overview":
        return build_overview_prompt(product_name, refund_reason, has_product_reference=False)
    elif declared_shot_type == "merchant_label_closeup":
        return build_merchant_label_closeup_prompt(product_name, refund_reason, has_linkage_reference=False)
    elif declared_shot_type == "auxiliary":
        return build_auxiliary_prompt(product_name, refund_reason, user_claim)
    else:
        # 默认使用 auxiliary
        return build_auxiliary_prompt(product_name, refund_reason, user_claim)


def build_reference_comparison_prompt(
    declared_shot_type: str = None,
    product_name: str = "",
    refund_reason: str = "",
    has_product_reference: bool = False,
    has_linkage_reference: bool = False,
    user_claim: str = "",
) -> str:
    """
    构建 reference 对照 prompt（有参考图）
    
    Args:
        declared_shot_type: 前端声明的图像类型
        product_name: 商品名称
        refund_reason: 退款原因
        has_product_reference: 是否有商品主体参考图
        has_linkage_reference: 是否有包装/标签参考图
        user_claim: 用户声明（仅 auxiliary 类型需要）
    
    Returns:
        完整的 prompt 字符串
    """
    if declared_shot_type == "damage_closeup":
        return build_damage_closeup_prompt(product_name, refund_reason, has_product_reference)
    elif declared_shot_type == "damage_with_order_link":
        return build_damage_with_order_link_prompt(product_name, refund_reason, has_product_reference, has_linkage_reference)
    elif declared_shot_type == "overview":
        return build_overview_prompt(product_name, refund_reason, has_product_reference)
    elif declared_shot_type == "merchant_label_closeup":
        return build_merchant_label_closeup_prompt(product_name, refund_reason, has_linkage_reference)
    elif declared_shot_type == "auxiliary":
        # auxiliary 不需要参考图
        return build_auxiliary_prompt(product_name, refund_reason, user_claim)
    else:
        return build_auxiliary_prompt(product_name, refund_reason, user_claim)


# ==================== JSON 解析辅助 ====================

def parse_model_output(content: str) -> Dict:
    """
    解析模型输出的 JSON
    
    Args:
        content: 模型返回的原始内容
    
    Returns:
        解析后的字典
    """
    # 提取 JSON 部分
    if "```json" in content:
        json_str = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        json_str = content.split("```")[1].split("```")[0].strip()
    else:
        json_str = content.strip()
    
    return json.loads(json_str)


def validate_single_image_output(output: Dict) -> Dict:
    """
    验证并补全单图分析输出
    
    Args:
        output: 模型输出的字典
    
    Returns:
        验证后的字典（补全缺失字段）
    """
    defaults = {
        "predicted_shot_type": "auxiliary",
        "contains_product": False,
        "contains_damage": False,
        "damage_confidence": 0.0,
        "damage_description": "",
        "contains_order_or_merchant_linkage": False,
        "linkage_confidence": 0.0,
        "linkage_description": "",
        "is_real_scene_likely": True,
        "real_scene_confidence": 0.5,
        "real_scene_risk_reason": "",
        "product_match_confidence": 0.0,
        "reference_linkage_confidence": 0.0,
        "summary": "",
    }
    
    result = {}
    for key, default_value in defaults.items():
        result[key] = output.get(key, default_value)
    
    # 数值范围约束
    result["damage_confidence"] = max(0.0, min(1.0, float(result.get("damage_confidence", 0))))
    result["linkage_confidence"] = max(0.0, min(1.0, float(result.get("linkage_confidence", 0))))
    result["real_scene_confidence"] = max(0.0, min(1.0, float(result.get("real_scene_confidence", 0.5))))
    result["product_match_confidence"] = max(0.0, min(1.0, float(result.get("product_match_confidence", 0))))
    result["reference_linkage_confidence"] = max(0.0, min(1.0, float(result.get("reference_linkage_confidence", 0))))
    
    return result


def validate_reference_comparison_output(output: Dict) -> Dict:
    """
    验证并补全 reference 对照输出
    
    Args:
        output: 模型输出的字典
    
    Returns:
        验证后的字典（补全缺失字段）
    """
    return validate_single_image_output(output)


# ==================== Schema 获取函数 ====================

def get_schema_for_shot_type(shot_type: str) -> Dict:
    """
    根据图像类型获取对应的 schema
    
    Args:
        shot_type: 图像类型
    
    Returns:
        对应的 schema 字典
    """
    schemas = {
        "damage_closeup": DAMAGE_CLOSEUP_SCHEMA,
        "damage_with_order_link": DAMAGE_WITH_ORDER_LINK_SCHEMA,
        "overview": OVERVIEW_SCHEMA,
        "merchant_label_closeup": MERCHANT_LABEL_CLOSEUP_SCHEMA,
        "auxiliary": AUXILIARY_SCHEMA,
    }
    return schemas.get(shot_type, AUXILIARY_SCHEMA)