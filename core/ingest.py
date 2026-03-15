import uuid
from typing import Dict, List, Union, Set


# 允许的 shot_type 枚举
VALID_SHOT_TYPES: Set[str] = {
    "damage_closeup",
    "damage_with_order_link",
    "overview",
    "merchant_label_closeup",
    "auxiliary",
}

# 旧名称到新名称的映射（用于兼容旧数据）
SHOT_TYPE_ALIAS_MAP: Dict[str, str] = {
    "merchant_marker": "merchant_label_closeup",
    "merchant_label": "merchant_label_closeup",
    "damage_link": "damage_with_order_link",
    "damage_with_link": "damage_with_order_link",
}

# 允许的 ref_type 枚举
VALID_REF_TYPES: Set[str] = {
    "product_reference",
    "order_link_reference",
}


def _normalize_required_shot_types(required_shot_types: List[str]) -> List[str]:
    """
    标准化并校验 required_shot_types 字段
    
    - 应用别名映射（兼容旧名称）
    - 校验每个值是否合法
    - 去重并保持顺序
    """
    if not required_shot_types:
        return ["damage_closeup", "damage_with_order_link"]
    
    normalized = []
    for idx, shot_type in enumerate(required_shot_types):
        shot_type = str(shot_type).strip()
        
        # 应用别名映射
        if shot_type in SHOT_TYPE_ALIAS_MAP:
            shot_type = SHOT_TYPE_ALIAS_MAP[shot_type]
        
        if shot_type not in VALID_SHOT_TYPES:
            raise ValueError(
                f"required_shot_types[{idx}]='{shot_type}' 不合法，"
                f"允许值: {', '.join(sorted(VALID_SHOT_TYPES))}"
            )
        
        if shot_type not in normalized:
            normalized.append(shot_type)
    
    return normalized


def _normalize_refund_images(refund_images_input: Union[List[str], List[Dict]]) -> tuple:
    """
    兼容新旧格式的退款图输入，返回 (路径列表, 元信息字典)
    
    旧格式: ["a.jpg", "b.jpg"]
    新格式: [{"path": "a.jpg", "shot_type": "overview"}, ...]
    
    会进行输入校验：
    - 检查 path 是否存在
    - 检查 shot_type 是否在允许枚举中
    - 检查 auxiliary 类型是否填写 user_claim
    - 检查路径是否重复
    
    返回：
    - paths: 路径列表
    - meta_dict: 元信息字典（按路径索引，便于后续查询）
    """
    if not refund_images_input:
        return [], {}
    
    paths = []
    meta_dict = {}
    seen_paths = set()
    
    for idx, item in enumerate(refund_images_input):
        if isinstance(item, str):
            # 旧格式：纯字符串路径
            path = str(item).strip()
            if not path:
                raise ValueError(f"退款图[{idx}] 路径为空")
            if path in seen_paths:
                raise ValueError(f"退款图[{idx}] 路径重复：{path}")
            seen_paths.add(path)
            paths.append(path)
            meta_dict[path] = {"shot_type": None, "user_claim": ""}
            
        elif isinstance(item, dict):
            # 新格式：带 shot_type 的对象
            path = str(item.get("path", "")).strip()
            if not path:
                raise ValueError(f"退款图[{idx}] 缺少 path 字段或 path 为空")
            if path in seen_paths:
                raise ValueError(f"退款图[{idx}] 路径重复：{path}")
            seen_paths.add(path)
            
            shot_type = item.get("shot_type")
            if shot_type is None:
                raise ValueError(f"退款图[{idx}] 缺少 shot_type 字段")
            
            shot_type = str(shot_type).strip()
            
            # 应用别名映射（兼容旧名称）
            if shot_type in SHOT_TYPE_ALIAS_MAP:
                shot_type = SHOT_TYPE_ALIAS_MAP[shot_type]
            
            if shot_type not in VALID_SHOT_TYPES:
                raise ValueError(
                    f"退款图[{idx}] shot_type='{shot_type}' 不合法，"
                    f"允许值: {', '.join(sorted(VALID_SHOT_TYPES))}"
                )
            
            # auxiliary 类型的用户声明（必填）
            user_claim = str(item.get("user_claim", "")).strip()
            if shot_type == "auxiliary" and not user_claim:
                raise ValueError(f"退款图[{idx}] 为 auxiliary 时必须填写 user_claim")
            
            paths.append(path)
            meta_dict[path] = {
                "shot_type": shot_type,
                "user_claim": user_claim,
            }
        else:
            raise ValueError(f"退款图[{idx}] 格式错误，应为字符串或对象")
    
    return paths, meta_dict


def _normalize_order_reference_images(reference_images_input: Union[List[str], List[Dict]]) -> tuple:
    """
    兼容新旧格式的参考图输入，返回 (路径列表, 元信息字典)
    
    旧格式: ["a.jpg", "b.jpg"]
    新格式: [{"path": "a.jpg", "ref_type": "product_reference"}, ...]
    
    会进行输入校验：
    - 检查 path 是否存在
    - 检查 ref_type 是否在允许枚举中
    - 检查路径是否重复
    
    返回：
    - paths: 路径列表
    - meta_dict: 元信息字典（按路径索引，便于后续查询）
    """
    if not reference_images_input:
        return [], {}
    
    paths = []
    meta_dict = {}
    seen_paths = set()
    
    for idx, item in enumerate(reference_images_input):
        if isinstance(item, str):
            # 旧格式：纯字符串路径，默认为 product_reference
            path = str(item).strip()
            if not path:
                raise ValueError(f"参考图[{idx}] 路径为空")
            if path in seen_paths:
                raise ValueError(f"参考图[{idx}] 路径重复：{path}")
            seen_paths.add(path)
            paths.append(path)
            meta_dict[path] = {"ref_type": "product_reference"}
            
        elif isinstance(item, dict):
            # 新格式：带 ref_type 的对象
            path = str(item.get("path", "")).strip()
            if not path:
                raise ValueError(f"参考图[{idx}] 缺少 path 字段或 path 为空")
            if path in seen_paths:
                raise ValueError(f"参考图[{idx}] 路径重复：{path}")
            seen_paths.add(path)
            
            ref_type = item.get("ref_type")
            if ref_type is None:
                raise ValueError(f"参考图[{idx}] 缺少 ref_type 字段")
            
            ref_type = str(ref_type).strip()
            if ref_type not in VALID_REF_TYPES:
                raise ValueError(
                    f"参考图[{idx}] ref_type='{ref_type}' 不合法，"
                    f"允许值: {', '.join(sorted(VALID_REF_TYPES))}"
                )
            
            paths.append(path)
            meta_dict[path] = {
                "ref_type": ref_type,
            }
        else:
            raise ValueError(f"参考图[{idx}] 格式错误，应为字符串或对象")
    
    return paths, meta_dict


def ingest(payload: Dict) -> Dict:
    required = ["order_id", "sku_id", "product_name", "product_desc", "refund_images"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # 兼容新旧格式的参考图输入
    order_reference_images_input = payload.get("order_reference_images") or payload.get("standard_images") or []
    order_reference_images, order_reference_meta = _normalize_order_reference_images(order_reference_images_input)
    
    # 兼容新旧格式的退款图输入
    refund_images_input = payload.get("refund_images") or []
    refund_images, refund_image_meta = _normalize_refund_images(refund_images_input)
    
    if not order_reference_images:
        raise ValueError("At least one order_reference_images image is required")
    if not refund_images:
        raise ValueError("At least one refund image is required")

    trace_id = payload.get("trace_id") or f"TRACE_{uuid.uuid4().hex[:12]}"
    
    # 标准化并校验 required_shot_types
    required_shot_types = _normalize_required_shot_types(payload.get("required_shot_types"))
    
    return {
        "order_id": payload["order_id"],
        "sku_id": payload["sku_id"],
        "product_name": str(payload.get("product_name", "")).strip(),
        "product_desc": str(payload.get("product_desc", "")).strip(),
        "category": str(payload.get("category", "")).strip().lower(),
        "expected_labels": payload.get("expected_labels") or [],
        "negative_labels": payload.get("negative_labels") or [],
        # 参考图：路径列表（供图像处理使用）
        "order_reference_images": order_reference_images,
        # 新增：参考图元信息列表（供后续关联分析使用）
        "order_reference_meta": order_reference_meta,
        # 退款图：路径列表（供图像处理使用）
        "refund_images": refund_images,
        # 退款图元信息列表（供合规层使用）
        "refund_image_meta": refund_image_meta,
        "brand": str(payload.get("brand", "")).strip(),
        "color": str(payload.get("color", "")).strip(),
        "package_form": str(payload.get("package_form", "")).strip(),
        "spec": str(payload.get("spec", "")).strip(),
        "refund_reason": str(payload.get("refund_reason", "")).strip(),
        # 标准化后的必传图类型列表
        "required_shot_types": required_shot_types,
        "trace_id": trace_id,
    }
