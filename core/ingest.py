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

# 允许的 ref_type 枚举
VALID_REF_TYPES: Set[str] = {
    "product_reference",
    "order_link_reference",
}


def _normalize_refund_images(refund_images_input: Union[List[str], List[Dict]]) -> tuple:
    """
    兼容新旧格式的退款图输入，返回 (路径列表, 元信息字典)
    
    旧格式: ["a.jpg", "b.jpg"]
    新格式: [{"path": "a.jpg", "shot_type": "overview"}, ...]
    
    会进行输入校验：
    - 检查 path 是否存在
    - 检查 shot_type 是否在允许枚举中
    
    返回：
    - paths: 路径列表
    - meta_dict: 元信息字典（按路径索引，便于后续查询）
    """
    if not refund_images_input:
        return [], {}
    
    paths = []
    meta_dict = {}
    
    for idx, item in enumerate(refund_images_input):
        if isinstance(item, str):
            # 旧格式：纯字符串路径
            path = str(item).strip()
            if not path:
                raise ValueError(f"退款图[{idx}] 路径为空")
            paths.append(path)
            meta_dict[path] = {"shot_type": None, "user_claim": ""}
            
        elif isinstance(item, dict):
            # 新格式：带 shot_type 的对象
            path = str(item.get("path", "")).strip()
            if not path:
                raise ValueError(f"退款图[{idx}] 缺少 path 字段或 path 为空")
            
            shot_type = item.get("shot_type")
            if shot_type is None:
                raise ValueError(f"退款图[{idx}] 缺少 shot_type 字段")
            
            shot_type = str(shot_type).strip()
            if shot_type not in VALID_SHOT_TYPES:
                raise ValueError(
                    f"退款图[{idx}] shot_type='{shot_type}' 不合法，"
                    f"允许值: {', '.join(sorted(VALID_SHOT_TYPES))}"
                )
            
            # auxiliary 类型的用户声明（必填）
            user_claim = str(item.get("user_claim", "")).strip()
            
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
    
    返回：
    - paths: 路径列表
    - meta_dict: 元信息字典（按路径索引，便于后续查询）
    """
    if not reference_images_input:
        return [], {}
    
    paths = []
    meta_dict = {}
    
    for idx, item in enumerate(reference_images_input):
        if isinstance(item, str):
            # 旧格式：纯字符串路径，默认为 product_reference
            path = str(item).strip()
            if not path:
                raise ValueError(f"参考图[{idx}] 路径为空")
            paths.append(path)
            meta_dict[path] = {"ref_type": "product_reference"}
            
        elif isinstance(item, dict):
            # 新格式：带 ref_type 的对象
            path = str(item.get("path", "")).strip()
            if not path:
                raise ValueError(f"参考图[{idx}] 缺少 path 字段或 path 为空")
            
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
        "required_shot_types": payload.get("required_shot_types") or ["overview"],
        "merchant_marker_required": bool(payload.get("merchant_marker_required", False)),
        "trace_id": trace_id,
    }