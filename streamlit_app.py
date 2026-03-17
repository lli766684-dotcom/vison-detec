import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageOps

from core.compliance import evidence_compliance
from core.evidence import build_object_evidence, evidence_writer
from core.fusion import fusion_engine
from core.ingest import ingest
from core.preprocess import preprocess_images
from core.multimodal_analyzer import multimodal_analyze_batch
from core.ai_fake_detector import analyze_refund_images_ai_fake
from main import load_threshold_config


st.set_page_config(page_title="第一模块核验工作台", page_icon="🧾", layout="wide")

ROOT = Path(__file__).resolve().parent
PROMPT_FILE = ROOT / "config" / "prompt_templates.json"
THRESH_FILE = ROOT / "config" / "thresholds.json"
MULTIMODAL_FILE = ROOT / "config" / "multimodal.json"
AI_DETECTOR_FILE = ROOT / "config" / "ai_detector.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# shot_type 配置
# 必传图：damage_closeup, damage_with_order_link（最少1张，最多3张）
# 推荐图：overview, merchant_label_closeup, auxiliary（没有最少要求，最多3张）
SHOT_TYPE_CONFIG = {
    "damage_closeup": {
        "label": "商品损坏细节图",
        "description": "展示腐坏、裂痕、破损、渗漏、变形等问题部位的实拍图",
        "category": "required",
        "min": 1,
        "max": 3,
    },
    "damage_with_order_link": {
        "label": "商品与订单/商家关联图",
        "description": "同时包含「问题商品」和「订单/商家来源信息」的实拍图（如包装、标签、面单等）",
        "category": "required",
        "min": 1,
        "max": 3,
    },
    "overview": {
        "label": "商品整体图",
        "description": "展示商品整体状态的实拍图",
        "category": "recommended",
        "min": 0,
        "max": 3,
    },
    "merchant_label_closeup": {
        "label": "商家标签/包装近景图",
        "description": "展示商家标签、面单、包装标识的近景实拍图",
        "category": "recommended",
        "min": 0,
        "max": 3,
    },
    "auxiliary": {
        "label": "其他辅助图",
        "description": "用于补充说明某项异常现象的实拍图，需填写文字说明",
        "category": "recommended",
        "min": 0,
        "max": 3,
        "requires_user_claim": True,
    },
}

REF_TYPE_CONFIG = {
    "product_reference": {
        "label": "商品主体参考图",
        "description": "用于提供订单商品正常外观的参考",
        "category": "required",
        "min": 1,
        "max": 5,
    },
    "order_link_reference": {
        "label": "包装/标签参考图",
        "description": "用于提供订单/商家侧包装、标签、贴纸、面单等关联物参考",
        "category": "required",
        "min": 1,
        "max": 5,
    },
}


def _init_state():
    if "app_page" not in st.session_state:
        st.session_state["app_page"] = "首页"
    if "business_payload" not in st.session_state:
        st.session_state["business_payload"] = {
            "order_id": "O001",
            "sku_id": "SKU_APPLE_01",
            "product_name": "蓝莓",
            "product_desc": "当季云南花香蓝莓鲜果脆甜新鲜水果纯甜特大果luckycup小杯装礼盒",
            "category": "fruit",
            "refund_reason": "damaged",
            "brand": "",
            "color": "",
            "package_form": "",
            "spec": "",
            "order_reference_images": [
                "E:/E-com/sample_data/normal_case/standard_1.png",
                "E:/E-com/sample_data/normal_case/standard_2.png",
            ],
            "refund_images": ["E:/E-com/sample_data/normal_case/refund_1.png"],
            "required_shot_types": ["overview", "damage_closeup", "merchant_label_closeup"],
            "merchant_marker_required": True,  # 保留兼容，实际使用 required_shot_types
            "expected_labels": ["apple", "damaged apple", "fruit"],
            "negative_labels": ["banana", "pear", "cartoon", "poster", "screenshot"],
        }
    if "business_result" not in st.session_state:
        st.session_state["business_result"] = None
    if "debug_state" not in st.session_state:
        st.session_state["debug_state"] = {
            "step_ok": {
                "ingest": False,
                "preprocess": False,
                "compliance": False,
                "multimodal_analysis": False,
                "ai_fake_detection": False,
                "fusion": False,
                "evidence": False,
            },
            "error": {},
            "ctx": {},
        }
    if "prompt_cfg_text" not in st.session_state:
        st.session_state["prompt_cfg_text"] = PROMPT_FILE.read_text(encoding="utf-8-sig") if PROMPT_FILE.exists() else "{}"
    if "thresh_cfg_text" not in st.session_state:
        st.session_state["thresh_cfg_text"] = THRESH_FILE.read_text(encoding="utf-8-sig") if THRESH_FILE.exists() else "{}"
    if "multimodal_cfg" not in st.session_state:
        if MULTIMODAL_FILE.exists():
            try:
                st.session_state["multimodal_cfg"] = json.loads(MULTIMODAL_FILE.read_text(encoding="utf-8-sig"))
            except Exception:
                st.session_state["multimodal_cfg"] = {
                    "use_multimodal_api": False,
                    "provider": "openai_compatible",
                    "api_base": "",
                    "api_key": "",
                    "model_name": "gpt-4o-mini",
                    "timeout_sec": 60,
                }
        else:
            st.session_state["multimodal_cfg"] = {
                "use_multimodal_api": False,
                "provider": "openai_compatible",
                "api_base": "",
                "api_key": "",
                "model_name": "gpt-4o-mini",
                "timeout_sec": 60,
            }
    # AI 检测器配置
    if "ai_detector_cfg" not in st.session_state:
        if AI_DETECTOR_FILE.exists():
            try:
                st.session_state["ai_detector_cfg"] = json.loads(AI_DETECTOR_FILE.read_text(encoding="utf-8-sig"))
            except Exception:
                st.session_state["ai_detector_cfg"] = {
                    "enabled": False,
                    "detector_type": "mock",
                    "model_path": "",
                    "python_exe": "",
                    "script_path": "scripts/univfd_single_infer.py",
                    "arch": "CLIP:ViT-L/14",
                    "device": "cuda",
                    "api_base": "",
                    "api_key": "",
                    "model_name": "",
                    "timeout_sec": 60,
                }
        else:
            st.session_state["ai_detector_cfg"] = {
                "enabled": False,
                "detector_type": "mock",
                "model_path": "",
                "python_exe": "",
                "script_path": "scripts/univfd_single_infer.py",
                "arch": "CLIP:ViT-L/14",
                "device": "cuda",
                "api_base": "",
                "api_key": "",
                "model_name": "",
                "timeout_sec": 60,
            }
    # 退款图上传状态（按图片粒度）
    if "refund_image_list" not in st.session_state:
        st.session_state["refund_image_list"] = []
    # 参考图上传状态（按图片粒度）
    if "reference_image_list" not in st.session_state:
        st.session_state["reference_image_list"] = []
    # 文件上传器的唯一键（用于重置）
    if "refund_uploader_key" not in st.session_state:
        st.session_state["refund_uploader_key"] = 0
    if "reference_uploader_key" not in st.session_state:
        st.session_state["reference_uploader_key"] = 0


def _split_lines(text: str) -> List[str]:
    return [x.strip() for x in text.splitlines() if x.strip()]


def _save_upload_file(file, tag: str, idx: int) -> str:
    """保存单个上传文件，返回路径"""
    cache_dir = ROOT / ".cache" / "streamlit_uploads" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = Path(file.name).suffix or ".jpg"
    candidate = cache_dir / f"{tag}_{idx}{suffix.lower()}"
    if candidate.exists():
        candidate = cache_dir / f"{tag}_{idx}_{next(tempfile._get_candidate_names())}{suffix.lower()}"
    candidate.write_bytes(file.read())
    return str(candidate)


def _show_thumb_gallery(paths: List[str], title: str, max_items: int = 4, thumb_size: int = 150):
    st.markdown(f"#### {title}")
    if not paths:
        st.info("暂无图片")
        return

    st.markdown(
        f"""
        <style>
        .thumb-box {{
            width: {thumb_size}px;
            height: {thumb_size}px;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            overflow: hidden;
            margin: 0 auto 6px auto;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8fafc;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    show_paths = paths[:max_items]
    cols = st.columns(len(show_paths))
    selected_key = f"preview_selected::{title}"
    for i, p in enumerate(show_paths):
        try:
            img = Image.open(p)
            thumb = ImageOps.fit(img.convert("RGB"), (thumb_size, thumb_size), method=Image.Resampling.LANCZOS)
            cols[i].markdown('<div class="thumb-box">', unsafe_allow_html=True)
            cols[i].image(thumb, width=thumb_size)
            cols[i].markdown("</div>", unsafe_allow_html=True)
            cols[i].caption(Path(p).name)
            if cols[i].button("查看", key=f"view::{title}::{i}"):
                st.session_state[selected_key] = p
        except Exception as e:
            cols[i].warning(f"无法预览\n{Path(p).name}\n{e}")
    if len(paths) > max_items:
        st.caption(f"已显示前 {max_items} 张，共 {len(paths)} 张")

    selected = st.session_state.get(selected_key)
    if selected and Path(selected).exists():
        st.markdown("##### 大图预览")
        st.image(str(selected), caption=selected)


def _section_header(title: str, source: str, badge_color: str = "#e2e8f0"):
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-top:8px;margin-bottom:2px;">
            <h3 style="margin:0;">{title}</h3>
            <span style="font-size:12px;font-weight:600;color:#0f172a;background:{badge_color};border:1px solid #cbd5e1;border-radius:999px;padding:2px 10px;">{source}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_refund_image_item(idx: int, item: Dict, shot_types: List[str]) -> Dict:
    """渲染单张退款图的元信息编辑区域，返回更新后的 item"""
    st.markdown(f"**图片 {idx + 1}**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 缩略图
        try:
            img = Image.open(item["path"])
            thumb = ImageOps.fit(img.convert("RGB"), (120, 120), method=Image.Resampling.LANCZOS)
            st.image(thumb, width=120)
        except Exception:
            st.warning("无法预览")
    
    with col2:
        # shot_type 选择
        current_shot_type = item.get("shot_type", "damage_closeup")
        shot_type_idx = shot_types.index(current_shot_type) if current_shot_type in shot_types else 0
        
        selected_shot_type = st.selectbox(
            "图片类型",
            options=shot_types,
            format_func=lambda x: f"{SHOT_TYPE_CONFIG[x]['label']} - {SHOT_TYPE_CONFIG[x]['description']}",
            index=shot_type_idx,
            key=f"refund_shot_type_{idx}",
        )
        
        item["shot_type"] = selected_shot_type
        
        # auxiliary 类型显示用户声明输入框
        if selected_shot_type == "auxiliary":
            st.markdown("**📝 用户声明（必填，10-80字）**")
            st.caption("请说明这张图要证明什么，例如：外箱破损、漏液污染、冰袋融化、数量不符等")
            
            current_claim = item.get("user_claim", "")
            user_claim = st.text_area(
                "用户声明",
                value=current_claim,
                placeholder="请输入这张图要证明的内容...",
                key=f"refund_user_claim_{idx}",
                max_chars=80,
                height=80,
                label_visibility="collapsed",
            )
            
            claim_len = len(user_claim.strip())
            if claim_len < 10:
                st.warning(f"⚠️ 至少需要 10 个字（当前 {claim_len} 字）")
                item["user_claim_valid"] = False
            else:
                st.success(f"✅ 已填写 {claim_len} 字")
                item["user_claim_valid"] = True
            
            item["user_claim"] = user_claim.strip()
        else:
            item["user_claim"] = ""
            item["user_claim_valid"] = True
    
    # 删除按钮
    if st.button("🗑️ 删除此图", key=f"refund_delete_{idx}"):
        return None  # 返回 None 表示删除
    
    st.markdown("---")
    return item




def _render_refund_upload_section_v2() -> Tuple[List[Dict], Dict]:
    """渲染退款图上传区域（新版：按图片粒度填写元信息）"""
    st.markdown("### 退款图片上传")
    
    # ==================== 必传区 ====================
    _section_header("必传图", "必须上传", badge_color="#fecaca")
    st.caption("以下两类图片必须上传，用于证明问题真实存在并与订单关联")
    
    # ==================== 上传区域 ====================
    st.markdown("---")
    
    # 文件上传器（使用动态 key，确认添加后重置）
    refund_uploader_key = f"refund_uploader_widget_{st.session_state.get('refund_uploader_key', 0)}"
    uploaded_files = st.file_uploader(
        "选择要上传的图片",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key=refund_uploader_key,
        help="选择图片后，点击「确认添加」按钮将图片添加到列表",
    )
    
    # 显示待添加的文件预览
    if uploaded_files:
        st.info(f"已选择 {len(uploaded_files)} 个文件，点击下方按钮确认添加")
        
        # 预览待添加的文件
        preview_cols = st.columns(min(len(uploaded_files), 4))
        for i, f in enumerate(uploaded_files[:4]):
            with preview_cols[i]:
                st.caption(f"📄 {f.name}")
    
    # 确认添加按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 确认添加选中的图片", key="confirm_add_refund", disabled=not uploaded_files):
            existing_signatures = set()
            for item in st.session_state["refund_image_list"]:
                existing_signatures.add(Path(item["path"]).stem)
            
            added_count = 0
            for f in uploaded_files:
                file_signature = Path(f.name).stem
                if file_signature not in existing_signatures:
                    saved_path = _save_upload_file(f, "refund", len(st.session_state["refund_image_list"]) + 1)
                    st.session_state["refund_image_list"].append({
                        "path": saved_path,
                        "shot_type": "damage_closeup",
                        "user_claim": "",
                        "user_claim_valid": True,
                    })
                    existing_signatures.add(file_signature)
                    added_count += 1
            
            # 重置 file_uploader（通过改变 key 来清空缓存）
            st.session_state["refund_uploader_key"] += 1
            
            if added_count > 0:
                st.success(f"已添加 {added_count} 张图片")
            else:
                st.warning("所有选中的图片已存在于列表中")
            st.rerun()
    
    with col2:
        if st.button("🗑️ 清空所有退款图", key="clear_all_refund"):
            st.session_state["refund_image_list"] = []
            st.rerun()
    
    # ==================== 已上传图片列表 ====================
    if st.session_state["refund_image_list"]:
        st.markdown("### 已上传图片（请为每张图选择类型）")
        st.markdown("---")
        
        # 获取可用的 shot_types（按类别分组）
        required_types = ["damage_closeup", "damage_with_order_link"]
        recommended_types = ["overview", "merchant_label_closeup", "auxiliary"]
        all_shot_types = required_types + recommended_types
        
        # 渲染每张图片，收集要删除的索引
        to_delete = []
        for idx, item in enumerate(st.session_state["refund_image_list"]):
            updated_item = _render_refund_image_item(idx, item, all_shot_types)
            if updated_item is None:
                to_delete.append(idx)
            else:
                st.session_state["refund_image_list"][idx] = updated_item
        
        # 处理删除（从后往前删除，避免索引错乱）
        if to_delete:
            for idx in sorted(to_delete, reverse=True):
                del st.session_state["refund_image_list"][idx]
            st.rerun()
    
    # ==================== 统计状态 ====================
    # 必传图：damage_closeup, damage_with_order_link（最少1张，最多3张）
    # 推荐图：overview, merchant_label_closeup, auxiliary（没有最少要求，最多3张）
    status_info = {
        "required": {},
        "recommended": {},
        "shot_type_counts": {},
    }
    
    for item in st.session_state["refund_image_list"]:
        shot_type = item.get("shot_type", "unknown")
        status_info["shot_type_counts"][shot_type] = status_info["shot_type_counts"].get(shot_type, 0) + 1
    
    # 必传图状态
    for shot_type in ["damage_closeup", "damage_with_order_link"]:
        config = SHOT_TYPE_CONFIG[shot_type]
        count = status_info["shot_type_counts"].get(shot_type, 0)
        status_info["required"][shot_type] = {
            "label": config["label"],
            "count": count,
            "min": config["min"],
            "max": config["max"],
            "satisfied": count >= config["min"],
        }
    
    # 推荐图状态
    for shot_type in ["overview", "merchant_label_closeup", "auxiliary"]:
        config = SHOT_TYPE_CONFIG[shot_type]
        count = status_info["shot_type_counts"].get(shot_type, 0)
        status_info["recommended"][shot_type] = {
            "label": config["label"],
            "count": count,
            "max": config["max"],
        }
    
    # ==================== 上传状态卡片 ====================
    _render_upload_status_v2(status_info)
    
    # 构建返回结果
    result = []
    for item in st.session_state["refund_image_list"]:
        result.append({
            "path": item["path"],
            "shot_type": item["shot_type"],
            "user_claim": item.get("user_claim", ""),
        })
    
    return result, status_info


def _render_upload_status_v2(status_info: Dict):
    """渲染上传状态卡片"""
    st.markdown("### 上传状态")
    
    # 必传图状态检查
    required_satisfied = all(r["satisfied"] for r in status_info["required"].values())
    
    # 超限检查
    over_limit_list = []
    for shot_type, count in status_info.get("shot_type_counts", {}).items():
        if count > 3:
            config = SHOT_TYPE_CONFIG.get(shot_type, {})
            label = config.get("label", shot_type)
            over_limit_list.append({"shot_type": shot_type, "label": label, "count": count})
    
    # 显示必传图状态
    st.markdown("**必传图（每种类型至少1张，最多3张）**")
    
    cols = st.columns(2)
    required_types = ["damage_closeup", "damage_with_order_link"]
    
    for i, shot_type in enumerate(required_types):
        info = status_info["required"].get(shot_type, {})
        count = info.get("count", 0)
        label = info.get("label", shot_type)
        
        # 检查是否超限
        is_over_limit = count > 3
        status_icon = "❌" if is_over_limit else ("✅" if info.get("satisfied", False) else "❌")
        
        with cols[i]:
            st.markdown(f"**{label}**")
            if is_over_limit:
                st.write(f"⚠️ 已上传 {count} 张（超出限制，最多3张）")
            else:
                st.write(f"{status_icon} 已上传 {count}/1 张")
    
    if required_satisfied and not over_limit_list:
        st.success("✅ 必传状态：已满足")
    elif over_limit_list:
        over_labels = [item["label"] for item in over_limit_list]
        st.error(f"❌ 上传超限：{', '.join(over_labels)} 超出最多3张的限制")
    else:
        missing = [info["label"] for info in status_info["required"].values() if not info.get("satisfied", False)]
        st.error(f"❌ 必传状态：缺少 {', '.join(missing)}")
    
    st.markdown("---")
    
    # 显示推荐图状态
    st.markdown("**推荐图（可选，最多3张）**")
    
    cols = st.columns(3)
    recommended_types = ["overview", "merchant_label_closeup", "auxiliary"]
    
    for i, shot_type in enumerate(recommended_types):
        info = status_info["recommended"].get(shot_type, {})
        count = info.get("count", 0)
        label = info.get("label", shot_type)
        
        # 检查是否超限
        is_over_limit = count > 3
        
        with cols[i]:
            if is_over_limit:
                st.markdown(f"**{label}**")
                st.write(f"⚠️ 已上传 {count} 张（超出限制）")
            else:
                status_icon = "✅" if count > 0 else "⚪"
                st.markdown(f"**{label}**")
                st.write(f"{status_icon} 已上传 {count}/3 张")


def _render_reference_image_item(idx: int, item: Dict, ref_types: List[str]) -> Dict:
    """渲染单张参考图的元信息编辑区域，返回更新后的 item"""
    st.markdown(f"**图片 {idx + 1}**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 缩略图
        try:
            img = Image.open(item["path"])
            thumb = ImageOps.fit(img.convert("RGB"), (120, 120), method=Image.Resampling.LANCZOS)
            st.image(thumb, width=120)
        except Exception:
            st.warning("无法预览")
    
    with col2:
        # ref_type 选择
        current_ref_type = item.get("ref_type", "product_reference")
        ref_type_idx = ref_types.index(current_ref_type) if current_ref_type in ref_types else 0
        
        selected_ref_type = st.selectbox(
            "参考图类型",
            options=ref_types,
            format_func=lambda x: f"{REF_TYPE_CONFIG[x]['label']} - {REF_TYPE_CONFIG[x]['description']}",
            index=ref_type_idx,
            key=f"ref_ref_type_{idx}",
        )
        
        item["ref_type"] = selected_ref_type
    
    # 删除按钮
    if st.button("🗑️ 删除此图", key=f"ref_delete_{idx}"):
        return None  # 返回 None 表示删除
    
    st.markdown("---")
    return item


def _render_reference_upload_section_v2() -> Tuple[List[Dict], Dict]:
    """渲染参考图上传区域（新版：按图片粒度填写元信息）"""
    st.markdown("### 参考图片上传")
    
    _section_header("商品主体参考图", "必须上传", badge_color="#fecaca")
    st.caption("订单参考图不是绝对标准图，而是订单侧或商家侧视觉参照，用于核验商品外观、包装和标识特征。")
    
    st.markdown("---")
    
    # 文件上传器（使用动态 key，确认添加后重置）
    reference_uploader_key = f"reference_uploader_widget_{st.session_state.get('reference_uploader_key', 0)}"
    uploaded_files = st.file_uploader(
        "选择要上传的参考图片",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key=reference_uploader_key,
        help="选择图片后，点击「确认添加」按钮将图片添加到列表",
    )
    
    # 显示待添加的文件预览
    if uploaded_files:
        st.info(f"已选择 {len(uploaded_files)} 个文件，点击下方按钮确认添加")
        
        # 预览待添加的文件
        preview_cols = st.columns(min(len(uploaded_files), 4))
        for i, f in enumerate(uploaded_files[:4]):
            with preview_cols[i]:
                st.caption(f"📄 {f.name}")
    
    # 确认添加按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 确认添加选中的参考图", key="confirm_add_reference", disabled=not uploaded_files):
            existing_signatures = set()
            for item in st.session_state["reference_image_list"]:
                existing_signatures.add(Path(item["path"]).stem)
            
            added_count = 0
            for f in uploaded_files:
                file_signature = Path(f.name).stem
                if file_signature not in existing_signatures:
                    saved_path = _save_upload_file(f, "reference", len(st.session_state["reference_image_list"]) + 1)
                    st.session_state["reference_image_list"].append({
                        "path": saved_path,
                        "ref_type": "product_reference",
                    })
                    existing_signatures.add(file_signature)
                    added_count += 1
            
            # 重置 file_uploader（通过改变 key 来清空缓存）
            st.session_state["reference_uploader_key"] += 1
            
            if added_count > 0:
                st.success(f"已添加 {added_count} 张参考图")
            else:
                st.warning("所有选中的图片已存在于列表中")
            st.rerun()
    
    with col2:
        if st.button("🗑️ 清空所有参考图", key="clear_all_reference"):
            st.session_state["reference_image_list"] = []
            st.rerun()
    
    # ==================== 已上传图片列表 ====================
    if st.session_state["reference_image_list"]:
        st.markdown("### 已上传参考图（请为每张图选择类型）")
        st.markdown("---")
        
        all_ref_types = ["product_reference", "order_link_reference"]
        
        # 渲染每张图片，收集要删除的索引
        to_delete = []
        for idx, item in enumerate(st.session_state["reference_image_list"]):
            updated_item = _render_reference_image_item(idx, item, all_ref_types)
            if updated_item is None:
                to_delete.append(idx)
            else:
                # 更新 item 的字段
                st.session_state["reference_image_list"][idx] = updated_item
        
        # 处理删除（从后往前删除，避免索引错乱）
        if to_delete:
            for idx in sorted(to_delete, reverse=True):
                del st.session_state["reference_image_list"][idx]
            st.rerun()
    
    # 统计状态（使用 REF_TYPE_CONFIG 中的配置）
    status_info = {
        "product_reference": {
            "count": 0,
            "min": REF_TYPE_CONFIG["product_reference"]["min"],
            "max": REF_TYPE_CONFIG["product_reference"]["max"],
            "label": REF_TYPE_CONFIG["product_reference"]["label"],
            "satisfied": False,
        },
        "order_link_reference": {
            "count": 0,
            "min": REF_TYPE_CONFIG["order_link_reference"]["min"],
            "max": REF_TYPE_CONFIG["order_link_reference"]["max"],
            "label": REF_TYPE_CONFIG["order_link_reference"]["label"],
            "satisfied": True,
        },
    }
    
    for item in st.session_state["reference_image_list"]:
        ref_type = item.get("ref_type", "product_reference")
        if ref_type in status_info:
            status_info[ref_type]["count"] += 1
    
    status_info["product_reference"]["satisfied"] = status_info["product_reference"]["count"] >= 1
    status_info["order_link_reference"]["satisfied"] = status_info["order_link_reference"]["count"] >= 1
    
    # 超限检查
    reference_over_limit = []
    for ref_type, info in status_info.items():
        if info["count"] > info["max"]:
            reference_over_limit.append({
                "ref_type": ref_type,
                "label": info["label"],
                "count": info["count"],
                "max": info["max"],
            })
    
    # 状态卡片
    st.markdown("### 参考图上传状态")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**商品主体参考图**")
        info = status_info["product_reference"]
        is_over_limit = info["count"] > info["max"]
        
        if is_over_limit:
            st.write(f"⚠️ 已上传 {info['count']} 张（超出限制，最多{info['max']}张）")
            st.error(f"❌ 超出上传限制")
        else:
            status_icon = "✅" if info["satisfied"] else "❌"
            st.write(f"{status_icon} 已上传 {info['count']}/{info['max']} 张（最少 {info['min']} 张）")
            if info["satisfied"]:
                st.success("已满足要求")
            else:
                st.error("至少需要 1 张商品主体参考图")
    
    with col2:
        st.markdown("**包装/标签参考图**")
        info = status_info["order_link_reference"]
        is_over_limit = info["count"] > info["max"]
        
        if is_over_limit:
            st.write(f"⚠️ 已上传 {info['count']} 张（超出限制，最多{info['max']}张）")
            st.error(f"❌ 超出上传限制")
        else:
            status_icon = "✅" if info["satisfied"] else "❌"
            st.write(f"{status_icon} 已上传 {info['count']}/{info['max']} 张（最少 {info['min']} 张）")
            if info["satisfied"]:
                st.success("已满足要求")
            else:
                st.error("至少需要 1 张包装/标签参考图")
    
    # 构建返回结果
    result = []
    for item in st.session_state["reference_image_list"]:
        result.append({
            "path": item["path"],
            "ref_type": item["ref_type"],
        })
    
    return result, status_info


def _render_sidebar():
    """渲染侧栏：多模态分析配置"""
    with st.sidebar:
        st.markdown("### 多模态分析配置")
        
        mm_cfg = st.session_state.get("multimodal_cfg", {})
        
        use_multimodal = st.toggle(
            "启用多模态分析",
            value=mm_cfg.get("use_multimodal_api", False),
            key="sidebar_use_multimodal",
        )
        
        if use_multimodal:
            st.markdown("---")
            
            provider = st.selectbox(
                "模型提供方式",
                ["openai_compatible"],
                index=0,
                key="sidebar_provider",
            )
            
            api_base = st.text_input(
                "API Base URL",
                value=mm_cfg.get("api_base", ""),
                key="sidebar_api_base",
                placeholder="https://api.openai.com/v1",
            )
            
            api_key = st.text_input(
                "API Key",
                value=mm_cfg.get("api_key", ""),
                key="sidebar_api_key",
                type="password",
            )
            
            model_name = st.text_input(
                "模型名称",
                value=mm_cfg.get("model_name", "gpt-4o-mini"),
                key="sidebar_model_name",
            )
            
            timeout_sec = st.number_input(
                "请求超时（秒）",
                min_value=10,
                max_value=300,
                value=mm_cfg.get("timeout_sec", 60),
                key="sidebar_timeout",
            )
            
            st.session_state["multimodal_cfg"] = {
                "use_multimodal_api": use_multimodal,
                "provider": provider,
                "api_base": api_base,
                "api_key": api_key,
                "model_name": model_name,
                "timeout_sec": timeout_sec,
            }
            
            if api_base and api_key:
                st.success("✅ 配置完整")
            else:
                st.warning("⚠️ 请填写 API Base URL 和 API Key")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 保存配置", use_container_width=True):
                    try:
                        config_to_save = {
                            "use_multimodal_api": use_multimodal,
                            "provider": provider,
                            "api_base": api_base,
                            "api_key": api_key,
                            "model_name": model_name,
                            "timeout_sec": timeout_sec,
                        }
                        MULTIMODAL_FILE.write_text(
                            json.dumps(config_to_save, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        st.success("✅ 配置已保存到文件")
                    except Exception as e:
                        st.error(f"保存失败：{e}")
            
            with col2:
                if st.button("🔄 重新加载", key="reload_multimodal_config", use_container_width=True):
                    if MULTIMODAL_FILE.exists():
                        try:
                            st.session_state["multimodal_cfg"] = json.loads(
                                MULTIMODAL_FILE.read_text(encoding="utf-8-sig")
                            )
                            st.success("✅ 已重新加载配置")
                            st.rerun()
                        except Exception as e:
                            st.error(f"加载失败：{e}")
                    else:
                        st.warning("配置文件不存在")
            
            with st.expander("📄 查看配置文件", expanded=False):
                if MULTIMODAL_FILE.exists():
                    try:
                        file_content = MULTIMODAL_FILE.read_text(encoding="utf-8-sig")
                        display_content = json.loads(file_content)
                        if display_content.get("api_key"):
                            key = display_content["api_key"]
                            if len(key) > 8:
                                display_content["api_key"] = key[:4] + "****" + key[-4:]
                        st.json(display_content)
                    except Exception as e:
                        st.error(f"读取失败：{e}")
                else:
                    st.info("配置文件不存在，保存后将创建")
        else:
            st.session_state["multimodal_cfg"]["use_multimodal_api"] = False
            st.info("多模态分析已禁用")
            
            if st.button("💾 保存当前配置", use_container_width=True):
                try:
                    config_to_save = {
                        "use_multimodal_api": False,
                        "provider": mm_cfg.get("provider", "openai_compatible"),
                        "api_base": mm_cfg.get("api_base", ""),
                        "api_key": mm_cfg.get("api_key", ""),
                        "model_name": mm_cfg.get("model_name", "gpt-4o-mini"),
                        "timeout_sec": mm_cfg.get("timeout_sec", 60),
                    }
                    MULTIMODAL_FILE.write_text(
                        json.dumps(config_to_save, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    st.success("✅ 配置已保存")
                except Exception as e:
                    st.error(f"保存失败：{e}")
        
        # ==================== AI 检测器配置 ====================
        st.markdown("---")
        st.markdown("### AI 生成检测配置")
        
        ai_cfg = st.session_state.get("ai_detector_cfg", {})
        
        # 第一层：通用字段
        use_ai_detector = st.toggle(
            "启用 AI 生成检测",
            value=ai_cfg.get("enabled", False),
            key="sidebar_use_ai_detector",
        )
        
        detector_type = st.selectbox(
            "检测器类型",
            ["mock", "local_univfd", "api"],
            index=["mock", "local_univfd", "api"].index(ai_cfg.get("detector_type", "mock")),
            key="sidebar_detector_type",
            help="mock: 模拟检测器（开发测试用）| local_univfd: 本地 UnivFD 模型 | api: 第三方 API 服务",
        )
        
        # 第二层：按 detector_type 动态显示字段
        # 初始化所有字段
        model_path = ""
        python_exe = ""
        script_path = ""
        arch = "CLIP:ViT-L/14"
        device = "cuda"
        api_base = ""
        api_key = ""
        model_name = ""
        
        if detector_type == "mock":
            st.info("ℹ️ 当前为模拟检测器，仅用于联调测试")
        
        elif detector_type == "local_univfd":
            st.markdown("**本地 UnivFD 配置**")
            
            model_path = st.text_input(
                "模型权重路径",
                value=ai_cfg.get("model_path", ""),
                key="sidebar_model_path",
                placeholder="pretrained_weights/fc_weights.pth",
            )
            
            python_exe = st.text_input(
                "UnivFD Python 路径",
                value=ai_cfg.get("python_exe", ""),
                key="sidebar_python_exe",
                placeholder="D:/python/anaconda/envs/univfd/python.exe",
            )
            
            script_path = st.text_input(
                "推理脚本路径",
                value=ai_cfg.get("script_path", "scripts/univfd_single_infer.py"),
                key="sidebar_script_path",
                placeholder="scripts/univfd_single_infer.py",
            )
            
            arch = st.text_input(
                "模型结构",
                value=ai_cfg.get("arch", "CLIP:ViT-L/14"),
                key="sidebar_arch",
            )
            
            device = st.selectbox(
                "运行设备",
                ["cuda", "cpu"],
                index=0 if ai_cfg.get("device", "cuda") == "cuda" else 1,
                key="sidebar_device",
            )
            
            # 校验
            if model_path and python_exe and script_path:
                st.success("✅ 本地 detector 配置完整")
            else:
                missing = []
                if not model_path:
                    missing.append("模型权重路径")
                if not python_exe:
                    missing.append("Python 路径")
                if not script_path:
                    missing.append("推理脚本路径")
                st.warning(f"⚠️ 本地 detector 配置不完整，缺少：{', '.join(missing)}")
        
        elif detector_type == "api":
            st.markdown("**API 配置**")
            
            api_base = st.text_input(
                "API Base URL",
                value=ai_cfg.get("api_base", ""),
                key="sidebar_ai_api_base",
                placeholder="https://api.example.com/v1/detect",
            )
            
            api_key = st.text_input(
                "API Key",
                value=ai_cfg.get("api_key", ""),
                key="sidebar_ai_api_key",
                type="password",
            )
            
            model_name = st.text_input(
                "Model Name（可选）",
                value=ai_cfg.get("model_name", ""),
                key="sidebar_ai_model_name",
                placeholder="gpt-4o-mini",
            )
            
            # 校验
            if api_base and api_key:
                st.success("✅ API 配置完整")
            else:
                missing = []
                if not api_base:
                    missing.append("API Base URL")
                if not api_key:
                    missing.append("API Key")
                st.warning(f"⚠️ API 配置不完整，缺少：{', '.join(missing)}")
        
        # 第三层：通用控制区
        st.markdown("---")
        
        timeout_sec = st.number_input(
            "请求超时（秒）",
            min_value=10,
            max_value=120,
            value=ai_cfg.get("timeout_sec", 60),
            key="sidebar_ai_timeout",
        )
        
        # 统一保存配置
        st.session_state["ai_detector_cfg"] = {
            "enabled": use_ai_detector,
            "detector_type": detector_type,
            "model_path": model_path,
            "python_exe": python_exe,
            "script_path": script_path,
            "arch": arch,
            "device": device,
            "api_base": api_base,
            "api_key": api_key,
            "model_name": model_name,
            "timeout_sec": timeout_sec,
        }
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 保存 AI 配置", use_container_width=True):
                try:
                    config_to_save = st.session_state["ai_detector_cfg"]
                    AI_DETECTOR_FILE.write_text(
                        json.dumps(config_to_save, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    st.success("✅ AI 检测配置已保存")
                except Exception as e:
                    st.error(f"保存失败：{e}")
        
        with col2:
            if st.button("🔄 重新加载", key="reload_ai_detector_config", use_container_width=True):
                if AI_DETECTOR_FILE.exists():
                    try:
                        st.session_state["ai_detector_cfg"] = json.loads(
                            AI_DETECTOR_FILE.read_text(encoding="utf-8-sig")
                        )
                        st.success("✅ 已重新加载配置")
                        st.rerun()
                    except Exception as e:
                        st.error(f"加载失败：{e}")
                else:
                    st.warning("配置文件不存在，请复制 ai_detector.example.json 为 ai_detector.json")
        
        # 查看当前配置
        with st.expander("📄 查看当前配置", expanded=False):
            display_cfg = st.session_state["ai_detector_cfg"].copy()
            if display_cfg.get("api_key"):
                key = display_cfg["api_key"]
                if len(key) > 8:
                    display_cfg["api_key"] = key[:4] + "****" + key[-4:]
            st.json(display_cfg)


def _render_top_nav():
    st.markdown(
        """
        <style>
        .hero-wrap {padding: 28px 24px; border-radius: 20px;
                    background: linear-gradient(135deg, #f7f8fa, #eef2f7);
                    border: 1px solid #e4e8ee; margin-bottom: 18px;}
        .hero-title {font-size: 34px; font-weight: 700; color:#111827; margin-bottom: 8px;}
        .hero-sub {font-size: 15px; color:#4b5563;}
        .card {padding:18px; border-radius:16px; border:1px solid #e5e7eb; background:#fff;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown(
            """
            <div class="hero-wrap">
              <div class="hero-title">第一模块 · 图像证据核验工作台</div>
              <div class="hero-sub">用于核验退款图在证据完整性、商品一致性、属性一致性、损坏证据上的可用性。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("进入业务视图", use_container_width=True):
            st.session_state["app_page"] = "业务视图"
            st.rerun()
    with c3:
        if st.button("进入调试测试视图", use_container_width=True):
            st.session_state["app_page"] = "调试测试视图"
            st.rerun()


def _home_page():
    _render_top_nav()
    st.markdown("### 子页面说明")
    a, b = st.columns(2)
    a.info("业务视图：只做角色信息展示与填写，输出案件预览。")
    b.info("调试测试视图：按流水线逐步测试，每一步都有成功/失败原因，最终结果可同步回业务视图。")


def _business_page():
    st.subheader("业务视图")
    if st.button("返回首页"):
        st.session_state["app_page"] = "首页"
        st.rerun()

    p = st.session_state["business_payload"]

    st.markdown("## 案件概览")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("订单号", p.get("order_id", "-"))
    c2.metric("商品名称", p.get("product_name", "-"))
    c3.metric("退款原因", p.get("refund_reason", "-"))
    c4.metric("案件状态", "已核验" if st.session_state["business_result"] else "待核验")

    _section_header("A. 用户申诉信息", "用户提交")
    refund_reason = st.text_input("退款原因（refund_reason）*", value=p.get("refund_reason", ""))
    
    # 使用新的退款图上传区域
    refund_images_with_meta, upload_status = _render_refund_upload_section_v2()
    
    # 必传状态检查
    required_satisfied = all(r["satisfied"] for r in upload_status["required"].values())
    
    # 超限检查：任何类型超过3张都不允许保存
    over_limit_types = []
    for shot_type, count in upload_status.get("shot_type_counts", {}).items():
        if count > 3:
            config = SHOT_TYPE_CONFIG.get(shot_type, {})
            label = config.get("label", shot_type)
            over_limit_types.append(f"{label}（已上传{count}张，最多3张）")
    
    # auxiliary 类型 user_claim 校验
    auxiliary_valid = True
    for item in st.session_state.get("refund_image_list", []):
        if item.get("shot_type") == "auxiliary":
            if not item.get("user_claim_valid", True):
                auxiliary_valid = False
                break

    _section_header("B. 订单与商品基础信息", "平台自动获取（只读）")
    b1, b2 = st.columns(2)
    with b1:
        order_id = st.text_input("order_id*", value=p.get("order_id", ""), disabled=True)
        sku_id = st.text_input("sku_id*", value=p.get("sku_id", ""), disabled=True)
        product_name = st.text_input("product_name*", value=p.get("product_name", ""), disabled=True)
    with b2:
        product_desc = st.text_area("product_desc", value=p.get("product_desc", ""), disabled=True)
        category = st.text_input("category", value=p.get("category", ""), disabled=True)

    _section_header("C. 商品参考信息（平台/商家侧）", "商家/商品库提供")
    
    # 使用新的参考图上传区域
    order_reference_images_with_meta, reference_status = _render_reference_upload_section_v2()
    
    # 参考图必传状态检查（两种类型都是必传）
    reference_satisfied = (
        reference_status["product_reference"]["satisfied"] 
        and reference_status["order_link_reference"]["satisfied"]
    )
    
    # 参考图超限检查
    reference_over_limit = []
    for ref_type, info in reference_status.items():
        if info["count"] > info["max"]:
            reference_over_limit.append(f"{info['label']}（已上传{info['count']}张，最多{info['max']}张）")
    
    # 可选增强信息（折叠区）
    with st.expander("可选增强信息（非必填）", expanded=False):
        st.caption("以下信息可增强核验效果，但非必填。系统主要依赖参考图和多模态理解。")
        r1, r2 = st.columns(2)
        with r1:
            brand = st.text_input("brand", value=p.get("brand", ""))
            color = st.text_input("color", value=p.get("color", ""))
        with r2:
            package_form = st.text_input("package_form", value=p.get("package_form", ""))
            spec = st.text_input("spec", value=p.get("spec", ""))

    save_btn = st.button("保存业务信息")
    if save_btn:
        thresholds_cfg = load_threshold_config(None)
        frontend_block = bool(
            thresholds_cfg.get("evidence_rules", {}).get("frontend_block_on_missing_required", True)
        )
        
        payload = {
            "order_id": order_id.strip(),
            "sku_id": sku_id.strip(),
            "product_name": product_name.strip(),
            "product_desc": product_desc.strip(),
            "category": category.strip(),
            "refund_reason": refund_reason.strip(),
            "brand": brand.strip(),
            "color": color.strip(),
            "package_form": package_form.strip(),
            "spec": spec.strip(),
            "refund_images": refund_images_with_meta,
            "order_reference_images": order_reference_images_with_meta,
        }
        
        errs = []
        for k in ["order_id", "sku_id", "product_name", "refund_reason"]:
            if not str(payload.get(k, "")).strip():
                errs.append(f"必填字段不能为空：{k}")
        
        if not refund_images_with_meta:
            errs.append("至少上传一张退款图")
        
        if not order_reference_images_with_meta:
            errs.append("至少上传一张商品主体参考图")
        
        # 验证退款图文件存在性
        for item in refund_images_with_meta:
            path = item.get("path", "")
            pp = Path(path)
            if not pp.exists():
                errs.append(f"退款图文件不存在：{path}")
            elif pp.suffix.lower() not in IMAGE_EXTS:
                errs.append(f"退款图格式不支持：{path}")
        
        # 验证参考图文件存在性
        for item in order_reference_images_with_meta:
            path = item.get("path", "")
            pp = Path(path)
            if not pp.exists():
                errs.append(f"参考图文件不存在：{path}")
            elif pp.suffix.lower() not in IMAGE_EXTS:
                errs.append(f"参考图格式不支持：{path}")
        
        # 必传图校验 - 退款图
        if frontend_block and not required_satisfied:
            missing_labels = [info["label"] for info in upload_status["required"].values() if not info["satisfied"]]
            errs.append(f"退款必传图未满足：缺少 {', '.join(missing_labels)}")
        
        # 超限校验 - 任何类型超过3张都不允许保存
        if over_limit_types:
            for over_limit_msg in over_limit_types:
                errs.append(f"上传超限：{over_limit_msg}")
        
        # 必传图校验 - 参考图（两种类型都是必传）
        if frontend_block and not reference_satisfied:
            missing_ref = []
            if not reference_status["product_reference"]["satisfied"]:
                missing_ref.append("商品主体参考图")
            if not reference_status["order_link_reference"]["satisfied"]:
                missing_ref.append("包装/标签参考图")
            errs.append(f"参考图未满足：缺少 {', '.join(missing_ref)}")
        
        # 参考图超限校验
        if reference_over_limit:
            for over_limit_msg in reference_over_limit:
                errs.append(f"参考图上传超限：{over_limit_msg}")
        
        # auxiliary 类型 user_claim 校验
        if not auxiliary_valid:
            errs.append("辅助图（auxiliary）必须填写用户声明（10-80字）")
        
        if errs:
            st.error("保存失败，请修正以下问题：")
            for e in errs:
                st.write(f"- {e}")
        else:
            st.session_state["business_payload"] = payload
            st.success("业务信息已保存。可进入「调试测试视图」执行分步测试。")

    st.markdown("## 案件预览（来自最后一次成功调试）")
    if not st.session_state["business_result"]:
        st.info("暂无核验结果。请到「调试测试视图」完成全流程测试。")
    else:
        r = st.session_state["business_result"]
        # 第一层：结论卡片
        st.markdown("### 结论卡片")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("商品对象匹配", str(r["object_match"]).upper())
        k2.metric("证据完整性", r["evidence_compliance"]["evidence_compliance"])
        k3.metric("风险等级", str(r["image_risk_level"]).upper())
        k4.metric("置信度", f"{r['confidence_score']:.4f}")

        # 第二层：证据摘要
        st.markdown("### 证据摘要")
        missing = r["evidence_compliance"].get("missing_required_shots", [])
        
        # 多模态证据摘要
        mm_evidence = r.get("object_evidence", {}).get("multimodal_evidence", {})
        if mm_evidence:
            s1, s2 = st.columns(2)
            with s1:
                st.write("损坏证据图：", len(mm_evidence.get("damage_images", [])), "张")
                st.write("关联证据图：", len(mm_evidence.get("linkage_images", [])), "张")
            with s2:
                st.write("缺失图像类型：", "、".join(missing) if missing else "无")
        else:
            st.write("缺失图像类型：", "、".join(missing) if missing else "无")

        # AI 生成/真实性检测证据
        ai_evidence = r.get("object_evidence", {}).get("ai_forgery_evidence", {})
        if ai_evidence:
            st.markdown("### AI 生成/真实性检测")
            
            ai_s1, ai_s2, ai_s3, ai_s4 = st.columns(4)
            with ai_s1:
                suspected_count = ai_evidence.get("suspected_ai_images_count", 0)
                st.metric("疑似 AI 图像", f"{suspected_count} 张")
            with ai_s2:
                high_risk_count = ai_evidence.get("high_risk_images_count", 0)
                st.metric("高风险图像", f"{high_risk_count} 张")
            with ai_s3:
                detector_name = ai_evidence.get("detector_name", "-")
                st.metric("检测器", detector_name)
            with ai_s4:
                overall_risk = ai_evidence.get("overall_forgery_risk", "unknown")
                risk_icon = {"low": "✅", "medium": "⚠️", "high": "❌"}.get(overall_risk, "❓")
                st.metric("整体风险", f"{risk_icon} {overall_risk.upper()}")
            
            # 各图像 AI 检测结果
            ai_images = ai_evidence.get("ai_forgery_images", [])
            if ai_images:
                with st.expander("各图像 AI 检测详情", expanded=False):
                    for idx, ai_img in enumerate(ai_images):
                        risk = ai_img.get("forgery_risk_level", "unknown")
                        risk_icon = {"low": "✅", "medium": "⚠️", "high": "❌"}.get(risk, "❓")
                        st.write(f"**图 {idx+1}**: {risk_icon} {risk.upper()} - AI分数: {ai_img.get('ai_gen_score', 0):.2f}")
                        if ai_img.get("forgery_reason"):
                            st.caption(f"原因: {ai_img['forgery_reason']}")
            
            # 建议
            recommendation = ai_evidence.get("recommendation", "")
            if recommendation:
                if "人工复核" in recommendation:
                    st.warning(f"⚠️ {recommendation}")
                else:
                    st.info(f"ℹ️ {recommendation}")

        # 第三层：原始 JSON（折叠）
        with st.expander("原始结果 JSON（技术详情）", expanded=False):
            st.json(r)


def _debug_page():
    st.subheader("调试测试视图")
    if st.button("返回首页"):
        st.session_state["app_page"] = "首页"
        st.rerun()

    p = st.session_state["business_payload"]
    st.markdown("### 业务视图输入快照（只读）")
    st.json(p)

    st.markdown("### 案件输入调试编辑（仅调试视图可改）")
    payload_edit_text = st.text_area(
        "编辑业务输入 JSON（可改 order_id/sku_id/product_name/category 及 expected_labels/negative_labels）",
        value=json.dumps(p, ensure_ascii=False, indent=2),
        height=260,
    )
    if st.button("应用调试输入到业务数据"):
        try:
            edited = json.loads(payload_edit_text)
            st.session_state["business_payload"] = edited
            p = edited
            st.success("已应用调试输入。")
        except Exception as e:
            st.error(f"应用失败：{e}")

    st.markdown("### 调试配置")
    thresh_cfg_text = st.text_area("threshold 配置字段（JSON）", value=st.session_state["thresh_cfg_text"], height=300)

    try:
        thresh_cfg = json.loads(thresh_cfg_text)
        st.success("配置 JSON 解析成功")
    except Exception as e:
        st.error(f"配置 JSON 解析失败：{e}")
        return

    st.session_state["thresh_cfg_text"] = thresh_cfg_text
    dbg = st.session_state["debug_state"]
    ctx = dbg["ctx"]

    st.markdown("### 执行链路")
    st.info("ingest → preprocess → compliance → multimodal_analysis → ai_fake_detection → fusion → evidence")
    st.markdown("---")
    st.markdown("### 分步测试（必须按顺序）")
    s1, s2, s3 = st.columns(3)
    s4, s5, s6, s7 = st.columns(4)
    s8, s9, s10, s11 = st.columns(4)

    def _ok(name):
        dbg["step_ok"][name] = True
        dbg["error"].pop(name, None)

    def _fail(name, err):
        dbg["step_ok"][name] = False
        dbg["error"][name] = str(err)

    if s1.button("Step1 ingest"):
        try:
            ctx["req"] = ingest(p)
            _ok("ingest")
            st.success("ingest 成功")
            st.json(ctx["req"])
        except Exception as e:
            _fail("ingest", e)
            st.error(f"ingest 失败：{e}")
    s1.caption("整理并标准化输入案件")

    if s2.button("Step2 preprocess", disabled=not dbg["step_ok"]["ingest"]):
        try:
            thresholds = load_threshold_config(None)
            thresholds.update(thresh_cfg)
            ctx["thresholds"] = thresholds
            ctx["prep"] = preprocess_images(ctx["req"], thresholds)
            _ok("preprocess")
            st.success("preprocess 成功")
            st.json(ctx["prep"]["image_quality_flags"])
        except Exception as e:
            _fail("preprocess", e)
            st.error(f"preprocess 失败：{e}")
    s2.caption("读取图像并做质量检查")

    if s3.button("Step3 compliance", disabled=not dbg["step_ok"]["preprocess"]):
        try:
            ctx["comp"] = evidence_compliance(ctx["req"], ctx["prep"], ctx["thresholds"])
            _ok("compliance")
            st.success("compliance 成功")
            st.json(ctx["comp"])
        except Exception as e:
            _fail("compliance", e)
            st.error(f"compliance 失败：{e}")
    s3.caption("检查证据图是否完整合规")

    mm_cfg = st.session_state.get("multimodal_cfg", {})
    use_multimodal = mm_cfg.get("use_multimodal_api", False)
    
    # Step4 multimodal_analysis
    if s4.button("Step4 multimodal_analysis", disabled=not dbg["step_ok"]["compliance"]):
        if not use_multimodal:
            st.info("多模态分析未启用，跳过此步骤")
            _ok("multimodal_analysis")
            ctx["mm_results"] = []
        else:
            try:
                with st.spinner("正在进行多模态分析..."):
                    ctx["mm_results"] = multimodal_analyze_batch(
                        refund_images=ctx["req"]["refund_images"],
                        refund_image_meta=ctx["req"]["refund_image_meta"],
                        product_name=ctx["req"].get("product_name", ""),
                        refund_reason=ctx["req"].get("refund_reason", ""),
                        order_reference_images=ctx["req"]["order_reference_images"],
                        order_reference_meta=ctx["req"]["order_reference_meta"],
                        api_config=mm_cfg,
                    )
                _ok("multimodal_analysis")
                st.success("multimodal_analysis 成功")
                
                for idx, mm_result in enumerate(ctx["mm_results"]):
                    with st.expander(f"退款图 {idx+1}: {Path(mm_result['path']).name}", expanded=False):
                        shot_type = mm_result.get('predicted_shot_type', 'auxiliary')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**类型判断**")
                            st.write(f"声明类型: {mm_result.get('declared_shot_type', '-')}")
                            st.write(f"预测类型: {shot_type}")
                            type_match = mm_result.get('declared_shot_type') == shot_type
                            st.write(f"类型一致: {'✅' if type_match else '⚠️'}")
                        
                        with col2:
                            st.markdown("**内容检测**")
                            st.write(f"包含商品: {'✅' if mm_result.get('contains_product') else '❌'}")
                            st.write(f"包含损坏: {'✅' if mm_result.get('contains_damage') else '❌'}")
                            st.write(f"损坏置信度: {mm_result.get('damage_confidence', 0):.2f}")
                            
                            if shot_type in ["damage_with_order_link", "merchant_label_closeup"]:
                                st.write(f"包含关联证据: {'✅' if mm_result.get('contains_order_or_merchant_linkage') else '❌'}")
                                st.write(f"关联置信度: {mm_result.get('linkage_confidence', 0):.2f}")
                        
                        has_match_confidence = mm_result.get('product_match_confidence', 0) > 0
                        if has_match_confidence:
                            st.markdown("**匹配度**")
                            st.write(f"商品匹配度: {mm_result.get('product_match_confidence', 0):.2f}")
                            if shot_type in ["damage_with_order_link", "merchant_label_closeup"]:
                                st.write(f"关联物匹配度: {mm_result.get('reference_linkage_confidence', 0):.2f}")
                        
                        st.write(f"实拍可能性: {'✅' if mm_result.get('is_real_scene_likely') else '⚠️'}")
                        
                        # 新字段展示
                        if shot_type in ["damage_closeup", "damage_with_order_link"]:
                            damage_desc = mm_result.get('damage_description', '')
                            if damage_desc:
                                st.markdown("**损坏描述**")
                                st.write(damage_desc)
                        
                        if shot_type == "damage_with_order_link":
                            linkage_desc = mm_result.get('linkage_description', '')
                            if linkage_desc:
                                st.markdown("**关联证据描述**")
                                st.write(linkage_desc)
                        
                        if shot_type == "merchant_label_closeup":
                            readability = mm_result.get('label_readability', '未知')
                            st.write(f"标签可读性: {readability}")
                        
                        if shot_type == "overview":
                            shows_overall = mm_result.get('shows_overall_product_view', False)
                            st.write(f"展示商品全貌: {'✅' if shows_overall else '⚠️'}")
                        
                        if shot_type == "auxiliary":
                            claim_level = mm_result.get('claim_support_level', 'unknown')
                            claim_analysis = mm_result.get('claim_analysis', '')
                            st.write(f"声明支持度: {claim_level}")
                            if claim_analysis:
                                st.write(f"分析: {claim_analysis}")
                        
                        st.markdown("**摘要**")
                        st.write(mm_result.get('summary', '-'))
                        
                        if not mm_result.get('analysis_success'):
                            st.error(f"分析失败: {mm_result.get('error', '未知错误')}")
            except Exception as e:
                _fail("multimodal_analysis", e)
                st.error(f"multimodal_analysis 失败：{e}")
    s4.caption("多模态图像理解")
    
    # Step5 ai_fake_detection
    ai_fake_disabled = not dbg["step_ok"]["multimodal_analysis"]
    ai_cfg = st.session_state.get("ai_detector_cfg", {})
    
    if s5.button("Step5 ai_fake_detection", disabled=ai_fake_disabled):
        if not ai_cfg.get("enabled", False):
            st.info("AI 生成检测未启用，跳过此步骤")
            _ok("ai_fake_detection")
            ctx["ai_results"] = []
        else:
            try:
                with st.spinner("正在进行 AI 生成/真实性检测..."):
                    ctx["ai_results"] = analyze_refund_images_ai_fake(
                        refund_images=ctx["req"]["refund_images"],
                        refund_image_meta=ctx["req"]["refund_image_meta"],
                        product_name=ctx["req"].get("product_name", ""),
                        refund_reason=ctx["req"].get("refund_reason", ""),
                        detector_config=ai_cfg,  # 使用侧边栏配置
                    )
                _ok("ai_fake_detection")
                st.success("ai_fake_detection 成功")
                
                for idx, ai_result in enumerate(ctx["ai_results"]):
                    with st.expander(f"退款图 {idx+1}: {Path(ai_result['path']).name}", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**检测结果**")
                            st.write(f"分析成功: {'✅' if ai_result.get('ai_analysis_success') else '❌'}")
                            st.write(f"疑似 AI 生成: {'⚠️' if ai_result.get('is_ai_generated_suspected') else '✅'}")
                            st.write(f"AI 生成分数: {ai_result.get('ai_gen_score', 0):.2f}")
                            st.write(f"检测器置信度: {ai_result.get('ai_detector_confidence', 0):.2f}")
                        
                        with col2:
                            st.markdown("**风险评估**")
                            risk = ai_result.get('forgery_risk_level', 'unknown')
                            risk_icon = {"low": "✅", "medium": "⚠️", "high": "❌"}.get(risk, "❓")
                            st.write(f"风险等级: {risk_icon} {risk.upper()}")
                            st.write(f"伪造类型: {ai_result.get('forgery_type', '-')}")
                            st.write(f"检测器: {ai_result.get('ai_detector_name', '-')}")
                        
                        reason = ai_result.get('forgery_reason', '')
                        if reason:
                            st.markdown("**原因说明**")
                            st.write(reason)
                        
                        forensic = ai_result.get('forensic_signals', {})
                        if forensic:
                            st.markdown("**取证信号**")
                            st.json(forensic)
                        
                        if not ai_result.get('ai_analysis_success'):
                            st.error(f"检测失败: {ai_result.get('ai_error', '未知错误')}")
            except Exception as e:
                _fail("ai_fake_detection", e)
                st.error(f"ai_fake_detection 失败：{e}")
    s5.caption("AI生成/真实性检测")
    
    # Step6 fusion
    fusion_disabled = not dbg["step_ok"]["ai_fake_detection"]
    if s6.button("Step6 fusion", disabled=fusion_disabled):
        try:
            mm_results = ctx.get("mm_results", [])
            ai_results = ctx.get("ai_results", [])
            ctx["fusion"] = fusion_engine(ctx["comp"], ctx["thresholds"], mm_results, ai_results, None)
            _ok("fusion")
            st.success("fusion 成功")
            st.json(ctx["fusion"])
        except Exception as e:
            _fail("fusion", e)
            st.error(f"fusion 失败：{e}")
    s6.caption("汇总风险等级")
    
    # Step7 evidence
    if s7.button("Step7 evidence", disabled=not dbg["step_ok"]["fusion"]):
        try:
            mm_results = ctx.get("mm_results", [])
            ai_results = ctx.get("ai_results", [])
            ctx["evidence_text"] = evidence_writer(ctx["comp"], None, ctx["fusion"], mm_results, ai_results)
            ctx["object_evidence"] = build_object_evidence(ctx["comp"], None, ctx["evidence_text"], mm_results, ai_results, ctx["fusion"])
            ctx["result"] = {
                "trace_id": ctx["req"]["trace_id"],
                "object_match": ctx["fusion"]["object_match"],
                "confidence_score": ctx["fusion"]["confidence_score"],
                "image_risk_level": ctx["fusion"]["image_risk_level"],
                "evidence_text": ctx["evidence_text"],
                "evidence_compliance": ctx["comp"],
                "object_consistency": None,
                "object_evidence": ctx["object_evidence"],
            }
            _ok("evidence")
            st.success("evidence 成功，已生成最终结果")
            st.json(ctx["result"])
        except Exception as e:
            _fail("evidence", e)
            st.error(f"evidence 失败：{e}")
    s7.caption("生成结构化证据输出")
    
    if s8.button("重置调试状态"):
        st.session_state["debug_state"] = {
            "step_ok": {k: False for k in dbg["step_ok"]},
            "error": {},
            "ctx": {},
        }
        st.success("调试状态已重置")
        st.rerun()
    s10.caption("清空当前分步执行状态")

    st.markdown("### 每步状态")
    order = ["ingest", "preprocess", "compliance", "multimodal_analysis", "ai_fake_detection", "fusion", "evidence"]
    cols = st.columns(len(order))
    for i, n in enumerate(order):
        if dbg["step_ok"][n]:
            cols[i].success(n)
        elif n in dbg["error"]:
            cols[i].error(n)
        else:
            cols[i].info(n)
    if dbg["error"]:
        st.markdown("### 失败原因")
        for k, v in dbg["error"].items():
            st.write(f"- {k}: {v}")

    if dbg["step_ok"]["evidence"] and "result" in ctx:
        if st.button("同步结果到业务视图案件预览"):
            st.session_state["business_result"] = ctx["result"]
            st.success("已同步。返回业务视图可查看案件预览。")


def main():
    _init_state()
    _render_sidebar()
    page = st.session_state["app_page"]
    if page == "首页":
        _home_page()
    elif page == "业务视图":
        _business_page()
    else:
        _debug_page()


if __name__ == "__main__":
    main()