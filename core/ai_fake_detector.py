"""
core/ai_fake_detector.py

AI生成/真实性检测模块
负责：对退款图逐张判断是否疑似 AI 生成 / 强生成式编辑

新主链：ingest -> preprocess -> compliance -> multimodal_analysis -> ai_fake_detector -> fusion -> evidence
"""

from typing import Dict, List, Optional
import random
import json
import subprocess
from pathlib import Path


# ==================== 错误兜底 ====================

def _get_default_ai_error_result(image_path: str, declared_shot_type: str, error_msg: str) -> Dict:
    """
    AI检测失败时的默认返回结构
    """
    return {
        "path": image_path,
        "declared_shot_type": declared_shot_type,

        "ai_analysis_success": False,
        "ai_error": error_msg,

        "is_ai_generated_suspected": False,
        "ai_gen_score": 0.0,
        "ai_detector_confidence": 0.0,
        "ai_detector_name": "",

        "forgery_risk_level": "unknown",
        "forgery_type": "unknown",
        "forgery_reason": f"AI真伪检测失败：{error_msg}",

        "forensic_signals": {},
    }


# ==================== Mock 检测器 ====================

def _mock_detector(image_path: str, detector_config: Dict) -> Dict:
    """
    Mock 检测器 - 用于开发和测试
    
    返回模拟的 AI 检测结果，随机生成但保持合理分布
    """
    # 使用图像路径作为随机种子，保证同一图像结果一致
    seed = hash(image_path) % (2**32)
    rng = random.Random(seed)
    
    # 模拟 AI 生成分数 (大多数情况应该是低分)
    # 80% 概率低分 (0.0-0.3), 15% 概率中等 (0.3-0.6), 5% 概率高分 (0.6-1.0)
    rand_val = rng.random()
    if rand_val < 0.80:
        ai_gen_score = rng.uniform(0.0, 0.3)
    elif rand_val < 0.95:
        ai_gen_score = rng.uniform(0.3, 0.6)
    else:
        ai_gen_score = rng.uniform(0.6, 0.95)
    
    # 模拟检测器置信度
    ai_detector_confidence = rng.uniform(0.6, 0.95)
    
    # 判断是否疑似 AI 生成
    is_ai_generated_suspected = ai_gen_score > 0.5
    
    # 确定风险等级
    if ai_gen_score < 0.3:
        forgery_risk_level = "low"
        forgery_type = "none"
        forgery_reason = "未检测到明显生成痕迹"
    elif ai_gen_score < 0.6:
        forgery_risk_level = "medium"
        forgery_type = "heavy_edit"
        forgery_reason = "检测到一定编辑痕迹，建议人工复核"
    else:
        forgery_risk_level = "high"
        forgery_type = "full_generation"
        forgery_reason = "检测到较强AI生成特征，存在较高伪造风险"
    
    # 模拟取证信号
    forensic_signals = {
        "texture_artifact_score": round(rng.uniform(0.0, ai_gen_score + 0.1), 4),
        "frequency_artifact_score": round(rng.uniform(0.0, ai_gen_score + 0.1), 4),
        "generator_prior_score": round(rng.uniform(0.0, ai_gen_score + 0.15), 4),
    }
    
    return {
        "ai_gen_score": round(ai_gen_score, 4),
        "ai_detector_confidence": round(ai_detector_confidence, 4),
        "ai_detector_name": "mock",
        "is_ai_generated_suspected": is_ai_generated_suspected,
        "forgery_risk_level": forgery_risk_level,
        "forgery_type": forgery_type,
        "forgery_reason": forgery_reason,
        "forensic_signals": forensic_signals,
    }


# ==================== 本地模型检测器 ====================

def _run_local_univfd(image_path: str, detector_config: Dict) -> Dict:
    """
    本地 UnivFD 检测器
    通过 subprocess 调用独立 conda 环境中的 single_infer.py
    """

    image_path = str(Path(image_path).resolve())

    # 这几个路径你后面可以放到 ai_detector.json 里配置
    python_exe = detector_config.get(
        "python_exe",
        r"D:\python\anaconda\envs\univfd\python.exe"
    )
    script_path = detector_config.get(
        "script_path",
        r"E:\detector_test\UniversalFakeDetect\single_infer.py"
    )
    ckpt_path = detector_config.get(
        "model_path",
        r"E:\detector_test\UniversalFakeDetect\pretrained_weights\fc_weights.pth"
    )
    arch = detector_config.get("arch", "CLIP:ViT-L/14")
    device = detector_config.get("device", "cuda")
    timeout_sec = int(detector_config.get("timeout_sec", 60))

    cmd = [
        python_exe,
        script_path,
        "--image_path", image_path,
        "--ckpt", ckpt_path,
        "--arch", arch,
        "--device", device,
    ]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8"
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"UnivFD subprocess failed: {proc.stderr or proc.stdout}"
        )

    try:
        payload = json.loads(proc.stdout.strip())
    except Exception as e:
        raise RuntimeError(f"UnivFD JSON parse failed: {e}, raw={proc.stdout}")

    if not payload.get("ok", False):
        raise RuntimeError(payload.get("error", "UnivFD infer failed"))

    ai_gen_score = float(payload.get("score", 0.0))

    # 先用一个临时阈值版本，后面再根据你自己的数据调
    if ai_gen_score < 0.30:
        forgery_risk_level = "low"
        forgery_type = "none"
        forgery_reason = "未检测到明显生成痕迹"
    elif ai_gen_score < 0.60:
        forgery_risk_level = "medium"
        forgery_type = "heavy_edit"
        forgery_reason = "检测到一定生成/编辑嫌疑，建议人工复核"
    else:
        forgery_risk_level = "high"
        forgery_type = "full_generation"
        forgery_reason = "检测到较强AI生成嫌疑，存在较高伪造风险"

    return {
        "ai_gen_score": round(ai_gen_score, 4),
        "ai_detector_confidence": round(max(0.5, min(0.99, abs(ai_gen_score - 0.5) * 2)), 4),
        "ai_detector_name": "local_univfd",
        "is_ai_generated_suspected": ai_gen_score >= 0.5,
        "forgery_risk_level": forgery_risk_level,
        "forgery_type": forgery_type,
        "forgery_reason": forgery_reason,
        "forensic_signals": {
            "univfd_raw_score": payload.get("raw_score"),
            "univfd_prob_score": round(ai_gen_score, 6),
            "device": payload.get("device", ""),
            "arch": payload.get("arch", ""),
        },
    }


# ==================== API 检测器（预留） ====================

def _run_api_detector(image_path: str, detector_config: Dict) -> Dict:
    """
    API 检测器 - 预留接口
    
    TODO: 接入第三方 API
    """
    raise NotImplementedError("API 检测器尚未实现")


# ==================== 检测器路由 ====================

def _run_detector(image_path: str, detector_config: Dict) -> Dict:
    """
    检测器路由 - 根据配置选择检测器
    """
    detector_type = detector_config.get("detector_type", "mock")

    if detector_type == "mock":
        return _mock_detector(image_path, detector_config)
    elif detector_type == "local_univfd":
        return _run_local_univfd(image_path, detector_config)
    elif detector_type == "api":
        return _run_api_detector(image_path, detector_config)
    else:
        raise ValueError(f"不支持的 detector_type: {detector_type}")


# ==================== 单张图分析 ====================

def analyze_single_image_ai_fake(
    image_path: str,
    declared_shot_type: str,
    product_name: str = "",
    refund_reason: str = "",
    detector_config: Optional[Dict] = None,
) -> Dict:
    """
    单张退款图的 AIGC/真实性检测
    
    Args:
        image_path: 图像路径
        declared_shot_type: 声明的图像类型
        product_name: 商品名称（可选，用于上下文）
        refund_reason: 退款原因（可选，用于上下文）
        detector_config: 检测器配置
    
    Returns:
        AI 检测结果字典
    """
    if detector_config is None:
        detector_config = {"detector_type": "mock"}
    
    try:
        # 调用检测器
        detector_result = _run_detector(image_path, detector_config)
        
        # 构建完整结果
        result = {
            "path": image_path,
            "declared_shot_type": declared_shot_type,
            
            "ai_analysis_success": True,
            "ai_error": None,
            
            "is_ai_generated_suspected": detector_result.get("is_ai_generated_suspected", False),
            "ai_gen_score": detector_result.get("ai_gen_score", 0.0),
            "ai_detector_confidence": detector_result.get("ai_detector_confidence", 0.0),
            "ai_detector_name": detector_result.get("ai_detector_name", ""),
            
            "forgery_risk_level": detector_result.get("forgery_risk_level", "unknown"),
            "forgery_type": detector_result.get("forgery_type", "unknown"),
            "forgery_reason": detector_result.get("forgery_reason", ""),
            
            "forensic_signals": detector_result.get("forensic_signals", {}),
        }
        
        return result
        
    except Exception as e:
        return _get_default_ai_error_result(image_path, declared_shot_type, str(e))


# ==================== 批量分析 ====================

def analyze_refund_images_ai_fake(
    refund_images: List[str],
    refund_image_meta: Dict[str, Dict],
    product_name: str = "",
    refund_reason: str = "",
    detector_config: Optional[Dict] = None,
) -> List[Dict]:
    """
    对退款图逐张进行 AI 生成/真实性风险检测，输出统一结构
    
    Args:
        refund_images: 退款图路径列表
        refund_image_meta: 退款图元信息字典 {path: {"shot_type": xxx, ...}}
        product_name: 商品名称
        refund_reason: 退款原因
        detector_config: 检测器配置
    
    Returns:
        每张图的 AI 检测结果列表
    """
    if detector_config is None:
        detector_config = {"detector_type": "mock"}
    
    results = []
    
    for image_path in refund_images:
        # 获取该图的声明类型
        meta = refund_image_meta.get(image_path, {})
        declared_shot_type = meta.get("shot_type", "unknown")
        
        # 单张分析
        result = analyze_single_image_ai_fake(
            image_path=image_path,
            declared_shot_type=declared_shot_type,
            product_name=product_name,
            refund_reason=refund_reason,
            detector_config=detector_config,
        )
        
        results.append(result)
    
    return results