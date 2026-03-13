# 图像证据核验模块

用于核验退款图在证据完整性、商品一致性、属性一致性、损坏证据上的可用性。

## 项目结构

```
E-com/
├── streamlit_app.py        # Streamlit 前端应用
├── core/                   # 核心模块
│   ├── compliance.py       # 证据合规性检查
│   ├── embeddings.py       # 向量嵌入
│   ├── evidence.py         # 证据输出
│   ├── fusion.py           # 融合模块
│   ├── ingest.py           # 数据摄入
│   ├── multimodal_analyzer.py  # 多模态分析
│   ├── multimodal_prompts.py   # 多模态提示词
│   ├── object_module.py    # 对象一致性模块
│   ├── preprocess.py       # 预处理
│   └── prompts.py          # 提示词加载
├── config/                 # 配置文件
│   ├── prompt_templates.json
│   └── thresholds.json
└── utils/                  # 工具函数
    ├── io_utils.py
    └── math_utils.py
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
streamlit run streamlit_app.py
```

## 退款图类型

| 类型 | 标签 | 必传 |
|------|------|------|
| damage_closeup | 商品损坏细节图 | ✓ |
| damage_with_order_link | 商品与订单/商家关联图 | ✓ |
| overview | 商品整体图 | 推荐 |
| merchant_label_closeup | 商家标签/包装近景图 | 推荐 |
| auxiliary | 其他辅助图 | 可选 |

## 参考图类型

| 类型 | 标签 |
|------|------|
| product_reference | 商品主体参考图 |
| order_link_reference | 包装/标签参考图 |

## 配置

首次运行会自动创建 `config/multimodal.json`，需配置 API：

```json
{
  "use_multimodal_api": true,
  "provider": "openai_compatible",
  "api_base": "your_api_base",
  "api_key": "your_api_key",
  "model_name": "gpt-4o-mini"
}
```

## 模型

CLIP 模型需从 [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32) 下载，放置在 `models/clip-vit-base-patch32/` 目录。

## License

MIT