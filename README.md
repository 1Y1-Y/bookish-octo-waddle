# 🌿 无人机烟叶病虫害智能诊断系统

基于**深度学习 + 大语言模型**的烟草病害智能诊断系统。面向烟农，实现从"无人机/手机拍图 → 病害识别 → AI 决策 → 施药建议"的全链路闭环，助力精准植保、增产增收。

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ 功能特性

| 模块 | 说明 |
|---|---|
| 🔬 **病害识别** | MobileNetV3，识别 8 类烟草病害，分组划分准确率 **90%** |
| 📷 **航拍诊断** | 大图滑窗切片分类，生成病害分布热力图 |
| 📍 **病害定位** | 像素坐标 → 经纬度映射，定位病害地块 |
| 🧠 **AI 专家问答** | 接入 DeepSeek（RAG 架构），烟农用大白话问、专家答 |
| 🎤 **语音咨询** | Whisper 语音识别，不会打字也能提问 |
| 🌦️ **发病预警** | 接入气象数据，预测未来发病风险 |
| 🖥️ **可视化网页** | Gradio 网页界面，开箱即用 |

---

## 🏗️ 系统架构

```
无人机/手机拍摄
      ↓
① 航拍大图滑窗切片（224×224）
      ↓
② MobileNetV3 分类模型识别病害
      ↓
③ 病害分布热力图 + GPS 定位
      ↓
④ 知识库 + DeepSeek（RAG）生成施药与预防建议
      ↓
⑤ 通俗化诊断报告（供烟农使用）
```

---

## 📊 实验结果（严谨评估）

| 实验 | 结果 |
|---|---|
| **数据泄露分析** | 随机划分 96.6% → **分组划分（无泄露）90.0%** |
| **多主干网络对比** | ResNet18(11.2M, 90.0%) / ResNet50(23.5M, 92.8%) / **MobileNetV3(1.5M, 92.0%)** |
| **K 折交叉验证** | **86.5% ± 6.5%**（真实泛化能力） |
| **类别不平衡处理** | 过采样最优 92.8%（+2.8%） |
| **鲁棒性测试** | 模糊/暗光/过曝下 77%~89%（增强后） |

> 核心亮点：发现并解决了**数据泄露**问题，诚实报告真实泛化能力。详见 [`补充实验报告.md`](补充实验报告.md)。

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
# torch 需装 CUDA 版（GPU），详见 requirements.txt 说明
```

### 2. 配置 API Key
```bash
# 复制示例配置，填入你的 DeepSeek API Key（https://platform.deepseek.com）
cp config.example.py config.py
```

### 3. 启动网页（推荐）
```bash
python web_demo.py
# 浏览器打开 http://127.0.0.1:7860
```
或双击 `启动系统.bat` 一键启动。

### 4. 命令行使用
```bash
# AI 专家问答
python ai_expert.py "烟叶发黄怎么办"

# 航拍图一体化诊断（识别+定位+建议）
python analyze_full.py 图片.jpg --lat 25.03 --lon 102.71 --alt 10

# 发病预警
python weather_alert.py 昆明
```

---

## 📁 项目结构

```
├── web_demo.py           # ⭐ 网页界面（4 个功能标签页）
├── ai_expert.py          # AI 专家问答（DeepSeek）
├── analyze_full.py       # 航拍一体化诊断（识别+定位+建议）
├── predict_aerial.py     # 航拍切片识别 + 热力图
├── agent.py              # 诊断报告生成
├── voice.py              # 语音识别（Whisper）
├── weather_alert.py      # 发病风险预警
├── geo_locate.py         # GPS 定位（像素→经纬度）
├── flight_calc.py        # 飞行参数/GSD 计算器
├── tobacco_knowledge.py  # 烟草全流程知识库
├── disease_knowledge.py  # 8 类病害防治知识库
├── run_experiment.py     # 通用训练/对比实验脚本
├── kfold.py              # K 折交叉验证
├── train_robust.py       # 鲁棒性增强训练
├── robustness_test.py    # 鲁棒性测试
├── models/cls_model.pt   # 训练好的模型权重
└── ...
```

---

## 🛠️ 技术栈

- **深度学习**：PyTorch、torchvision、MobileNetV3、ResNet、迁移学习
- **计算机视觉**：OpenCV、滑窗切片、数据增强
- **大模型**：DeepSeek API、RAG（检索增强生成）、Prompt 工程
- **语音**：faster-whisper
- **Web**：Gradio
- **其他**：NumPy、Pandas、气象 API、GPS/EXIF 解析

---

## ⚠️ 说明与局限

- 模型基于近景叶片数据训练，**尚未在真实无人机航拍图上验证**
- 少样本类（黑胫病等）准确率偏低，需补充数据
- 系统定位为"离线采集 + 落地分析"，适配田间无网络场景

---

## 📄 License

本项目采用 [MIT License](LICENSE)。
