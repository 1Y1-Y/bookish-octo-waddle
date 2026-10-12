import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr
import cv2
from openai import OpenAI
from tobacco_knowledge import TOBACCO_KNOWLEDGE
from disease_knowledge import KNOWLEDGE

try:
    from config import DEEPSEEK_API_KEY as _cfg_key
except ImportError:
    _cfg_key = ""
API_KEY = os.environ.get("DEEPSEEK_API_KEY", _cfg_key)
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

disease_text = "\n".join(
    f"{k['中文名']}（{name}）：症状-{k['症状']} 用药-{k['防治药剂']} 预防-{k['预防措施']}"
    for name, k in KNOWLEDGE.items()
)

SYSTEM_PROMPT = f"""你是一位经验丰富的烟草种植专家，服务对象是普通烟农（可能没太多文化）。
回答要求：
1. 用通俗易懂的大白话，避免专业术语，必要时打比方。
2. 回答要具体、可操作，告诉烟农"具体做什么、怎么做、用多少"。
3. 语气亲切、耐心，像村里的老技术员。
4. 涉及打药时提醒注意安全和采收前安全间隔期。

下面是你的专业知识库，请基于它回答：

{TOBACCO_KNOWLEDGE}

常见病害资料：
{disease_text}
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;900&display=swap');

.gradio-container {
    background: linear-gradient(160deg, #eaf7ea 0%, #e6f4f1 45%, #e8f0fb 100%) !important;
    font-family: 'Noto Sans SC', 'Microsoft YaHei', sans-serif !important;
    max-width: 100% !important;
    min-height: 100vh;
}

/* 顶部渐变横幅 */
#hero {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 45%, #43a047 75%, #66bb6a 100%);
    border-radius: 0 0 32px 32px;
    padding: 38px 24px 34px;
    text-align: center;
    box-shadow: 0 12px 40px rgba(27,94,32,0.3);
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}
#hero::after {
    content: "";
    position: absolute;
    top: -60%; right: -10%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%);
    border-radius: 50%;
}
#hero h1 {
    font-size: 2.3rem; font-weight: 900; margin: 0;
    color: #ffffff; letter-spacing: 1px;
    text-shadow: 0 2px 12px rgba(0,0,0,0.15);
}
#hero p {
    margin: 12px auto 0; max-width: 720px;
    color: rgba(255,255,255,0.95);
    font-size: 1.08rem; line-height: 1.7;
}
#hero .tags {
    margin-top: 18px; display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;
}
#hero .tag {
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.35);
    color: #fff; padding: 6px 16px; border-radius: 999px;
    font-size: 0.92rem; backdrop-filter: blur(6px);
}

/* 标签页 */
.tab-nav {
    border-bottom: 2px solid rgba(76,175,80,0.2) !important;
    gap: 6px !important;
    padding: 0 12px !important;
}
.tab-nav button {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #33691e !important;
    border: none !important;
    border-radius: 14px 14px 0 0 !important;
    padding: 12px 22px !important;
    transition: all 0.2s ease !important;
    background: transparent !important;
}
.tab-nav button:hover {
    background: rgba(76,175,80,0.12) !important;
}
.tab-nav button.selected {
    background: linear-gradient(135deg, #2e7d32, #43a047) !important;
    color: #fff !important;
    box-shadow: 0 -4px 16px rgba(46,125,50,0.3) !important;
}

/* 毛玻璃卡片 */
.glass {
    background: rgba(255,255,255,0.72) !important;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.7) !important;
    border-radius: 22px !important;
    box-shadow: 0 10px 34px rgba(31,60,40,0.1) !important;
    padding: 20px !important;
}

/* 输入框 */
input[type="text"], textarea, .gr-textbox textarea, .gr-textbox input {
    border-radius: 12px !important;
    border: 1.5px solid rgba(76,175,80,0.25) !important;
    transition: all 0.2s !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #43a047 !important;
    box-shadow: 0 0 0 3px rgba(76,175,80,0.15) !important;
}

/* 主按钮 */
button.primary {
    background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%) !important;
    border: none !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 18px rgba(46,125,50,0.32) !important;
    transition: all 0.22s ease !important;
    padding: 12px 20px !important;
}
button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 26px rgba(46,125,50,0.42) !important;
}
button.primary:active {
    transform: translateY(0);
}

/* 示例按钮 */
.examples button, .gr-button {
    border-radius: 12px !important;
}

/* 底部 */
#foot {
    text-align: center; color: #6b8f71; font-size: 0.88rem;
    padding: 22px 0 10px;
}

/* 隐藏 gradio 页脚 */
footer { display: none !important; }
"""

_det_model = None


def get_model():
    global _det_model
    if _det_model is None:
        from predict_aerial import load_model
        _det_model = load_model()
    return _det_model


def ask(message, history=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in (history or []):
        if isinstance(item, dict):
            msgs.append({"role": item.get("role", "user"), "content": item.get("content", "")})
        else:
            u, a = item
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": message})
    resp = client.chat.completions.create(model="deepseek-chat", messages=msgs)
    return resp.choices[0].message.content


def chat_fn(message, history):
    return ask(message, history)


def voice_fn(audio_path):
    if not audio_path:
        return "请先按下录音按钮，说您的问题。", ""
    from voice import transcribe
    text = transcribe(audio_path)
    if not text:
        return "没听清，您再说一遍？", ""
    answer = ask(text)
    return text, answer


def image_fn(image):
    if image is None:
        return "请先上传一张烟田航拍图。"
    from predict_aerial import scan_image, CLASSES
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    model = get_model()
    counts, total = scan_image(model, bgr)
    if total == 0:
        return "没看出明显病害，这片烟田看起来比较健康。建议继续做好日常管理。"
    parts = []
    for i, n in counts.items():
        parts.append(f"{KNOWLEDGE[CLASSES[i]]['中文名']}（{n}处）")
    result = "从这张航拍图看，检测到：" + "、".join(parts)
    advice = ask(
        "这是一张烟田航拍图的检测结果：" + result + "。请给烟农一份完整、通俗的防治和用药建议，"
        "说清楚这是什么病、严重不严重、打什么药怎么打、平时怎么预防。"
    )
    return result + "\n\n" + "—" * 20 + "\n\n" + advice


def weather_fn(city):
    if not city or not city.strip():
        return "请输入您所在的地区，比如「昆明」。"
    from weather_alert import assess
    report = assess(city.strip())
    advice = ask(f"这是{city}未来天气的病害发病预警：{report}。请用大白话给烟农讲一下重点，该防什么、怎么防。")
    return report + "\n\n" + "—" * 20 + "\n\n" + advice


with gr.Blocks(title="烟草病虫害智能诊断专家") as demo:
    gr.HTML(
        """
        <div id="hero">
            <h1>🌿 烟草病虫害智能诊断专家</h1>
            <p>从育苗、施肥、防病到采收，全程给烟农出主意的 AI 农技专家</p>
            <div class="tags">
                <span class="tag">🔬 病害识别 96.6%</span>
                <span class="tag">🤖 AI 专家问答</span>
                <span class="tag">🎤 语音提问</span>
                <span class="tag">📷 航拍诊断</span>
                <span class="tag">🌦️ 发病预警</span>
            </div>
        </div>
        """
    )

    with gr.Tab("💬 病害咨询"):
        with gr.Group(elem_classes="glass"):
            gr.ChatInterface(
                fn=chat_fn,
                examples=[
                    "烟叶发黄是咋回事，咋办？",
                    "什么时候采收烟叶最好？",
                    "烟叶怎么施钾肥？",
                    "打顶留多少片叶子合适？",
                ],
            )
    with gr.Tab("🎤 语音咨询"):
        with gr.Group(elem_classes="glass"):
            gr.Markdown("### 不会打字？直接说话就行\n按下录音，说完点「识别并回答」。")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_in = gr.Audio(sources=["microphone"], type="filepath", label="按下录音说话")
                    voice_btn = gr.Button("🔊 识别并回答", variant="primary")
                with gr.Column(scale=1):
                    voice_text = gr.Textbox(label="我听到的是", lines=3)
                    voice_out = gr.Textbox(label="专家回答", lines=15)
            voice_btn.click(voice_fn, inputs=audio_in, outputs=[voice_text, voice_out])
    with gr.Tab("📷 航拍图诊断"):
        with gr.Group(elem_classes="glass"):
            gr.Markdown("### 上传烟田航拍图，自动识病、开药方")
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(label="上传烟田航拍图", type="numpy", height=320)
                    btn = gr.Button("🔍 开始诊断", variant="primary")
                with gr.Column(scale=1):
                    out = gr.Textbox(label="诊断结果与施药建议", lines=22)
            btn.click(image_fn, inputs=img_in, outputs=out)
    with gr.Tab("🌦️ 发病预警"):
        with gr.Group(elem_classes="glass"):
            gr.Markdown("### 输入你所在的地区，看看未来几天会不会闹病")
            with gr.Row():
                city_in = gr.Textbox(label="您所在的地区", value="昆明", scale=3)
                weather_btn = gr.Button("🌦️ 查看预警", variant="primary", scale=1)
            weather_out = gr.Textbox(label="发病风险预警与预防建议", lines=24)
            weather_btn.click(weather_fn, inputs=city_in, outputs=weather_out)

    gr.HTML('<div id="foot">© 烟叶病虫害智能诊断系统 · 让烟农种烟更省心</div>')

if __name__ == "__main__":
    demo.launch(css=CSS, server_name="127.0.0.1", server_port=7860, inbrowser=True)
