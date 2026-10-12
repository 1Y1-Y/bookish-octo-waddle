import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from tobacco_knowledge import TOBACCO_KNOWLEDGE
from disease_knowledge import KNOWLEDGE

try:
    from config import DEEPSEEK_API_KEY as _cfg_key
except ImportError:
    _cfg_key = ""
API_KEY = os.environ.get("DEEPSEEK_API_KEY", _cfg_key)
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

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


def analyze_image(path):
    from predict_aerial import load_model, scan_image, CLASSES
    import cv2
    model = load_model()
    img = cv2.imread(path)
    if img is None:
        return "图片打不开，检查一下路径对不对。"
    counts, total = scan_image(model, img)
    if total == 0:
        return "这张图没看出明显的病害。"
    parts = []
    for i, n in counts.items():
        parts.append(f"{KNOWLEDGE[CLASSES[i]]['中文名']} {n}处")
    return "从这张航拍图看，检测到：" + "、".join(parts) + "。"


def main():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def ask(q):
        messages.append({"role": "user", "content": q})
        resp = client.chat.completions.create(model=MODEL, messages=messages)
        ans = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": ans})
        return ans

    if len(sys.argv) > 1:
        ans = ask(" ".join(sys.argv[1:]))
        with open("expert_answer.txt", "w", encoding="utf-8") as f:
            f.write(ans)
        print(ans)
        return

    print("=" * 44)
    print("   烟草种植专家（AI）已就绪")
    print("   直接输入问题，或输入：分析图片 <路径>")
    print("   输入 q 退出")
    print("=" * 44)
    while True:
        try:
            q = input("\n你：").strip()
        except EOFError:
            break
        if q.lower() in ("q", "quit", "退出"):
            break
        if q.startswith("分析图片"):
            path = q.replace("分析图片", "").strip()
            result = analyze_image(path)
            print("\n[识别结果] " + result)
            print("\n专家：" + ask("这张烟田的检测结果是：" + result + "。请给烟农完整的防治和用药建议，说清楚打什么药、怎么打。"))
        else:
            print("\n专家：" + ask(q))


if __name__ == "__main__":
    main()
