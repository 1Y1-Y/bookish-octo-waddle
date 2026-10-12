"""端到端一体化：输入航拍图 → 识别病害 + 定位 + AI 建议，输出完整报告。

用法:
    python analyze_full.py 航拍图.jpg
    python analyze_full.py 航拍图.jpg --lat 25.03 --lon 102.71 --alt 10 --heading 0
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import cv2
from openai import OpenAI

from predict_aerial import load_model, scan_image, CLASSES, PATCH, STRIDE
from disease_knowledge import KNOWLEDGE
from geo_locate import pixel_to_gps, read_gps

try:
    from config import DEEPSEEK_API_KEY as _cfg_key
except ImportError:
    _cfg_key = ""
API_KEY = os.environ.get("DEEPSEEK_API_KEY", _cfg_key)
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")


def gsd_from_altitude(alt_m, sensor_w=6.17, focal=4.5, img_w=4000):
    return (sensor_w * alt_m) / (focal * img_w)


def analyze(image_path, lat=None, lon=None, alt=None, heading=0):
    img = cv2.imread(image_path)
    if img is None:
        return "图片打不开，检查路径。"

    h, w = img.shape[:2]
    model = load_model()
    counts, total = scan_image(model, img)
    lines = []
    lines.append("=" * 44)
    lines.append("        烟田航拍智能诊断报告")
    lines.append("=" * 44)
    lines.append(f"图像尺寸: {w}x{h}")

    if total == 0:
        lines.append("未检测到明显病害，烟田总体健康。")
        return "\n".join(lines)

    lines.append(f"分析区块: {total} 个")
    lines.append("")

    # 空间位置统计（图上位置）
    regions = {"左上": 0, "右上": 0, "左下": 0, "右下": 0}
    detections = []
    for y in range(0, h - PATCH + 1, STRIDE):
        for x in range(0, w - PATCH + 1, STRIDE):
            patch = img[y:y + PATCH, x:x + PATCH]
            from predict_aerial import TF
            import torch
            t = TF(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
            with torch.no_grad():
                out = model(t)
                prob = torch.softmax(out, dim=1)
                ci = int(out.argmax(1).item())
                conf = float(prob[0, ci].item())
            if conf < 0.5:
                continue
            cx, cy = x + PATCH // 2, y + PATCH // 2
            detections.append((cx, cy, ci, conf))
            rx = "左" if cx < w / 2 else "右"
            ry = "上" if cy < h / 2 else "下"
            regions[rx + ry] += 1

    disease_counts = {}
    for _, _, ci, _ in detections:
        name = CLASSES[ci]
        disease_counts[name] = disease_counts.get(name, 0) + 1

    lines.append("【一、病害分布】")
    for name, n in sorted(disease_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {KNOWLEDGE[name]['中文名']}：{n} 处")
    lines.append("")

    lines.append("【二、空间分布】")
    for r, n in regions.items():
        if n > 0:
            lines.append(f"  {r}：{n} 处")
    lines.append("")

    # GPS 定位
    gps_lat, gps_lon = lat, lon
    if gps_lat is None:
        g = read_gps(image_path)
        if g:
            gps_lat, gps_lon = g
    if gps_lat is not None and alt:
        gsd = gsd_from_altitude(alt)
        lines.append("【三、病害 GPS 定位】")
        lines.append(f"  拍摄点: {gps_lat:.6f}, {gps_lon:.6f}, 高度 {alt}m (GSD≈{gsd*100:.2f}cm/px)")
        for cx, cy, ci, conf in detections[:5]:
            dlat, dlon = pixel_to_gps(cx, cy, w, h, gsd, gps_lat, gps_lon, heading)
            lines.append(f"    {KNOWLEDGE[CLASSES[ci]]['中文名']} → {dlat:.6f}, {dlon:.6f}")
        lines.append("")

    # AI 建议
    summary = "、".join(f"{KNOWLEDGE[n]['中文名']} {c} 处" for n, c in disease_counts.items())
    prompt = (f"这是一张烟田航拍图的检测结果：{summary}。请给烟农一份简明、通俗的防治建议，"
              f"说清楚最要紧先处理哪个病、打什么药、平时怎么防。")
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
        )
        advice = resp.choices[0].message.content
    except Exception as e:
        advice = "（AI 建议获取失败：" + str(e)[:50] + "）"

    lines.append("【四、AI 防治建议】")
    lines.append(advice)

    report = "\n".join(lines)
    with open("航拍诊断报告.txt", "w", encoding="utf-8") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--alt", type=float, default=None)
    ap.add_argument("--heading", type=float, default=0)
    args = ap.parse_args()
    print(analyze(args.image, args.lat, args.lon, args.alt, args.heading))
    print("\n[报告已保存到 航拍诊断报告.txt]")
