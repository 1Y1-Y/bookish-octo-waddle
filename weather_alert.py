import urllib.request
import urllib.parse
import json

DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean"

RULES = {
    "黑斑病": {"气象": "高温高湿", "check": lambda tmax, tmin, rain, hum: 25 <= tmax <= 32 and hum >= 80},
    "黑胫病": {"气象": "高温高湿", "check": lambda tmax, tmin, rain, hum: 28 <= tmax <= 36 and hum >= 78},
    "叶斑病": {"气象": "温暖潮湿", "check": lambda tmax, tmin, rain, hum: 20 <= tmax <= 29 and hum >= 80},
    "赤星病": {"气象": "高温多雨", "check": lambda tmax, tmin, rain, hum: 24 <= tmax <= 32 and rain >= 1 and hum >= 75},
    "花叶病毒病": {"气象": "干热虫媒活跃", "check": lambda tmax, tmin, rain, hum: 24 <= tmax <= 34 and hum < 68},
    "卷叶病": {"气象": "干热虫媒活跃", "check": lambda tmax, tmin, rain, hum: 24 <= tmax <= 34 and hum < 68},
    "黄曲叶病": {"气象": "干热虫媒活跃", "check": lambda tmax, tmin, rain, hum: 24 <= tmax <= 34 and hum < 68},
}


def geocode(city):
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": city, "count": 1, "language": "zh"}
    )
    with urllib.request.urlopen(url, timeout=15) as r:
        d = json.load(r)
    res = d.get("results")
    if not res:
        return None
    r0 = res[0]
    return r0["latitude"], r0["longitude"], r0.get("name", city)


def get_forecast(lat, lon, days=7):
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "daily": DAILY,
            "timezone": "Asia/Shanghai",
            "forecast_days": days,
        }
    )
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


def assess(city, days=7):
    loc = geocode(city)
    if loc is None:
        return f"没找到地名「{city}」，换个说法试试（比如「昆明」「玉溪」）。"
    lat, lon, name = loc
    data = get_forecast(lat, lon, days)
    daily = data["daily"]
    dates = daily["time"]
    tmaxs = daily["temperature_2m_max"]
    tmins = daily["temperature_2m_min"]
    rains = daily["precipitation_sum"]
    hums = daily["relative_humidity_2m_mean"]

    lines = []
    lines.append("=" * 40)
    lines.append(f"    {name} 未来{days}天 病害发病风险预警")
    lines.append("=" * 40)
    lines.append("")
    lines.append("【天气预报】")
    for i in range(len(dates)):
        lines.append(
            f"  {dates[i]}：{tmins[i]:.0f}~{tmaxs[i]:.0f}℃  降雨{rains[i]:.1f}mm  湿度{hums[i]:.0f}%"
        )
    lines.append("")

    risk_days = {d: [] for d in RULES}
    for i in range(len(dates)):
        for d, rule in RULES.items():
            if rule["check"](tmaxs[i], tmins[i], rains[i], hums[i]):
                risk_days[d].append(dates[i])

    lines.append("【发病风险预警】")
    any_risk = False
    for d, days_hit in risk_days.items():
        if days_hit:
            any_risk = True
            level = "高" if len(days_hit) >= 3 else "中" if len(days_hit) >= 1 else "低"
            lines.append(f"  ⚠ {d}：{level}风险（{len(days_hit)}天符合发病条件：{RULES[d]['气象']}）")
    if not any_risk:
        lines.append("  未来几天气象条件不利于病害流行，风险较低，保持日常管理即可。")
    lines.append("")

    lines.append("【预防建议】")
    if risk_days["赤星病"] or risk_days["黑斑病"] or risk_days["叶斑病"]:
        lines.append("  · 有真菌病风险：雨前雨后注意排水降湿，及时摘除下部老叶病叶，可提前喷保护性杀菌剂（代森锰锌）。")
    if risk_days["黑胫病"]:
        lines.append("  · 有黑胫病风险：注意田间不要积水，避免伤根，发现病株立即拔除并消毒土壤。")
    if risk_days["花叶病毒病"] or risk_days["卷叶病"] or risk_days["黄曲叶病"]:
        lines.append("  · 有病毒病风险：干热天气虫媒（烟粉虱、蚜虫）活跃，要及时防虫，清除田间杂草和病株。")
    lines.append("")
    lines.append("提示：本预警基于气象条件推算，仅供参考，请结合田间实际情况判断。")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(assess(sys.argv[1]))
    else:
        print(assess("昆明"))
