"""飞行参数计算器：根据相机参数计算 GSD / 飞行高度 / 覆盖范围。"""


def gsd_cm(altitude_m, sensor_width_mm, focal_length_mm, image_width_px):
    """每像素代表多少厘米（越小越清晰）"""
    return (sensor_width_mm * altitude_m) / (focal_length_mm * image_width_px) * 100


def height_for_gsd(target_gsd_cm, sensor_width_mm, focal_length_mm, image_width_px):
    """想要达到某个 GSD，需要飞多高（米）"""
    return target_gsd_cm * focal_length_mm * image_width_px / (sensor_width_mm * 100)


def ground_coverage(altitude_m, sensor_width_mm, sensor_height_mm, focal_length_mm):
    """单张照片覆盖的地面范围（米）：宽 × 高"""
    w = sensor_width_mm * altitude_m / focal_length_mm
    h = sensor_height_mm * altitude_m / focal_length_mm
    return w, h


if __name__ == "__main__":
    # 示例：1/2.3英寸传感器 6.17×4.62mm，实际焦距 4.5mm，4000×3000 像素
    Sw, Sh, f, W = 6.17, 4.62, 4.5, 4000
    print("=== 常用机型参考（传感器6.17×4.62mm, 焦距4.5mm, 4000×3000px）===")
    for alt in [5, 10, 15, 20, 30]:
        g = gsd_cm(alt, Sw, f, W)
        cw, ch = ground_coverage(alt, Sw, Sh, f)
        print(f"  高度 {alt:>2}m → GSD {g:.2f} cm/px，单张覆盖 {cw:.1f}×{ch:.1f} 米")

    print("\n=== 目标 GSD 所需高度 ===")
    for tg in [0.2, 0.3, 0.5]:
        h = height_for_gsd(tg, Sw, f, W)
        print(f"  想要 {tg} cm/px → 飞 {h:.1f} 米")

    print("\n注：请把 Sw/Sh/f/W 换成你实际无人机的参数，结果才准确。")
