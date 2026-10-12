"""GPS 定位模块：把航拍图上的病害像素坐标换算成经纬度（确定病害在田里的位置）。

原理：航拍图正射向下，已知拍摄点 GPS + 飞行高度推导的 GSD，
就能把"图上的像素位置"换算成"地面的经纬度位置"。
"""
import math
from PIL import Image
from PIL.ExifTags import GPSTAGS


def _to_deg(value):
    d, m, s = value
    return float(d) + float(m) / 60 + float(s) / 3600


def read_gps(image_path):
    """读取照片 EXIF 里的拍摄点 GPS，返回 (lat, lon) 或 None"""
    img = Image.open(image_path)
    exif = img.getexif()
    if not exif:
        return None
    gps_ifd = exif.get_ifd(0x8825)
    if not gps_ifd:
        return None
    gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
    lat = gps.get("GPSLatitude")
    lon = gps.get("GPSLongitude")
    if lat is None or lon is None:
        return None
    lat_ref = gps.get("GPSLatitudeRef", "N")
    lon_ref = gps.get("GPSLongitudeRef", "E")
    lat = _to_deg(lat) * (-1 if lat_ref == "S" else 1)
    lon = _to_deg(lon) * (-1 if lon_ref == "W" else 1)
    return lat, lon


def pixel_to_gps(px, py, img_w, img_h, gsd_m, center_lat, center_lon, heading_deg=0):
    """把图片像素坐标 (px,py) 换算为经纬度。

    gsd_m: 每像素多少米（由飞行高度和相机参数算出）
    center_lat/lon: 图片中心（拍摄点）的经纬度
    heading_deg: 无人机机头朝向（正北为0，顺时针），默认0（图上方即正北）
    """
    dx = px - img_w / 2
    dy = py - img_h / 2
    forward = -dy * gsd_m
    right = dx * gsd_m
    h = math.radians(heading_deg)
    north = forward * math.cos(h) - right * math.sin(h)
    east = forward * math.sin(h) + right * math.cos(h)
    dlat = north / 111320.0
    dlon = east / (111320.0 * math.cos(math.radians(center_lat)))
    return center_lat + dlat, center_lon + dlon


if __name__ == "__main__":
    print("=== GPS 定位模块 演示 ===")
    # 模拟：昆明附近某烟田，飞行高度10米(gsd约0.34cm=0.0034m)，图片4000x3000
    center_lat, center_lon = 25.0389, 102.7183
    gsd_m = 0.0034
    img_w, img_h = 4000, 3000
    print(f"图片中心（拍摄点）：{center_lat}, {center_lon}")
    print(f"GSD：{gsd_m*100:.2f} cm/像素，图像 {img_w}x{img_h}")
    print("\n图上几个检测框中心 → 对应经纬度：")
    for name, px, py in [("左上角病斑", 800, 500), ("中心", 2000, 1500), ("右下角病斑", 3200, 2500)]:
        lat, lon = pixel_to_gps(px, py, img_w, img_h, gsd_m, center_lat, center_lon)
        print(f"  {name} (px={px},py={py}) → {lat:.6f}, {lon:.6f}")
    print("\n说明：无人机拍的图若带 EXIF GPS，read_gps() 可自动读取拍摄点坐标。")
