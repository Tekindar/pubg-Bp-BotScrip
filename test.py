# test.py —— 模板匹配度测试
# 用法：
#   python test.py                      # 默认比较 debug_lobby.png 和 start_btn.png
#   python test.py 大图名 模板名         # 指定文件，都在 config.py 的 IMAGE_DIR 里
#   python test.py debug_plane.png naozhong.png 0.5   # 附带自定义阈值，打印是否命中
#
# 判定逻辑与脚本完全一致：cv2.matchTemplate + TM_CCOEFF_NORMED，取最高分

import os
import sys
import numpy as np
import cv2

from rapidocr_onnxruntime import RapidOCR
from config import Config

IMAGE_DIR = Config.IMAGE_DIR

ocr = RapidOCR()
def imread_u(path):
    """cv2.imread 的 Unicode 安全替代，支持中文路径"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def m():
    big_name = "test2.png"
    tpl_name = "f.png"
    thresh = float(sys.argv[3]) if len(sys.argv) > 3 else None

    big_path = os.path.join(IMAGE_DIR, big_name)
    tpl_path = os.path.join(IMAGE_DIR, tpl_name)

    if not os.path.exists(big_path):
        print(f"❌ 找不到大图: {big_path}")
        sys.exit(1)
    if not os.path.exists(tpl_path):
        print(f"❌ 找不到模板: {tpl_path}")
        sys.exit(1)

    big = imread_u(big_path)
    tpl = imread_u(tpl_path)

    if big is None:
        print(f"❌ 大图读不出来（可能损坏或格式问题）: {big_path}")
        sys.exit(1)
    if tpl is None:
        print(f"❌ 模板读不出来: {tpl_path}")
        sys.exit(1)

    # 与 capture_game_screen 一致：大图统一到 1920x1080 虚拟坐标系
    if big.shape[1] != 1920 or big.shape[0] != 1080:
        big = cv2.resize(big, (1920, 1080), interpolation=cv2.INTER_AREA)

    print(f"大图: {big_name} ({big.shape[1]}x{big.shape[0]})")
    print(f"模板: {tpl_name} ({tpl.shape[1]}x{tpl.shape[0]})")

    if tpl.shape[0] >= big.shape[0] or tpl.shape[1] >= big.shape[1]:
        print("❌ 模板比大图还大，检查是不是文件选反了")
        sys.exit(1)

    result = cv2.matchTemplate(big, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    h, w = tpl.shape[:2]
    print(f"\n匹配最高分: {max_val:.4f}")
    print(f"命中位置: 模板左上角在 ({max_loc[0]}, {max_loc[1]})，中心在 "
          f"({max_loc[0] + w // 2}, {max_loc[1] + h // 2})")

    if thresh is not None:
        hit = max_val >= thresh
        print(f"阈值 {thresh:.2f}: {'✅ 命中' if hit else '❌ 未命中'}")
    else:
        if max_val >= 0.8:
            print("评价: 模板很准，放心用")
        elif max_val >= 0.65:
            print("评价: 接近可用，建议把 config.py 阈值降到 0.6~0.65")
        elif max_val >= 0.4:
            print("评价: 有点模糊，可能是缩放/画质差异，考虑重裁模板")
        else:
            print("评价: 完全对不上，必须重新裁图")

def n():
        
        path = os.path.join(IMAGE_DIR, "image.png")
        game_img = imread_u(path)
        if game_img is None:
            return False
        
        h, w = game_img.shape[:2]
        roi = game_img[h // 2 - int(h * 0.2): h // 2 + int(h * 0.3), w // 2 - int(h * 0.2): w // 2 + int(h * 0.3)]
        if roi.size == 0:
            return False
        roi = cv2.resize(roi, None, fx=2, fy=2)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 白底黑字：亮字变暗、暗底变亮，阈值二值化后就是标准 OCR 输入
        _, roi = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result, _ = ocr(roi)
        print(result)

        if not result:
            return False
        return any('F' in str(item[1]).upper() for item in result) or any('推门' in str(item[1]).upper() for item in result)

print(n())


