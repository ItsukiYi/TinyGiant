#!/usr/bin/env python3
"""下载 ArcFace ONNX 人脸识别模型（insightface buffalo_s 包中的 w600k_mbf.onnx）。

运行：python download_models.py
模型缺失时 picker 的 _get_face_recognizer() 会返回 None，识别功能优雅降级。
"""
import os, sys, zipfile, urllib.request, shutil
from pathlib import Path

MODELS = {
    'buffalo_s': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip',
}

def main():
    root = Path(__file__).parent
    target = root / 'face_recognizer.onnx'
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"已存在: {target} ({target.stat().st_size // 1024}KB)")
        return

    url = MODELS['buffalo_s']
    zip_path = root / 'buffalo_s.zip'
    extract_dir = root / '_buffalo_s'

    print(f"下载 {url} ...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"!! 下载失败: {e}")
        print("请手动下载 buffalo_s.zip 并解压其中的 w600k_mbf.onnx 为 face_recognizer.onnx")
        sys.exit(1)

    print("解压...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    # buffalo_s.zip 包含: det_500m.onnx(检测) w600k_mbf.onnx(识别) w600k_r50.onnx(识别-大)
    src = extract_dir / 'w600k_mbf.onnx'  # 轻量版 ~44MB
    if not src.exists():
        src = extract_dir / 'w600k_r50.onnx'
    if not src.exists():
        # 搜索任何 .onnx
        onnx_files = list(extract_dir.rglob('*.onnx'))
        # 排除检测模型（名称含 det）
        recog = [f for f in onnx_files if 'det' not in f.name.lower()]
        if recog:
            src = recog[0]

    if src and src.exists():
        shutil.copy2(src, target)
        print(f"已安装: {target} ({target.stat().st_size // 1024}KB)")
    else:
        print("!! 未在 buffalo_s.zip 中找到识别模型")
        sys.exit(1)

    # 清理
    zip_path.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)
    print("完成。")

if __name__ == '__main__':
    main()
