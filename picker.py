#!/usr/bin/env python3
"""
tgphoto-pick — 极简连拍照片挑选工具 v2
"""

import os, sys, json, re, threading, webbrowser, argparse, io, traceback, importlib
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote, unquote
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
import pillow_heif

pillow_heif.register_heif_opener()

THUMB_W = 250
SKIP_DIRS = {'_tg_trash', '_tg_report', '_tg_cache', '__pycache__', 'jpgs', '视频', 'clip'}
PREVIEW_EXTS = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.HIF', '.hif', '.HEIC', '.heic'}

def is_real_photo(path: Path) -> bool:
    name = path.name
    if name.startswith('.__') or name.startswith('._'):
        return False
    try:
        return path.stat().st_size > 15000
    except OSError:
        return False

RE_NUM = re.compile(r'_?DSC(\d+)', re.IGNORECASE)

# ═══════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════

class Photo:
    _next_id: int = 0
    def __init__(self, preview: Path):
        self.preview = preview
        self.raw: Path | None = None
        self.keep = False
        self.star = False
        self.score = 0.0
        self.score_sharp = 0.0
        self.score_exp = 0.0
        self.score_contrast = 0.0
        self.score_face = 0.0
        self.tags: list = []
        self.burst: int = 0   # 时间组内的真连拍子序列号
        self.person_id: int | None = None  # coser 人物 ID（ArcFace，跨目录持久化）
        self.role_id: int | None = None  # 角色 ID（视觉相似度，跨目录持久化）
        self.merge_pending: bool = False  # 待确认候选（重构后：簇未确认/质心中置信）
        self.is_group: bool = False  # 是否为合照（含已知 coser + 路人）
        self.group_pids: list = []  # 合照中命中的所有已知 coser 的 person_id
        self._face_data: list = []  # 会话内暂存检测到的人脸（含 embedding），不入 to_dict
        self.id = Photo._next_id
        Photo._next_id += 1

    def find_raw(self):
        parent = self.preview.parent
        stem = self.preview.stem
        for ext in ('.ARW', '.arw'):
            p = parent / f"{stem}{ext}"
            if p.exists(): self.raw = p; return
            p = parent / "arw" / f"{stem}{ext}"
            if p.exists(): self.raw = p; return

    @property
    def num(self) -> int:
        m = RE_NUM.search(self.preview.stem)
        return int(m.group(1)) if m else 0

    @property
    def fmt(self) -> str:
        return 'HIF' if self.preview.suffix.upper() in ('.HIF', '.HEIC') else 'JPG'

    def to_dict(self):
        return {'id': self.id, 'name': self.preview.name, 'num': self.num,
                'path': str(self.preview), 'has_raw': self.raw is not None,
                'fmt': self.fmt, 'keep': self.keep, 'star': self.star,
                'score': round(self.score, 1), 's_sharp': round(self.score_sharp, 1),
                's_exp': round(self.score_exp, 1), 's_contrast': round(self.score_contrast, 1),
                's_face': round(self.score_face, 1), 'tags': list(self.tags),
                'burst': self.burst, 'person_id': self.person_id, 'role_id': self.role_id,
                'merge_pending': self.merge_pending, 'is_group': self.is_group,
                'group_pids': list(self.group_pids)}

# ═══════════════════════════════════════════════════════════
#  扫描 & 自动分组
# ═══════════════════════════════════════════════════════════

def _get_dt(path: Path) -> datetime | None:
    try:
        img = Image.open(path)
        exif = None
        try:
            exif = img._getexif()
        except AttributeError:
            exif = img.getexif()
        if exif:
            dt_str = exif.get(36867) or exif.get(306)
            if dt_str:
                return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    return None

def _best_dt(photo: Photo) -> datetime | None:
    dt = _get_dt(photo.preview)
    if dt: return dt
    if photo.preview.suffix.upper() not in ('.HIF', '.HEIC'):
        for sfx in ('.HIF', '.hif', '.HEIC', '.heic'):
            hif = photo.preview.with_suffix(sfx)
            if hif.exists():
                dt = _get_dt(hif)
                if dt: return dt
    if photo.raw:
        dt = _get_dt(photo.raw)
        if dt: return dt
    try:
        return datetime.fromtimestamp(photo.preview.stat().st_mtime)
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════
#  自动评分（人像为主 · 组内相对排名）
#  分数=该照片在本组连拍中的百分位(100=本组最佳)，清晰度测主脸/眼区，
#  曝光看死白死黑，表情看睁眼；相对排名消除 Laplacian 绝对标定难题。
# ═══════════════════════════════════════════════════════════

_SCORE_LONG = 1280   # 评分工作图长边
_FACE_LONG = 960     # 人脸检测图长边（InsightFace 内部 letterbox 到 640，此值过小会漏小脸）
_EAR_OPEN = 0.18     # EAR 回退阈值（InsightFace 106点 EAR，≈0.18 以下视为闭眼）
_BLINK_OPEN = 0.35   # blendshape 睁眼阈值（单眼 < 此值视为该眼睁开）
_BLINK_CLOSED = 0.50  # blendshape 闭眼阈值（单眼 >= 此值视为该眼闭合）
_NEAR_IDENTICAL_CV = 0.06  # 组内原始 quality 变异系数低于此值判近同质
_SUBBURST_FLOOR = 0.04     # 真连拍边界距离下限（低于此不切分，抗噪声）

def _lap_var(gray_crop) -> float:
    """Laplacian 方差，越大越锐利。长边 >256 时缩到 256 再算（只缩不放，避免放大小 crop 造细节）。"""
    import cv2
    if gray_crop is None or gray_crop.size < 16:
        return 0.0
    h, w = gray_crop.shape
    m = max(h, w)
    if m > 256:
        s = 256.0 / m
        gray_crop = cv2.resize(gray_crop, (max(1, int(w * s)), max(1, int(h * s))),
                               interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())

def _ear_from_landmarks(lm) -> float:
    """由关键点算 EAR（眼纵横比），<0.18 视为闭眼。lm 为 (x,y) 归一化坐标列表。"""
    try:
        le = [lm[i] for i in (33, 160, 158, 133, 153, 144)]
        re = [lm[i] for i in (362, 385, 387, 263, 373, 380)]
        def _ear(eye):
            x0, _ = eye[0]; x3, _ = eye[3]
            if abs(x0 - x3) <= 0:
                return 0.0
            return (abs(eye[1][1] - eye[5][1]) + abs(eye[2][1] - eye[4][1])) / (2 * abs(x0 - x3))
        return (_ear(le) + _ear(re)) / 2
    except Exception:
        return 0.0

def _blink_from_blendshapes(bs) -> tuple | None:
    """从 MediaPipe blendshape 取 (eyeBlinkLeft, eyeBlinkRight)，0-1 越大越闭。缺失返回 None。"""
    if not bs:
        return None
    try:
        d = {c.category_name: c.score for c in bs}
        l = d.get('eyeBlinkLeft'); r = d.get('eyeBlinkRight')
        if l is None or r is None:
            return None
        return (l, r)
    except Exception:
        return None

def _box_from_lm(lm, idx_x, idx_y, xpad=1.0, ypad=1.0) -> tuple:
    """由 landmark 索引取归一化框：横向取 idx_x 的 min/max，纵向取 idx_y 的 min/max，再加 pad。"""
    xs = [lm[i][0] for i in idx_x]; ys = [lm[i][1] for i in idx_y]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    bw, bh = max(1e-4, x1 - x0), max(1e-4, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (max(0.0, cx - bw * xpad / 2), max(0.0, cy - bh * ypad / 2),
            min(1.0, cx + bw * xpad / 2), min(1.0, cy + bh * ypad / 2))

def _percentile_rank(values: list) -> list:
    """稳健百分位 0..1（平均排名处理并列，对异常值不敏感）。"""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]
    out = []
    for v in values:
        less = sum(1 for x in values if x < v)
        eq = sum(1 for x in values if x == v)
        out.append((less + (eq - 1) / 2) / (n - 1))
    return out

def _resize_long(img, long_side):
    import cv2, numpy as np
    h, w = img.shape[:2]
    scale = long_side / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)

def _load_score_image(root, photo):
    """加载评分用工作图：优先 1600 预览缓存，落空读原文件，缩到长边 1280。"""
    import numpy as np
    from PIL import Image as _PIL
    try:
        cf = _cache_path(Path(str(root)), photo, 1600)
        if cf and cf.exists():
            pil = _PIL.open(cf).convert('RGB')
        else:
            pil = _PIL.open(photo.preview).convert('RGB')
        return _resize_long(np.array(pil), _SCORE_LONG)
    except Exception:
        return None

def _ear_from_106(lm106, eye_indices) -> float:
    """从 106 点 landmark 的眼睛轮廓算 EAR。eye_indices: [外角, 上1, 上2, 内角, 下2, 下1]。"""
    try:
        pts = [lm106[i] for i in eye_indices]
        # pts: (x, y) 像素坐标
        x0, _ = pts[0]; x3, _ = pts[3]
        if abs(x0 - x3) <= 0:
            return 0.3
        v1 = abs(pts[1][1] - pts[5][1])
        v2 = abs(pts[2][1] - pts[4][1])
        return (v1 + v2) / (2 * abs(x0 - x3))
    except Exception:
        return 0.3

# 106 点 landmark 眼睛轮廓索引（标准 InsightFace 106 布局）
_L_EYE_106 = [35, 36, 37, 38, 39, 40]   # 左眼外→上→内→下
_R_EYE_106 = [89, 90, 91, 92, 93, 94]   # 右眼外→上→内→下

def _detect_faces(rgb_small) -> list:
    """用 InsightFace buffalo_s 检测人脸，返回 [{box, lm, ear, area, blink, blink_l, blink_r, embedding}]。
    box/lm 为归一化坐标。insightface 缺失返回 []。"""
    try:
        import cv2, numpy as np
        app = _get_face_app()
        if app is None:
            return []
        h, w = rgb_small.shape[:2]
        bgr = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
        with _mp_lock:
            faces = app.get(bgr)
        out = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            box = (max(0, x1 / w), max(0, y1 / h), min(1, x2 / w), min(1, y2 / h))
            area = (box[2] - box[0]) * (box[3] - box[1])
            # 5 点 kps → 归一化，映射到 468 关键位
            kps = getattr(face, 'kps', None)
            lm_full = [(0.0, 0.0)] * 468
            if kps is not None and len(kps) >= 5:
                lm5 = [(float(kps[j][0]) / w if w > 0 else 0,
                        float(kps[j][1]) / h if h > 0 else 0) for j in range(5)]
                lm_full[33] = lm5[0]; lm_full[263] = lm5[1]
                lm_full[1] = lm5[2]; lm_full[61] = lm5[3]
                lm_full[199] = (lm5[3][0], min(1.0, lm5[3][1] + 0.15))
            # 106 点 landmark → 算 EAR → 转为"闭合度"(0=睁开, 1=闭合)，兼容 blendshape 阈值
            lm106 = getattr(face, 'landmark_2d_106', None)
            ear_l = _ear_from_106(lm106, _L_EYE_106) if lm106 is not None else 0.25
            ear_r = _ear_from_106(lm106, _R_EYE_106) if lm106 is not None else 0.25
            # EAR→闭合度：EAR≈0.30→0, EAR≈0.10→1（线性映射，clamp）
            blink_l = max(0.0, min(1.0, (0.30 - ear_l) / 0.20))
            blink_r = max(0.0, min(1.0, (0.30 - ear_r) / 0.20))
            ear = (ear_l + ear_r) / 2
            out.append({
                'box': box, 'lm': lm_full, 'ear': ear, 'area': area,
                'blink': None, 'blink_l': blink_l, 'blink_r': blink_r,
                'embedding': getattr(face, 'embedding', None),
                'det_score': float(face.det_score),
            })
        return out
    except Exception:
        return []

def _score_photo(root, photo, image=None, faces=None) -> dict:
    """单张原始指标（未归一化）。image/faces 可注入用于测试。"""
    import numpy as np, cv2
    if image is None:
        image = _load_score_image(root, photo)
    if image is None:
        return {'lap': 0.0, 'exp': 0.0, 'contrast': 0.0, 'face': 0.0, 'tags': ['废片'],
                'person_id': None, 'blink_l': None, 'blink_r': None}
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    total = H * W
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    shadow = float(np.sum(hist[:15])) / total
    highlight = float(np.sum(hist[240:])) / total
    mean = float(gray.mean()) / 255.0
    std = float(gray.std())
    tags = []
    # 废片：全黑/镜头盖
    if std < 5:
        return {'lap': 0.0, 'exp': 0.0, 'contrast': 0.0, 'face': 0.0, 'tags': ['废片'],
                'person_id': None, 'blink_l': None, 'blink_r': None}
    # 人脸（重构后：compute_persons 先检测并填入 _face_data，此处复用避免重复检测）
    if faces is None:
        faces = (photo._face_data if photo is not None else None) or None
    if faces is None:
        small = _resize_long(image, _FACE_LONG)
        faces = _detect_faces(small)
        if photo is not None:
            photo._face_data = list(faces) if faces else []
    if faces:
        primary = max(faces, key=lambda f: f['area'])
        blink = primary.get('blink')
        ear = primary['ear']
        bx0, by0, bx1, by1 = primary['box']
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        bw, bh = (bx1 - bx0), (by1 - by0)
        # 主脸框 ×1.5 crop（清晰度，含少许头发/边缘）
        x0 = int(max(0.0, cx - bw * 0.75) * W); x1 = int(min(1.0, cx + bw * 0.75) * W)
        y0 = int(max(0.0, cy - bh * 0.75) * H); y1 = int(min(1.0, cy + bh * 0.75) * H)
        face_crop = gray[y0:y1, x0:x1]
        lap_face = _lap_var(face_crop)
        # 紧脸框（曝光主体测光用，不含背景，避免逆光误判）
        face_tight = gray[int(by0 * H):int(by1 * H), int(bx0 * W):int(bx1 * W)]
        # 眼区 crop（左眼 33外/133内、右眼 362内/263外 横向；上 159/386、下 145/374 纵向），纵向 pad 1.5×
        lm = primary.get('lm') or []
        lap = lap_face
        if len(lm) >= 380:
            ex0, ey0, ex1, ey1 = _box_from_lm(lm, (33, 133, 362, 263),
                                              (159, 145, 386, 374), xpad=1.15, ypad=1.5)
            eye_crop = gray[int(ey0 * H):int(ey1 * H), int(ex0 * W):int(ex1 * W)]
            if eye_crop.size >= 1600:
                lap = _lap_var(eye_crop) * 0.6 + lap_face * 0.4
        # 表情/睁眼：双眼同时闭合才算闭眼；单眼闭=wink=好表情不扣分
        face = 0.4
        bl = primary.get('blink_l'); br = primary.get('blink_r')
        if bl is not None and br is not None:
            l_closed = bl >= _BLINK_CLOSED
            r_closed = br >= _BLINK_CLOSED
            l_open = bl < _BLINK_OPEN
            r_open = br < _BLINK_OPEN
            if l_closed and r_closed:
                # 双眼闭合 = 真闭眼
                face -= 0.6; tags.append('闭眼')
            elif l_closed != r_closed:
                # 单眼闭合 = wink，好表情，额外加分
                face += 0.15; tags.append('wink')
            elif l_open and r_open:
                # 双眼睁开
                face += 0.4
            else:
                # 介于中间（半睁），不奖不罚
                pass
        else:
            if ear >= _EAR_OPEN:
                face += 0.4
            else:
                face -= 0.6; tags.append('闭眼')
        if primary['area'] > 0.04:
            face += 0.15
        face = max(0.0, min(1.0, face))
    else:
        cy, cx = H // 2, W // 2
        center = gray[cy - H // 4:cy + H // 4, cx - W // 4:cx + W // 4]
        lap = _lap_var(center) * 0.7 + _lap_var(gray) * 0.3
        face = 0.5
        face_tight = None
    # 曝光：有人脸时主体测光（紧脸框，不含背景），逆光背景高光不再惩罚
    if face_tight is not None and face_tight.size >= 1600:
        fhist = cv2.calcHist([face_tight], [0], None, [256], [0, 256]).flatten()
        ftotal = face_tight.size
        fshadow = float(np.sum(fhist[:15])) / ftotal
        fhighlight = float(np.sum(fhist[240:])) / ftotal
        fmean = float(face_tight.mean()) / 255.0
    else:
        fshadow, fhighlight, fmean = shadow, highlight, mean
    clip = max(0.0, fshadow - 0.02) + max(0.0, fhighlight - 0.02)
    exp = max(0.0, 1.0 - clip * 3.0)
    if fmean < 0.12 or fshadow > 0.55:
        exp *= 0.4; tags.append('欠曝')
    elif fmean > 0.88 or fhighlight > 0.32:
        exp *= 0.4; tags.append('过曝')
    # 对比度：std 软曲线 40~80 → 0~1
    contrast = max(0.0, min(1.0, (std - 40) / 40.0))
    # 收集 blink 值供人物基线重标用
    bl_out, br_out = None, None
    if faces:
        primary = max(faces, key=lambda f: f['area'])
        bl_out = primary.get('blink_l'); br_out = primary.get('blink_r')
    # person_id 由 compute_persons 预赋值（评分时读取，供闭眼基线按人标定）
    return {'lap': lap, 'exp': max(0.0, min(1.0, exp)), 'contrast': contrast,
            'face': face, 'tags': tags,
            'person_id': photo.person_id if photo is not None else None,
            'blink_l': bl_out, 'blink_r': br_out}

def _cv(vals):
    """变异系数 std/mean（均值≤0 返回 1.0 视为有差异）。"""
    if len(vals) < 2:
        return 1.0
    m = sum(vals) / len(vals)
    if m <= 0:
        return 1.0
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return (var ** 0.5) / m

# 评分维度默认权重（对焦/表情/曝光/对比度）
_DIM_WEIGHTS = {'sharp': 0.45, 'face': 0.30, 'exp': 0.15, 'contrast': 0.10}
_ALL_DIMS = ('sharp', 'face', 'exp', 'contrast')

def _single_dim_value(r: dict, dim: str) -> float:
    """单张照片某维度的原始值（0..1）。sharp 无原比值（相对量），单张时给中值 0.5。"""
    if dim == 'sharp':
        return 0.5
    if dim == 'face':
        return r.get('face', 0.0)
    if dim == 'exp':
        return r.get('exp', 0.0)
    if dim == 'contrast':
        return r.get('contrast', 0.0)
    return 0.0

def _normalize_group(raws: list, dims=None) -> tuple:
    """组内各指标转百分位 0..1，合成 quality；返回 (per_photo, near_identical)。
    dims: 参与评分的维度子集（默认全部 4 项）。未勾选的维度权重归零并重新归一化，
    保证勾选维度权重和为 1；其子分置 0。near_identical 只考察勾选维度。"""
    if not dims:
        dims = list(_ALL_DIMS)
    laps = [r['lap'] for r in raws]; exps = [r['exp'] for r in raws]
    cons = [r['contrast'] for r in raws]; faces = [r['face'] for r in raws]
    has_junk = any('废片' in r['tags'] for r in raws)
    # 逐轴 cv：仅对勾选维度判近同质
    cv_checks = {'sharp': _cv(laps), 'face': _cv(faces),
                 'exp': _cv(exps), 'contrast': _cv(cons)}
    near_identical = (not has_junk) and len(raws) > 1 and \
        all(cv_checks[d] < _NEAR_IDENTICAL_CV for d in dims)
    lp = _percentile_rank(laps); ep = _percentile_rank(exps)
    cp = _percentile_rank(cons); fp = _percentile_rank(faces)
    pct = {'sharp': lp, 'face': fp, 'exp': ep, 'contrast': cp}
    # 动态权重：勾选维度权重归一化
    total_w = sum(_DIM_WEIGHTS[d] for d in dims) or 1.0
    out = []
    for i in range(len(raws)):
        if '废片' in raws[i]['tags']:
            out.append({'quality': 0.0, 'quality_pct': 0.0, 'sharp_pct': 0.0,
                        'exp_pct': 0.0, 'contrast_pct': 0.0, 'face_pct': 0.0})
        elif near_identical:
            out.append({'quality': 0.5, 'quality_pct': 0.5, 'sharp_pct': 0.5,
                        'exp_pct': 0.5, 'contrast_pct': 0.5, 'face_pct': 0.5})
        else:
            q = sum((_DIM_WEIGHTS[d] / total_w) * pct[d][i] for d in dims)
            out.append({'quality': q, 'quality_pct': 0.0,
                        'sharp_pct': pct['sharp'][i] if 'sharp' in dims else 0.0,
                        'face_pct': pct['face'][i] if 'face' in dims else 0.0,
                        'exp_pct': pct['exp'][i] if 'exp' in dims else 0.0,
                        'contrast_pct': pct['contrast'][i] if 'contrast' in dims else 0.0})
    qp = _percentile_rank([o['quality'] for o in out])
    for i in range(len(out)):
        out[i]['quality_pct'] = qp[i]
    return out, near_identical

def _select_stars(group, norm) -> int:
    """按 quality 选 top N（>10 张选 3 否则 2），跳过相邻帧与废片，保底选最佳 1 张。"""
    n = 3 if len(group) > 10 else 2
    ranked = sorted(range(len(group)), key=lambda i: norm[i]['quality'], reverse=True)
    qualities = sorted(n['quality'] for n in norm)
    median_q = qualities[len(qualities) // 2]
    selected = []
    for idx in ranked:
        if len(selected) >= n:
            break
        if any(abs(idx - s) <= 2 for s in selected):
            continue
        if '废片' in getattr(group[idx], 'tags', []):
            continue
        if norm[idx]['quality'] <= median_q:
            continue
        selected.append(idx)
    if not selected:  # 保底：最佳非废片 1 张
        for idx in ranked:
            if '废片' in getattr(group[idx], 'tags', []):
                continue
            selected.append(idx); break
    for idx in selected:
        group[idx].star = True
    return len(selected)

_BLINK_BASELINE_MARGIN = 0.30  # 相对基线（该人最小 blink）超过此值视为闭眼

def _apply_person_blink_baseline(photos, raws):
    """按 person_id 分组，取该人本段 min(blink_l, blink_r) 作为"完全睁开"基线。
    若 blink 值 >= 基线 + margin → 重标闭眼/wink（覆盖全局阈值判定）。
    需该人本段出现 >=2 次才有基线；否则保持全局判定不变。"""
    # 收集每人的 blink 最小值
    person_baselines = {}
    for i, r in enumerate(raws):
        pid = r.get('person_id')
        if pid is None:
            continue
        bl = r.get('blink_l'); br = r.get('blink_r')
        if bl is None or br is None:
            continue
        mn = min(bl, br)
        if pid not in person_baselines:
            person_baselines[pid] = {'min': mn, 'count': 1}
        else:
            person_baselines[pid]['min'] = min(person_baselines[pid]['min'], mn)
            person_baselines[pid]['count'] += 1
    # 只对出现 >=2 次的人做基线重标
    for pid, info in person_baselines.items():
        if info['count'] < 2:
            continue
        baseline = info['min']
        closed_thr = baseline + _BLINK_BASELINE_MARGIN
        for i, (p, r) in enumerate(zip(photos, raws)):
            if r.get('person_id') != pid:
                continue
            bl = r.get('blink_l'); br = r.get('blink_r')
            if bl is None or br is None:
                continue
            l_closed = bl >= closed_thr
            r_closed = br >= closed_thr
            # 先移除全局判定留下的 闭眼/wink 标签，再用相对基线重判
            tags = [t for t in r['tags'] if t not in ('闭眼', 'wink')]
            # 撤销全局判定对 face 的影响，重算 face 分数
            face_base = r['face']
            # 简化：重设 face 的奖惩部分
            if l_closed and r_closed:
                tags.append('闭眼')
                r['face'] = max(0.0, r['face'] - 0.6)  # 闭眼扣分
            elif l_closed != r_closed:
                tags.append('wink')
                r['face'] = min(1.0, r['face'] + 0.15)
            r['tags'] = tags
            # 同步到 Photo.tags
            p.tags = list(tags)

# ── 真连拍：时间组内按相邻帧视觉相似度细分 ───────────────
def _frame_sig(root, photo):
    """96×96 灰度签名（取自缓存的 250px 缩略图）；未就绪返回 None。"""
    import numpy as np
    from PIL import Image as _PIL
    try:
        cache = _cache_dir(Path(str(root))) / f'thumb_{photo.id}.jpg'
        if not cache.exists():
            return None
        pil = _PIL.open(cache).convert('L').resize((96, 96), _PIL.BILINEAR)
        return np.array(pil, dtype=np.uint8)
    except Exception:
        return None

def _frame_distance(a, b) -> float:
    """相邻帧距离 0..1 = 像素均差×0.6 + 直方图距离×0.4。"""
    import numpy as np
    if a is None or b is None or a.shape != b.shape:
        return 1.0  # 不可比 → 视为边界
    px = float(np.mean(np.abs(a.astype(int) - b.astype(int)))) / 255.0
    ha = np.histogram(a, bins=16, range=(0, 255))[0].astype(float)
    hb = np.histogram(b, bins=16, range=(0, 255))[0].astype(float)
    sa, sb = ha.sum(), hb.sum()
    if sa > 0 and sb > 0:
        inter = np.minimum(ha / sa, hb / sb).sum()
    else:
        inter = 0.0
    return px * 0.6 + (1.0 - inter) * 0.4

def compute_subbursts(root, groups) -> None:
    """每个时间组内按相邻帧距离自适应阈值切真连拍，写回 p.burst。"""
    import numpy as np
    for g in groups:
        if len(g) <= 1:
            if g: g[0].burst = 0
            continue
        sigs = [_frame_sig(root, p) for p in g]
        dists = []
        for i in range(1, len(g)):
            d = _frame_distance(sigs[i - 1], sigs[i])
            # 签名缺失时按 num 间隔兜底：非连续编号视为边界
            if sigs[i - 1] is None or sigs[i] is None:
                d = max(d, 1.0 if g[i].num - g[i - 1].num > 1 else 0.0)
            dists.append(d)
        arr = np.array(dists)
        med = float(np.median(arr)) if len(arr) else 0.0
        mad = float(np.median(np.abs(arr - med))) if len(arr) else 0.0
        thr = max(_SUBBURST_FLOOR, med + 3 * mad)
        b = 0; g[0].burst = 0
        for i in range(1, len(g)):
            if dists[i - 1] > thr:
                b += 1
            g[i].burst = b

def _split_subbursts(photos, raws) -> list:
    """按 p.burst 切连续子段，返回 [(sub_photos, sub_raws), ...]，保持原顺序。"""
    out = []
    cur_p, cur_r = [], []
    last = None
    for p, r in zip(photos, raws):
        if last is None or p.burst == last:
            cur_p.append(p); cur_r.append(r)
        else:
            out.append((cur_p, cur_r)); cur_p, cur_r = [p], [r]
        last = p.burst
    if cur_p:
        out.append((cur_p, cur_r))
    return out

_ROLE_CONFIRM = 0.08   # 帧距离 ≤ 此值直接归类角色
_ROLE_SUGGEST = 0.15    # 帧距离 ≤ 此值归类但标 merge_pending

def compute_roles(root, groups) -> None:
    """角色自动归类：对未分配 role_id 的照片，与全局角色 exemplar 的帧签名比较，
    最小距离 < 阈值 → 自动分配 role_id。二次元友好（用视觉相似度而非 ArcFace）。"""
    import time as _t
    t0 = _t.time()
    roles_db = _load_roles()
    roles = roles_db.get('roles', {})
    if not roles:
        return
    # 预加载每个角色的 exemplar 签名
    role_sigs = {}  # rid -> [sig, ...]
    for rid, rdata in roles.items():
        sigs = []
        for ex_path in rdata.get('exemplars', [])[:10]:
            try:
                from PIL import Image as _PIL
                pil = _PIL.open(ex_path).convert('L').resize((96, 96), _PIL.BILINEAR)
                import numpy as np
                sigs.append(np.array(pil, dtype=np.uint8))
            except Exception:
                pass
        if sigs:
            role_sigs[int(rid)] = sigs
    if not role_sigs:
        return
    assigned = 0
    for g in groups:
        for p in g:
            if p.role_id is not None:
                continue  # 已有角色，跳过
            sig = _frame_sig(root, p)
            if sig is None:
                continue
            best_rid, best_dist = None, _ROLE_SUGGEST
            for rid, sigs in role_sigs.items():
                for s in sigs:
                    d = _frame_distance(sig, s)
                    if d < best_dist:
                        best_dist = d; best_rid = rid
            if best_rid is not None:
                p.role_id = best_rid
                # 两级：> _ROLE_CONFIRM 标 merge_pending
                p.merge_pending = best_dist > _ROLE_CONFIRM
                assigned += 1
    if assigned:
        save_state(scan_root, groups)
        print(f"  >> 角色自动归类 {assigned} 张 ({_t.time() - t0:.1f}s)")

def score_groups(root, groups, dims=None) -> None:
    """主入口：per-photo 原始指标 → 按真连拍分片组内百分位 → 写回 p.score/score_*/tags → 星级推荐。
    dims: 参与评分的维度子集（默认全开）。"""
    import time as _t
    t0 = _t.time()
    total_star = 0
    for g in groups:
        raws = []
        for p in g:
            try:
                raws.append(_score_photo(root, p))
            except Exception:
                raws.append({'lap': 0.0, 'exp': 0.0, 'contrast': 0.0, 'face': 0.0, 'tags': ['废片']})
        # 按真连拍分片，每段独立评分（百分位/近同质扁平/星标都落在真连拍粒度）
        for sub_p, sub_r in _split_subbursts(g, raws):
            if len(sub_p) <= 1:
                if sub_p:
                    r, p = sub_r[0], sub_p[0]
                    # 单张组：用 dims 动态权重（默认全开），未勾选维度子分 0
                    dims_now = dims or list(_ALL_DIMS)
                    total_w = sum(_DIM_WEIGHTS[d] for d in dims_now) or 1.0
                    q = 0.0 if '废片' in r['tags'] else \
                        sum((_DIM_WEIGHTS[d] / total_w) * _single_dim_value(r, d) for d in dims_now)
                    p.score = round(q * 100)
                    p.score_sharp = 50 if 'sharp' in dims_now else 0
                    p.score_exp = round(r['exp'] * 100) if 'exp' in dims_now else 0
                    p.score_contrast = round(r['contrast'] * 100) if 'contrast' in dims_now else 0
                    p.score_face = round(r['face'] * 100) if 'face' in dims_now else 0
                    p.tags = list(r['tags'])
                    p.star = True
                    total_star += 1
                continue
            # 人物基线重标闭眼（先于归一化，face 值会被更新，归一化用新值）
            _apply_person_blink_baseline(sub_p, sub_r)
            norm, near_identical = _normalize_group(sub_r, dims)
            for i, p in enumerate(sub_p):
                r, n = sub_r[i], norm[i]
                if '废片' in r['tags']:
                    p.score = 0; p.score_sharp = 0; p.score_exp = 0
                    p.score_contrast = 0; p.score_face = 0
                else:
                    # 非废片保底 5 分（区分"本组最差但可用"与"废片 0"）；未勾选维度子分保持 0
                    p.score = max(5, round(n['quality_pct'] * 100))
                    p.score_sharp = max(5, round(n['sharp_pct'] * 100)) if 'sharp' in (dims or _ALL_DIMS) else 0
                    p.score_exp = max(5, round(n['exp_pct'] * 100)) if 'exp' in (dims or _ALL_DIMS) else 0
                    p.score_contrast = max(5, round(n['contrast_pct'] * 100)) if 'contrast' in (dims or _ALL_DIMS) else 0
                    p.score_face = max(5, round(n['face_pct'] * 100)) if 'face' in (dims or _ALL_DIMS) else 0
                p.tags = list(r['tags'])
            if near_identical:
                for p in sub_p:
                    p.tags.append('无明显差异')
                # 近同质真连拍不制造假区分，跳过星标，让用户自决
            else:
                total_star += _select_stars(sub_p, norm)
    print(f"  >> 评分完成 ({_t.time() - t0:.1f}s, 推荐 {total_star} 张)")

def scan(root: str, max_gap: int = 0, time_gap: int = 0, start_workers: bool = False) -> list[list[Photo]]:
    global scoring_busy
    Photo._next_id = 0
    root = Path(root).resolve()
    if not root.is_dir():
        print(f"!! 目录不存在: {root}")
        return []

    files = list(root.rglob("*"))
    previews = [f for f in files if f.suffix in PREVIEW_EXTS
                and f.parent.name not in SKIP_DIRS and is_real_photo(f)]
    if not previews:
        print("!! 未找到预览文件（JPG/HIF/HEIC）")
        return []

    by_stem: dict[str, Path] = {}
    for f in previews:
        stem = f.stem
        if stem not in by_stem:
            by_stem[stem] = f
        else:
            cur_ext = by_stem[stem].suffix.upper()
            new_ext = f.suffix.upper()
            if new_ext == '.JPG' and cur_ext != '.JPG':
                by_stem[stem] = f
    previews = list(by_stem.values())

    photos = []
    for f in previews:
        p = Photo(f); p.find_raw(); photos.append(p)

    photos.sort(key=lambda p: p.num)
    if not photos:
        return []

    for p in photos:
        p._dt = _best_dt(p)

    # 还原持久化状态（keep/star/score）
    _restore_state(photos, load_state(str(root)))
    # 检测历史导出目录，把仍在原相册中的已导出 ARW 对应照片默认选中（提醒已导出过）
    _mark_previously_exported(root, photos)

    groups = [[photos[0]]]
    for i in range(1, len(photos)):
        prev, cur = photos[i-1], photos[i]
        gap = cur.num - prev.num
        td = 999
        if prev._dt and cur._dt:
            td = (cur._dt - prev._dt).total_seconds()
        if gap > 3 or td > 30:
            groups.append([])
        groups[-1].append(cur)

    total_raw = sum(1 for p in photos if p.raw)
    print(f"> {len(photos)} 张, {total_raw} 张 ARW, {len(groups)} 个连拍组")

    # 后台评分（不阻塞 HTTP 响应）：缓存生成 → 清晰度/曝光评分 → 人脸评分
    if start_workers:
        scoring_busy = True
        root_s = str(root)
        def _bg_score():
            global scoring_busy
            try:
                pregenerate_cache(root_s, photos)
                compute_subbursts(root_s, groups)
                score_groups(root_s, groups)
                compute_roles(root_s, groups)
            except Exception as e:
                print(f"  >> 后台评分异常: {e}")
            finally:
                scoring_busy = False
        threading.Thread(target=_bg_score, daemon=True).start()

    for i, g in enumerate(groups):
        if len(g) > 1:
            print(f"  组{i+1}: {len(g)} 张 ({g[0].num}-{g[-1].num})")
    return groups

# ═══════════════════════════════════════════════════════════
#  图片服务 + 缓存
# ═══════════════════════════════════════════════════════════

def make_thumb(path: Path, size: int = THUMB_W) -> bytes | None:
    try:
        img = Image.open(path)
        # 用 draft 快速解码到目标尺寸附近，避免解码全分辨率
        try:
            img.draft('RGB', (size, size))
        except:
            pass
        img = ImageOps.exif_transpose(img) or img
        w, h = img.size
        if w > size:
            img = img.resize((size, int(h * size / w)), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=85)
        return buf.getvalue()
    except Exception as e:
        print(f"  !! 图片处理失败: {path.name} -- {e}")
        return None

def serve_image(path: Path) -> bytes | None:
    ext = path.suffix.upper()
    if ext in ('.JPG', '.JPEG'):
        try:
            return path.read_bytes()
        except Exception as e:
            print(f"  !! 读取失败: {path.name} -- {e}")
            return None
    elif ext in ('.HIF', '.HEIC'):
        return make_thumb(path, size=2400)
    return None

def _cache_dir(root: Path) -> Path:
    return root / '_tg_cache'

def _cache_path(root: Path, photo: Photo, size: int) -> Path:
    name = f'thumb_{photo.id}.jpg' if size <= 500 else f'preview_{photo.id}.jpg'
    return _cache_dir(root) / name

def pregenerate_cache(root: str, photos: list):
    import time as _t
    root_p = Path(root).resolve()
    cache = _cache_dir(root_p)
    cache.mkdir(exist_ok=True)
    t0 = _t.time()
    count = 0
    for p in photos:
        tp = _cache_path(root_p, p, 250)
        if not tp.exists():
            d = make_thumb(p.preview, 250)
            if d: tp.write_bytes(d); count += 1
        pp = _cache_path(root_p, p, 1600)
        if not pp.exists():
            d = make_thumb(p.preview, 1600)
            if d: pp.write_bytes(d); count += 1
        if (p.id + 1) % 100 == 0:
            print(f"  cache {p.id+1}/{len(photos)}...", end='', flush=True)
    el = _t.time() - t0
    if count: print(f"  生成 {count} 个缓存 ({el:.0f}s)")
    else: print(f"  缓存已就绪 ({el:.0f}s)")



# ═══════════════════════════════════════════════════════════
#  路径安全 / 持久化 / 删除（可测纯函数）
# ═══════════════════════════════════════════════════════════

def _safe_path(root: str, p: str) -> Path | None:
    """校验 p 必须位于 root 子树内，防路径穿越。返回 resolve 后的 Path 或 None。"""
    if not p or not root:
        return None
    try:
        root_p = Path(root).resolve()
        cand = Path(p).resolve()
    except (OSError, ValueError):
        return None
    try:
        cand.relative_to(root_p)
        return cand
    except ValueError:
        return None

def _state_path(root) -> Path:
    return _cache_dir(Path(str(root)).resolve()) / 'state.json'

def save_state(root, groups) -> None:
    """把 keep/star/score 落盘到 _tg_cache/state.json（按路径键，原子写）。"""
    try:
        sp = _state_path(root)
        sp.parent.mkdir(parents=True, exist_ok=True)
        photos = {}
        for g in groups:
            for p in g:
                photos[str(p.preview)] = {
                    'keep': p.keep, 'star': p.star,
                    'score': p.score, 's_sharp': p.score_sharp,
                    's_exp': p.score_exp, 's_contrast': p.score_contrast,
                    's_face': p.score_face,
                    'person_id': p.person_id,
                    'role_id': p.role_id,
                    'merge_pending': p.merge_pending,
                    'is_group': p.is_group,
                    'group_pids': list(p.group_pids),
                }
        data = {'version': 1, 'photos': photos}
        tmp = sp.with_suffix('.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, sp)
    except Exception as e:
        print(f"  !! 保存状态失败: {e}")

def load_state(root) -> dict:
    """读取持久化状态 {path: {...}}；文件缺失或损坏返回 {}。"""
    try:
        sp = _state_path(root)
        if not sp.exists():
            return {}
        with open(sp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('photos', {})
    except Exception:
        return {}

def _restore_state(photos, state: dict) -> None:
    """用 load_state 的结果还原 Photo 的 keep/star/score。"""
    for p in photos:
        rec = state.get(str(p.preview))
        if not rec:
            continue
        p.keep = bool(rec.get('keep', False))
        p.star = bool(rec.get('star', False))
        try:
            p.score = float(rec.get('score', 0.0))
            p.score_sharp = float(rec.get('s_sharp', 0.0))
            p.score_exp = float(rec.get('s_exp', 0.0))
            p.score_contrast = float(rec.get('s_contrast', 0.0))
            p.score_face = float(rec.get('s_face', 0.0))
            pid = rec.get('person_id')
            p.person_id = int(pid) if pid is not None else None
            rid = rec.get('role_id')
            p.role_id = int(rid) if rid is not None else None
            p.merge_pending = bool(rec.get('merge_pending', False))
            p.is_group = bool(rec.get('is_group', False))
            gp = rec.get('group_pids', [])
            p.group_pids = [int(x) for x in gp] if isinstance(gp, list) else []
        except (TypeError, ValueError):
            pass

def _exported_arw_names(root: Path) -> set:
    """收集相册内所有"挑选_*"导出目录中已导出的 ARW 文件名（去重）。"""
    if not root.is_dir():
        return set()
    exported = set()
    try:
        for d in root.iterdir():
            if d.is_dir() and d.name.startswith('挑选_'):
                try:
                    for f in d.iterdir():
                        if f.suffix.upper() == '.ARW':
                            exported.add(f.name)
                except OSError:
                    continue
    except OSError:
        pass
    return exported


def _mark_previously_exported(root: Path, photos: list) -> int:
    """检测相册内的"挑选_YYYYMMDD_HHMMSS"导出目录，找出其中导出的 ARW 文件名。
    若这些 ARW 仍存在于原相册（photo.raw 存在），则对应照片标记 keep=True（提醒已导出过）。
    返回标记的照片数。只读不改，不覆盖用户手动取消的选中。"""
    exported = _exported_arw_names(root)
    if not exported:
        return 0
    # 标记：照片有对应 raw 且 raw 文件名在已导出集合中 → keep=True
    count = 0
    for p in photos:
        if p.raw and p.raw.name in exported:
            p.keep = True
            count += 1
    if count:
        print(f"  > 检测到 {count} 张已导出过的照片（{len(exported)} 个 ARW 曾在挑选目录中），已默认选中")
    return count

def compute_delete_list(groups, mode: str, types) -> tuple[list[Path], list[str]]:
    """纯计算：给定模式与类型，返回待删文件列表与保留文件名列表。"""
    type_set = set(t.upper() for t in types)
    del_list: list[Path] = []
    keep_names: list[str] = []
    for g in groups:
        for p in g:
            should_del = (p.keep and mode == 'selected') or (not p.keep and mode == 'unselected')
            if not should_del:
                if p.keep: keep_names.append(p.preview.name)
                continue
            if p.fmt in type_set: del_list.append(p.preview)
            if p.raw and 'ARW' in type_set: del_list.append(p.raw)
            if p.keep: keep_names.append(p.preview.name)
    return del_list, keep_names

def execute_delete(root, del_list) -> tuple[int, int, Path, set]:
    """将待删文件移动到 _tg_trash/<时间戳>/ 下保持相对结构（不直接 unlink）。返回 (ok, err, trash, touched_dirs)。"""
    import time as _t, shutil as _sh
    root_p = Path(str(root)).resolve()
    trash = root_p / '_tg_trash' / _t.strftime('%Y%m%d_%H%M%S')
    trash.mkdir(parents=True, exist_ok=True)
    ok, err = 0, 0
    touched = set()
    for f in del_list:
        try:
            if not f.exists():
                continue
            try:
                rel = f.resolve().relative_to(root_p)
                dst = trash / rel
            except ValueError:
                dst = trash / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst = trash / f"{f.stem}_{_t.strftime('%H%M%S')}{f.suffix}"
            _sh.move(str(f), str(dst))
            ok += 1
            touched.add(f.parent)
        except Exception as e:
            print(f"  !! 移动到回收站失败: {f} -- {e}"); err += 1
    return ok, err, trash, touched

def _cleanup_empty_dirs(root, dirs) -> None:
    """删除空目录，但严格限制在 root 子树内、且不得删除 root 本身。"""
    root_p = Path(str(root)).resolve()
    for d in sorted(dirs, reverse=True):
        try:
            d = d.resolve()
            if d == root_p:
                continue
            try:
                d.relative_to(root_p)
            except ValueError:
                continue
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass

def _read_cache_bytes(root, photo, size) -> bytes | None:
    cf = _cache_path(Path(str(root)), photo, size) if root else None
    if cf and cf.exists():
        try:
            return cf.read_bytes()
        except Exception:
            return None
    return None

# ═══════════════════════════════════════════════════════════
#  HTTP 服务器
# ═══════════════════════════════════════════════════════════

groups: list[list[Photo]] = []
photo_list: list[Photo] = []
photo_dict: dict[int, Photo] = {}
current_dir: str = ""
scan_root: str = ""

# 并发：groups/photo_list/photo_dict 的读写守卫
_state_lock = threading.RLock()
# 并发：MediaPipe landmarker 单例初始化与 detect 调用守卫（IMAGE 模式非线程安全）
_mp_lock = threading.Lock()
# 后台评分进行中标志
scoring_busy: bool = False

# 缓存的 MediaPipe 模型
_face_landmarker = None
_pose_landmarker = None

def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        with _mp_lock:
            if _face_landmarker is None:
                import mediapipe as mp
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core import base_options as bo
                _model = Path(__file__).parent / 'face_landmarker.task'
                if _model.exists():
                    _face_landmarker = vision.FaceLandmarker.create_from_options(
                        vision.FaceLandmarkerOptions(
                            base_options=bo.BaseOptions(model_asset_path=str(_model)),
                            running_mode=vision.RunningMode.IMAGE, num_faces=5,
                            output_face_blendshapes=True))
    return _face_landmarker

def _get_pose_landmarker():
    global _pose_landmarker
    if _pose_landmarker is None:
        with _mp_lock:
            if _pose_landmarker is None:
                import mediapipe as mp
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core import base_options as bo
                _model = Path(__file__).parent / 'pose_landmarker_lite.task'
                if _model.exists():
                    _pose_landmarker = vision.PoseLandmarker.create_from_options(
                        vision.PoseLandmarkerOptions(
                            base_options=bo.BaseOptions(model_asset_path=str(_model)),
                            running_mode=vision.RunningMode.IMAGE))
    return _pose_landmarker

# ── 人物识别：InsightFace buffalo_s 嵌入 + 跨目录全局持久化 ───────────────
_face_app = None  # insightFace FaceAnalysis 单例

def _get_face_app():
    """单例加载 InsightFace buffalo_s（检测+landmark+embedding+genderage）。缺失返回 None。"""
    global _face_app
    if _face_app is None:
        with _mp_lock:
            if _face_app is None:
                try:
                    from insightface.app import FaceAnalysis
                    _face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
                    _face_app.prepare(ctx_id=-1, det_size=(640, 640))
                except Exception:
                    _face_app = None
    return _face_app
_PERSON_PALETTE = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                   '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4']

def _person_color(pid: int | None) -> str:
    """人物 ID 映射到稳定颜色（用于 UI 色块）。"""
    if pid is None:
        return '#555'
    return _PERSON_PALETTE[pid % len(_PERSON_PALETTE)]


_onnx_lock = threading.Lock()
_persons_lock = threading.Lock()
_face_recognizer = None  # onnxruntime.InferenceSession

_RECOG_THRESHOLD = 0.45   # 匹配下限：≥此值才归类到已有人物（保守，避免相似不同人误合并）
_RECOG_CONFIRM = 0.55     # 高置信直认阈值：≥此值不标 merge_pending
_ARCFACE_SIZE = 112       # ArcFace 标准输入尺寸
# ArcFace 对齐参考点（112×112 坐标系）
_ARCFACE_REF = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                         [56.0252, 71.7366], [41.5493, 92.3655],
                         [70.7299, 92.2041]], dtype=np.float32)

def _get_face_recognizer():
    """单例加载 ArcFace ONNX 模型。模型缺失返回 None（识别降级）。"""
    global _face_recognizer
    if _face_recognizer is None:
        with _onnx_lock:
            if _face_recognizer is None:
                import os
                _model = Path(__file__).parent / 'face_recognizer.onnx'
                if _model.exists() and _model.stat().st_size > 100_000:
                    import onnxruntime as ort
                    _face_recognizer = ort.InferenceSession(str(_model))
    return _face_recognizer

def _cosine_sim(a, b) -> float:
    """余弦相似度，返回 -1..1。"""
    import numpy as np
    na = np.asarray(a, dtype=np.float32); nb = np.asarray(b, dtype=np.float32)
    na = na.ravel(); nb = nb.ravel()
    denom = (np.linalg.norm(na) * np.linalg.norm(nb))
    if denom <= 0:
        return 0.0
    return float(np.dot(na, nb) / denom)

def _align_face(image_rgb, lm, size=_ARCFACE_SIZE):
    """用 MediaPipe 5 点(左眼外33/右眼外263/鼻1/嘴61/下巴199)做仿射对齐到 112×112。"""
    import cv2, numpy as np
    if not lm or len(lm) < 200:
        # landmark 不足，退回中心 crop
        h, w = image_rgb.shape[:2]
        s = min(h, w, size)
        y0 = max(0, (h - s) // 2); x0 = max(0, (w - s) // 2)
        crop = image_rgb[y0:y0+s, x0:x0+s]
        return cv2.resize(crop, (size, size))
    H, W = image_rgb.shape[:2]
    src = np.array([
        [lm[33][0] * W, lm[33][1] * H],   # 左眼外
        [lm[263][0] * W, lm[263][1] * H], # 右眼外
        [lm[1][0] * W, lm[1][1] * H],     # 鼻尖
        [lm[61][0] * W, lm[61][1] * H],   # 左嘴角
        [lm[199][0] * W, lm[199][1] * H], # 下巴
    ], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, _ARCFACE_REF)
    if M is None:
        h, w = image_rgb.shape[:2]
        s = min(h, w, size)
        y0 = max(0, (h - s) // 2); x0 = max(0, (w - s) // 2)
        crop = image_rgb[y0:y0+s, x0:x0+s]
        return cv2.resize(crop, (size, size))
    aligned = cv2.warpAffine(image_rgb, M, (size, size), borderValue=0)
    return aligned

def _embed_face(session, aligned_rgb) -> 'np.ndarray | None':
    """ArcFace ONNX 推理，返回 512 维 L2 归一化向量。"""
    import numpy as np
    if session is None or aligned_rgb is None:
        return None
    try:
        # ArcFace 标准：BGR, (1,3,112,112), (img-127.5)/127.5
        import cv2
        bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
        blob = (bgr.astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1,3,112,112)
        inp = session.get_inputs()[0].name
        out = session.run(None, {inp: blob})[0]
        emb = out[0].astype(np.float32)
        n = np.linalg.norm(emb)
        if n > 0:
            emb = emb / n
        return emb
    except Exception:
        return None

# ── 全局数据库：D:/0_camera/_tg_global/（跨目录，跟照片放一起方便备份）──
_global_root_cache: Path | None = None

def _global_root() -> Path:
    """全局库根目录。优先第一个配置的相册根目录/_tg_global；
    否则从 scan_root 上溯启发式；最后回退 ~/.tgphoto/。"""
    global _global_root_cache
    if _global_root_cache is not None:
        return _global_root_cache
    # 0) 用户配置的相册根目录（roots.json），优先使用第一个
    roots = _load_roots()
    if roots:
        rp = Path(roots[0])
        if rp.is_dir():
            _global_root_cache = rp / '_tg_global'
            return _global_root_cache
    # 1) 如果 scan_root 已设，检测其父目录是否含多个相册子目录
    if scan_root:
        root = Path(scan_root)
        # 上溯找含 _tg_global 或多个含 _state.json 子目录的祖先
        for ancestor in [root, root.parent, root.parent.parent]:
            if (ancestor / '_tg_global').is_dir():
                _global_root_cache = ancestor / '_tg_global'
                return _global_root_cache
            # 检测是否是相册根（多个子目录含照片）
            if ancestor.is_dir():
                sub_albums = [d for d in ancestor.iterdir()
                              if d.is_dir() and not d.name.startswith('_')
                              and ((d / '_tg_cache' / 'state.json').exists() or (d / '_tg_cache').is_dir())]
                if len(sub_albums) >= 2:
                    _global_root_cache = ancestor / '_tg_global'
                    return _global_root_cache
    # 2) 回退到 ~/.tgphoto/
    _global_root_cache = Path.home() / '.tgphoto'
    return _global_root_cache

def _persons_path() -> Path:
    return _global_root() / 'persons.json'

def _load_persons() -> dict:
    """加载全局人物库 v3。自动从旧版本迁移（v1/v2 → v3）。"""
    p = _persons_path()
    data = None
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        data = None
    # 迁移：旧位置有但新位置没有
    if data is None:
        old = Path.home() / '.tgphoto' / 'persons.json'
        if old.exists() and not p.exists():
            try:
                data = json.loads(old.read_text(encoding='utf-8'))
            except Exception:
                data = None
    if data is None or not isinstance(data, dict) or 'persons' not in data:
        return {'version': 3, 'persons': {}, 'next_id': 1}
    if int(data.get('version', 1)) < 3:
        data = _migrate_persons_v3(data)
    return data

def _migrate_persons_v3(data: dict) -> dict:
    """v1/v2 → v3：把每人的滑动窗口 embeddings 聚成一个归一化质心 prototype；
    name 非空置 confirmed=True；无 embedding 的删除。"""
    import numpy as np
    out = {'version': 3, 'persons': {}, 'next_id': int(data.get('next_id', 1))}
    for pid, pdata in (data.get('persons') or {}).items():
        embs = pdata.get('embeddings') or []
        if not embs:
            continue  # 无 embedding 的人物删除
        arr = np.asarray(embs, dtype=np.float32)
        proto = arr.mean(axis=0)
        n = np.linalg.norm(proto)
        if n > 0:
            proto = (proto / n).astype(np.float32)
        name = pdata.get('name', '') or ''
        out['persons'][pid] = {
            'name': name,
            'confirmed': bool(name),          # 命名过的视为已确认
            'prototype': proto.tolist(),
            'count': len(embs),
            'created_at': pdata.get('created_at', 0),
        }
        # next_id 至少大于所有现有 id
        try:
            out['next_id'] = max(out['next_id'], int(pid) + 1)
        except (TypeError, ValueError):
            pass
    return out

def _save_persons(persons: dict) -> None:
    """原子写入全局人物库 v3。"""
    p = _persons_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(persons, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, p)
    except Exception:
        pass

_PERSONS_VERSION = 3

def _person_proto(pdata: dict) -> list | None:
    """取人物的质心向量。无则返回 None。"""
    proto = pdata.get('prototype')
    return proto if proto else None

def _match_person(emb, threshold=_RECOG_THRESHOLD) -> tuple:
    """匹配全局人物库（质心匹配）。返回 (person_id, similarity, n=1/0)。
    只匹配 confirmed 人物（未命名的人物不参与自动跨库匹配）。
    取与各 confirmed 人物质心的最高余弦；≥ threshold 返回。"""
    if emb is None:
        return None, 0.0, 0
    import numpy as np
    with _persons_lock:
        persons = _load_persons()
    best_id, best_sim = None, 0.0
    for pid, pdata in persons['persons'].items():
        if not pdata.get('confirmed', False):
            continue  # 未确认人物不参与自动匹配
        proto = _person_proto(pdata)
        if proto is None:
            continue
        sim = _cosine_sim(emb, np.asarray(proto, dtype=np.float32))
        if sim > best_sim:
            best_sim, best_id = sim, int(pid)
    if best_id is not None and best_sim >= threshold:
        return best_id, best_sim, 1
    return None, 0.0, 0

def _register_person(emb, exemplar_path: str = '', name: str = '', confirmed: bool = False) -> int:
    """新建人物 v3，存 prototype（单个向量即质心）。返回 person_id。"""
    if emb is None:
        return -1
    import numpy as np
    proto = np.asarray(emb, dtype=np.float32)
    n = np.linalg.norm(proto)
    if n > 0:
        proto = proto / n
    with _persons_lock:
        persons = _load_persons()
        pid = persons['next_id']
        persons['persons'][str(pid)] = {
            'name': name, 'confirmed': confirmed,
            'prototype': proto.tolist(), 'count': 1, 'created_at': 0}
        persons['next_id'] = pid + 1
        _save_persons(persons)
        return pid

_UPDATE_MIN_SIM = 0.55  # 新 embedding 与该人物质心相似度低于此值视为 outlier，拒绝更新

def _update_person(pid: int, emb, exemplar_path: str = '') -> bool:
    """用新 embedding 更新人物质心（在线均值，count 加权）。outlier 拒绝。返回是否更新。"""
    if emb is None or pid is None or pid < 0:
        return False
    import numpy as np
    e = np.asarray(emb, dtype=np.float32)
    n = np.linalg.norm(e)
    if n > 0:
        e = e / n
    with _persons_lock:
        persons = _load_persons()
        key = str(pid)
        if key not in persons['persons']:
            return False
        pdata = persons['persons'][key]
        proto = _person_proto(pdata)
        # 内部一致性：与现有质心足够相似才更新，防污染
        if proto is not None:
            if _cosine_sim(e, np.asarray(proto, dtype=np.float32)) < _UPDATE_MIN_SIM:
                return False
        cnt = int(pdata.get('count', 1))
        if proto is None:
            new_proto = e
        else:
            new_proto = (np.asarray(proto, dtype=np.float32) * cnt + e) / (cnt + 1)
        nn = np.linalg.norm(new_proto)
        if nn > 0:
            new_proto = new_proto / nn
        pdata['prototype'] = new_proto.astype(np.float32).tolist()
        pdata['count'] = cnt + 1
        _save_persons(persons)
        return True

_CLEANUP_MIN_SIM = 0.60  # 兼容保留（v3 用质心，不再拆分）

def cleanup_persons() -> dict:
    """v3 清理：删除 count=0 或 prototype 为空的人物。返回清理报告。"""
    with _persons_lock:
        persons = _load_persons()
        removed = []
        for pid, pdata in list(persons['persons'].items()):
            if pdata.get('count', 0) <= 0 or not _person_proto(pdata):
                del persons['persons'][pid]
                removed.append(int(pid))
        if removed:
            _save_persons(persons)
    return {'cleaned': len(removed), 'details': [{'person': p} for p in removed]}

# ── 相册索引（_tg_global/albums.json）── 跨目录打通 ──────────────
def _albums_path() -> Path:
    return _global_root() / 'albums.json'

def _load_albums() -> dict:
    p = _albums_path()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
            if isinstance(data, dict) and 'albums' in data:
                return data
    except Exception:
        pass
    return {'version': 1, 'albums': []}

def _save_albums(albums: dict) -> None:
    try:
        p = _albums_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(albums, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, p)
    except Exception:
        pass

# ── 相册根目录配置（~/.tgphoto/roots.json，固定位置，独立于 _global_root）──
def _roots_path() -> Path:
    return Path.home() / '.tgphoto' / 'roots.json'

def _load_roots() -> list:
    try:
        p = _roots_path()
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
            if isinstance(data, list):
                return [r for r in data if isinstance(r, str)]
            if isinstance(data, dict) and isinstance(data.get('roots'), list):
                return [r for r in data['roots'] if isinstance(r, str)]
    except Exception:
        pass
    return []

def _save_roots(roots: list) -> None:
    try:
        p = _roots_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(roots, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, p)
    except Exception:
        pass

# ── 照片文件夹检测（相册发现）──────────────────────────
def _count_photos(dirpath: Path) -> int:
    """统计目录内直接含有的预览照片数量（不递归）。"""
    try:
        return sum(1 for f in dirpath.iterdir()
                   if f.is_file() and f.suffix in PREVIEW_EXTS and is_real_photo(f))
    except OSError:
        return 0

def _album_count_and_cover(dirpath: Path) -> tuple:
    """递归统计相册照片数并取首张封面（可跨任意深度）。

    跳过 SKIP_DIRS 与隐藏/下划线目录；按文件名过滤 mac 资源叉等小文件，
    不做 stat（快，实测与 stat 结果一致）。返回 (count, cover_path)。"""
    count = 0
    cover = ''
    try:
        for dp, dirnames, filenames in os.walk(dirpath):
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIRS and not x.startswith(('_', '.'))]
            for fn in filenames:
                if fn.startswith(('._', '.__')):
                    continue
                if fn.lower().endswith(('.jpg', '.jpeg', '.hif', '.heic')):
                    if not cover:
                        cover = str(Path(dp) / fn)
                    count += 1
    except OSError:
        pass
    return count, cover

def _first_photo(dirpath: Path) -> Path | None:
    """返回目录内第一张预览照片路径（按文件名排序）。"""
    try:
        for f in sorted(dirpath.iterdir()):
            if f.is_file() and f.suffix in PREVIEW_EXTS and is_real_photo(f):
                return f
    except OSError:
        return None

def _is_photo_folder(dirpath: Path) -> bool:
    """判断子目录是否符合相册要求：已有 _tg_cache 标记，或含预览照片（递归）。"""
    if not dirpath.is_dir():
        return False
    if (dirpath / '_tg_cache' / 'state.json').exists():
        return True
    return _album_count_and_cover(dirpath)[0] > 0

def _all_albums() -> tuple:
    """跨所有已配置相册根目录聚合索引。返回 (roots, albums)。"""
    roots = _load_roots()
    albums = []
    for r in roots:
        rp = Path(r)
        if not rp.is_dir():
            continue
        albums.extend(_build_album_index(rp))
    return roots, albums

ALBUM_CACHE_TTL = 300  # 相册索引缓存秒数（避免大相册根每次首页都全量重扫）

def _rebuild_albums(rlist) -> list:
    """对一组相册根目录重建索引（全量扫描）。"""
    albums = []
    for r in rlist:
        rp = Path(r)
        if rp.is_dir():
            albums.extend(_build_album_index(rp))
    return albums

def _save_album_cache(albums) -> None:
    _save_albums({'version': 2, 'scanned_at': datetime.now().timestamp(), 'albums': albums})

def _cached_albums(rlist) -> list | None:
    """返回新鲜缓存索引；无/过期返回 None（调用方重建）。"""
    cached = _load_albums()
    ts = cached.get('scanned_at') or 0
    if cached.get('albums') and (datetime.now().timestamp() - ts) < ALBUM_CACHE_TTL:
        return cached['albums']
    return None

def _is_album_root(path: Path) -> bool:
    """检测是否是相册根目录（含多个相册子目录）。"""
    if not path.is_dir():
        return False
    # 含 _tg_global 说明是根目录
    if (path / '_tg_global').is_dir():
        return True
    # 含 2+ 个带 _tg_cache/state.json 的子目录
    count = 0
    for d in path.iterdir():
        if not d.is_dir() or d.name.startswith('_') or d.name.startswith('.'):
            continue
        if (d / '_tg_cache' / 'state.json').exists() or (d / '_tg_cache').is_dir():
            count += 1
            if count >= 2:
                return True
    return False

def _build_album_index(root: Path) -> list:
    """遍历 root 下的子目录，构建相册索引。
    有 _tg_cache/state.json 的读元数据；未扫过的目录快速直扫（计数+封面）。
    跳过不含照片的目录。"""
    import time as _t
    albums = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith('_') or d.name.startswith('.') or d.name in SKIP_DIRS:
            continue
        state_path = d / '_tg_cache' / 'state.json'
        photo_count = 0
        cover_path = ''
        persons = set()
        roles = set()
        date_str = d.name.split(' ')[0] if ' ' in d.name else d.name[:10]
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding='utf-8'))
                photos = state.get('photos', {})
                photo_count = len(photos)
                cover_path = next(iter(photos.keys()), '') if photos else ''
                for v in photos.values():
                    if v.get('person_id') is not None:
                        persons.add(v['person_id'])
                    if v.get('role_id') is not None:
                        roles.add(v['role_id'])
            except Exception:
                pass
        if photo_count == 0:
            # 未扫过的目录：递归统计（支持 DCIM/子目录等多层结构）
            photo_count, cover_path = _album_count_and_cover(d)
            if cover_path:
                try:
                    ts = datetime.fromtimestamp(Path(cover_path).stat().st_mtime)
                    date_str = ts.strftime('%Y-%m-%d')
                except OSError:
                    pass
        if photo_count == 0:
            continue  # 非照片文件夹，跳过
        albums.append({
            'path': str(d), 'name': d.name, 'photo_count': photo_count,
            'date': date_str, 'persons': sorted(persons), 'roles': sorted(roles),
            'cover_path': cover_path, 'root': str(root),
        })
    return albums

# ── 全局角色库持久化（~tg_global/roles.json）── coser + 角色双维度 ──
def _roles_path() -> Path:
    return _global_root() / 'roles.json'

def _load_roles() -> dict:
    p = _roles_path()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding='utf-8'))
            if isinstance(data, dict) and 'roles' in data:
                return data
    except Exception:
        pass
    old = Path.home() / '.tgphoto' / 'roles.json'
    if old.exists() and not p.exists():
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(old.read_bytes())
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'version': 2, 'roles': {}, 'next_id': 1}

def _save_roles(roles: dict) -> None:
    try:
        path = _roles_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(roles, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, path)
    except Exception:
        pass

def _merge_persons(src_id: int, dst_id: int):
    """合并 src→dst（v3 质心模型）：dst 质心 = 两质心按 count 加权平均并归一化，删除 src。"""
    import numpy as np
    with _persons_lock:
        persons = _load_persons()
        sk, dk = str(src_id), str(dst_id)
        if sk not in persons['persons'] or dk not in persons['persons']:
            return False
        src = persons['persons'][sk]
        dst = persons['persons'][dk]
        sp = src.get('prototype'); dp = dst.get('prototype')
        sc, dc = int(src.get('count', 1)), int(dst.get('count', 1))
        if sp is not None and dp is not None:
            merged = (np.asarray(sp, dtype=np.float32) * sc +
                      np.asarray(dp, dtype=np.float32) * dc) / (sc + dc)
            n = np.linalg.norm(merged)
            if n > 0:
                dst['prototype'] = (merged / n).astype(np.float32).tolist()
        elif sp is not None:
            dst['prototype'] = sp
        dst['count'] = sc + dc
        # 命名优先保留已确认的名字
        if not dst.get('name') and src.get('name'):
            dst['name'] = src.get('name', '')
        dst['confirmed'] = bool(dst.get('name', ''))
        del persons['persons'][sk]
        _save_persons(persons)
    with _state_lock:
        for g in groups:
            for p in g:
                if p.person_id == src_id:
                    p.person_id = dst_id
        save_state(scan_root, groups)
    return True

def _rename_person(pid: int, name: str):
    """重命名人物，并标记 confirmed=True（命名即确认，参与跨库自动匹配）。"""
    with _persons_lock:
        persons = _load_persons()
        key = str(pid)
        if key in persons['persons']:
            persons['persons'][key]['name'] = name
            persons['persons'][key]['confirmed'] = True
            _save_persons(persons)

def _rename_role(rid: int, name: str):
    with _persons_lock:
        roles = _load_roles()
        key = str(rid)
        if key in roles['roles']:
            roles['roles'][key]['name'] = name
            _save_roles(roles)

_MERGE_SUGGEST_THRESHOLD = 0.50  # 人物对相似度 ≥ 此值才建议合并（保守，避免相似不同人）

def _compute_merge_suggestions() -> list:
    """只计算**当前目录出现**的人物对，返回合并建议（避免推荐不在本目录的无关人物）。"""
    import numpy as np
    with _state_lock:
        snap = [[p for p in g] for g in groups]
    # 当前目录出现的 person_id 集合 + 本目录张数
    dir_pids = {}
    for g in snap:
        for p in g:
            if p.person_id is not None:
                dir_pids[p.person_id] = dir_pids.get(p.person_id, 0) + 1
    pids = sorted(dir_pids.keys())
    if len(pids) < 2:
        return []
    with _persons_lock:
        persons = _load_persons()
    protos = {}
    for pid in pids:
        key = str(pid)
        if key in persons['persons']:
            pdata = persons['persons'][key]
            pr = _person_proto(pdata)
            if pr is not None:
                import numpy as np
                protos[pid] = np.asarray(pr, dtype=np.float32)
    suggestions = []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            a, b = pids[i], pids[j]
            if a not in protos or b not in protos:
                continue
            best_sim = _cosine_sim(protos[a], protos[b])
            if best_sim >= _MERGE_SUGGEST_THRESHOLD:
                # 本目录张数小的合并到大的
                ca, cb = dir_pids[a], dir_pids[b]
                src, dst = (a, b) if ca <= cb else (b, a)
                suggestions.append({
                    'src_id': int(src), 'dst_id': int(dst), 'similarity': round(best_sim, 3),
                    'src_count': dir_pids[src],
                    'dst_count': dir_pids[dst],
                    'src_name': persons['persons'].get(str(src), {}).get('name', ''),
                    'dst_name': persons['persons'].get(str(dst), {}).get('name', ''),
                })
    suggestions.sort(key=lambda x: -x['similarity'])
    return suggestions

# ── 人物识别 v3：文件夹内批量聚类 + prototype 质心匹配 ────────────
# 设计对齐行业方案：批量聚类（非在线逐张）、人物存归一化质心、已确认/未确认分离。
# 流程：扫描时收集全部 face embedding → sklearn HAC 一次性聚类 → 每簇一个"人物候选" →
#       小簇（<3 张脸，多为路人/模糊脸）不建人物 → 簇质心与全局 confirmed 人物 prototype 匹配 →
#       sim≥0.55 自动归并 / 0.45~0.55 待确认候选(merge_pending) / <0.45 新建未确认人物。

_CLUSTER_DIST = 0.50       # HAC cosine 距离阈值（sim≥0.50 视为同一人）
_MIN_CLUSTER = 3           # 簇最小脸数（小于则视为路人，不建人物）
_AUTO_MERGE_SIM = 0.55     # 簇质心与全局 confirmed 人物 ≥ 此值 → 自动归并
_PENDING_SIM = 0.45        # ≥ 此值但 < AUTO → 待确认候选

def _extract_all_faces(root, photos) -> list:
    """收集全部人脸 embedding（供聚类）。_face_data 为空时现场检测并回填（避免二次检测）。
    返回 [(photo, face, emb_norm), ...]。"""
    import numpy as np
    rows = []
    for p in photos:
        faces = getattr(p, '_face_data', None) or []
        if not faces:
            # 现场检测并回填，供后续评分复用
            try:
                image = _load_score_image(root, p)
                if image is not None:
                    small = _resize_long(image, _FACE_LONG)
                    faces = _detect_faces(small)
                    p._face_data = list(faces)
            except Exception:
                faces = []
        for face in faces:
            e = face.get('embedding')
            if e is None:
                continue
            arr = np.asarray(e, dtype=np.float32)
            n = np.linalg.norm(arr)
            if n > 0:
                arr = arr / n
            rows.append((p, face, arr))
    return rows

def _hac_cluster(X, dist_thr=_CLUSTER_DIST) -> list:
    """对 embedding 矩阵做层次聚类，返回每行的簇标签。sklearn 缺失时退回贪心最近邻聚类。"""
    import numpy as np
    try:
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=None, metric='cosine',
                                        linkage='average', distance_threshold=dist_thr)
        labels = model.fit_predict(X)
        return list(labels)
    except Exception:
        # 贪心：按行序，与已建簇任一成员 sim≥dist_thr 归入，否则新建簇
        labels = []
        clusters = []
        for i in range(len(X)):
            placed = None
            for ci, cl in enumerate(clusters):
                if any(_cosine_sim(X[i], X[j]) >= dist_thr for j in cl):
                    placed = ci; break
            if placed is None:
                clusters.append([i]); labels.append(len(clusters) - 1)
            else:
                clusters[placed].append(i); labels.append(placed)
        return labels

def _prototype_of(embs) -> 'np.ndarray | None':
    """一组 embedding 的归一化质心。"""
    import numpy as np
    if not embs:
        return None
    arr = np.asarray(embs, dtype=np.float32)
    proto = arr.mean(axis=0)
    n = np.linalg.norm(proto)
    if n > 0:
        return (proto / n).astype(np.float32)
    return None

def compute_persons(root, groups) -> dict:
    """人物识别主入口：文件夹内批量聚类 + 全局质心匹配。
    对每张照片设 person_id / merge_pending / is_group / group_pids，并把聚类结果持久化到全局库。
    返回 {'auto': 自动归并, 'pending': 待确认, 'new': 新建, 'group': 合照} 统计。"""
    import time as _t, numpy as np
    t0 = _t.time()
    photos = [p for g in groups for p in g]
    rows = _extract_all_faces(root, photos)
    stats = {'auto': 0, 'pending': 0, 'new': 0, 'group': 0}
    if not rows:
        return stats
    X = np.stack([r[2] for r in rows])
    labels = _hac_cluster(X)
    # 每簇 → 全局人物（或 None=路人小簇）
    cluster_person = {}  # label -> person_id or None
    cluster_proto = {}   # label -> prototype
    cluster_pending = {} # label -> bool (待确认候选)
    n_clusters = max(labels) + 1 if labels else 0
    for c in range(n_clusters):
        idxs = [i for i, l in enumerate(labels) if l == c]
        if len(idxs) < _MIN_CLUSTER:
            cluster_person[c] = None   # 小簇：路人，不建人物
            continue
        proto = _prototype_of([X[i] for i in idxs])
        if proto is None:
            cluster_person[c] = None; continue
        cluster_proto[c] = proto
        # 与全局 confirmed 人物质心匹配
        pid, sim, _ = _match_person(proto, threshold=_PENDING_SIM)
        if pid is not None and sim >= _AUTO_MERGE_SIM:
            _update_person(pid, proto)
            cluster_person[c] = pid; cluster_pending[c] = False
            stats['auto'] += 1
        elif pid is not None:
            # 中置信待确认候选：新建未确认人物（不污染已知人物）
            new_pid = _register_person(proto, confirmed=False)
            cluster_person[c] = new_pid; cluster_pending[c] = True
            stats['pending'] += 1
        else:
            new_pid = _register_person(proto, confirmed=False)
            cluster_person[c] = new_pid; cluster_pending[c] = True
            stats['new'] += 1
    # 逐照片写回 person_id（主脸=面积最大脸所属簇）；合照=一张照片内出现 ≥2 个不同已确认人物
    from collections import defaultdict
    face_owner = []  # 每行 face 的 person_id
    for i, (p, face, _) in enumerate(rows):
        lab = labels[i]
        face_owner.append(cluster_person.get(lab))
    # 按照片聚合（保留全局行索引，供 merge_pending 查簇）
    photo_faces = defaultdict(list)  # id(p) -> [(row_idx, face, pid)]
    for i, (p, face, _) in enumerate(rows):
        photo_faces[id(p)].append((i, face, face_owner[i]))
    for p in photos:
        pf = photo_faces.get(id(p), [])
        if not pf:
            p.person_id = None; p.merge_pending = False; p.is_group = False; p.group_pids = []
            continue
        # 主脸 = 面积最大
        primary_row, primary_face, primary_pid = max(pf, key=lambda t: t[1]['area'])
        # 合照：≥2 个不同已确认人物出现在同张照片
        known = {}
        for row, f, pid in pf:
            if pid is None:
                continue
            known.setdefault(pid, 0)
            known[pid] += 1
        distinct = list(known.keys())
        if len(distinct) >= 2:
            p.person_id = primary_pid if primary_pid is not None else distinct[0]
            p.is_group = True
            p.group_pids = sorted(distinct)
            stats['group'] += 1
        else:
            p.person_id = primary_pid
            p.is_group = False
            p.group_pids = []
        # merge_pending：主脸所属簇是否为"待确认候选"
        plabel = labels[primary_row]
        p.merge_pending = bool(primary_pid is not None and cluster_pending.get(plabel, False))
    if root:
        save_state(root, groups)
    print(f"  >> 人物聚类完成 ({_t.time()-t0:.1f}s: 自动归并{stats['auto']} 待确认{stats['pending']} 新建{stats['new']} 合照{stats['group']})")
    return stats

def get_photo_by_id(pid: int) -> Photo | None:
    return photo_dict.get(pid)

def rebuild_photo_list():
    global photo_list, photo_dict
    photo_list = []
    photo_dict = {}
    for g in groups:
        for p in g:
            photo_list.append(p)
            photo_dict[p.id] = p

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _check_host(self) -> bool:
        """仅允许本机回环 Host 头，防跨源/CSRF 打本地接口。"""
        host = self.headers.get('Host', '')
        port = self.server.server_address[1]
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}',
                   '127.0.0.1', 'localhost'}
        return host in allowed

    def do_GET(self):
        if not self._check_host():
            self.send_error(403); return
        try:
            p = urlparse(self.path)
            q = parse_qs(p.query)
            if p.path == '/':           return self.html_index()
            if p.path == '/api/data':   return self.api_data()
            if p.path == '/api/status': return self.api_status()
            if p.path == '/api/log':   return self.api_log(q)
            if p.path == '/api/reload': return self.api_reload()
            if p.path == '/api/persons': return self.api_persons(q)
            if p.path == '/api/roles':   return self.api_roles(q)
            if p.path == '/api/person_photos': return self.api_person_photos(q)
            if p.path == '/api/merge_suggestions': return self.api_merge_suggestions()
            if p.path == '/api/cleanup': return self.api_cleanup()
            photo_id = q.get('id', [None])[0]
            path_str = q.get('p', [''])[0]
            if photo_id is not None:
                try:
                    photo_id = int(photo_id)
                except ValueError:
                    photo_id = None
            if p.path == '/thumb':
                return self.serve_img(photo_id, path_str, 250)
            if p.path == '/preview':
                return self.serve_img(photo_id, path_str, 1600)
            if p.path == '/preview_quick':
                return self.serve_preview_quick(photo_id)
            if p.path == '/api/debug_photo':
                return self.api_debug_photo(photo_id)
            self.send_error(404)
        except Exception as e:
            traceback.print_exc()
            self.send_error(500)

    def do_POST(self):
        if not self._check_host():
            self.send_error(403); return
        try:
            p = urlparse(self.path)
            # /api/browse 不需要读取 body
            if p.path == '/api/browse':   return self.api_browse()
            body = self._read_json()
            if p.path == '/api/toggle':   return self.api_toggle(body)
            if p.path == '/api/delete':   return self.api_delete(body)
            if p.path == '/api/scan':     return self.api_scan(body)
            if p.path == '/api/export':   return self.api_export(body)
            if p.path == '/api/albums':   return self.api_albums(body)
            if p.path == '/api/reassign':  return self.api_reassign(body)
            if p.path == '/api/assign_role': return self.api_assign_role(body)
            if p.path == '/api/score':     return self.api_score(body)
            if p.path == '/api/resolve_person': return self.api_resolve_person(body)
            self.send_error(404)
        except Exception as e:
            traceback.print_exc()
            self._json({'ok': False, 'error': str(e)}, 500)

    def _read_json(self):
        n = int(self.headers.get('Content-Length', 0))
        raw = self.rfile.read(n)
        for enc in ('utf-8', 'gbk', 'gb2312', 'utf-16'):
            try:
                return json.loads(raw.decode(enc))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        return json.loads(raw.decode('utf-8', errors='replace'))

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def _html(self, text):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.end_headers()
        self.wfile.write(text.encode('utf-8'))

    def html_index(self):
        self._html(_get_html_template())

    def api_reload(self):
        """热更新：restart=1 重启整个 Python 进程（后端代码生效）；默认只重载 HTML 模板。"""
        try:
            restart = parse_qs(urlparse(self.path).query).get('restart', ['0'])[0] == '1'
            if restart:
                self._json({'ok': True, 'restarting': True})
                def _restart():
                    import time
                    time.sleep(0.5)  # 等响应发出
                    print(">> 重启进程...")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                threading.Thread(target=_restart, daemon=True).start()
                return
            global HTML_TEMPLATE
            src = Path(__file__).read_text(encoding='utf-8')
            m = re.search(r'HTML_TEMPLATE\s*=\s*r"""(.*?)"""', src, re.DOTALL)
            if m:
                HTML_TEMPLATE = m.group(1)
                self._json({'ok': True})
            else:
                self._json({'ok': False, 'error': 'HTML_TEMPLATE not found in source'})
        except Exception as e:
            self._json({'ok': False, 'error': str(e)})

    def api_status(self):
        self._json({'scoring_busy': scoring_busy})

    def api_log(self, q):
        """前端日志落盘：POST/GET msg=... 存到 _tg_cache/pv_debug.log，GET 无参=读取全部"""
        logf = _cache_dir(Path(scan_root)) / 'pv_debug.log' if scan_root else None
        msg = q.get('msg', [None])[0]
        if msg:
            line = f"[{datetime.now():%H:%M:%S}] {unquote(msg)}\n"
            try:
                if logf:
                    with open(logf, 'a', encoding='utf-8') as f:
                        f.write(line)
            except Exception:
                pass
            self._json({'ok': True})
            return
        data = ''
        if logf and logf.exists():
            try:
                data = logf.read_text(encoding='utf-8')
            except Exception:
                pass
        self._json({'ok': True, 'log': data})

    def api_data(self):
        # 视图推断：view=albums 强制相册首页（"返回相册"）；否则按当前状态推断
        q = parse_qs(urlparse(self.path).query)
        force_albums = q.get('view', [''])[0] == 'albums'
        try:
            cur_is_root = bool(scan_root) and _is_album_root(Path(scan_root))
            roots = _load_roots()
            scan_is_root = bool(scan_root) and str(Path(scan_root).resolve()) in [str(Path(r).resolve()) for r in roots]
            # 相册首页：强制返回 / 当前目录是相册根 / 尚未进入具体相册（空 scan_root）
            if force_albums or cur_is_root or scan_is_root or not scan_root:
                rlist = list(roots)
                if (cur_is_root or scan_is_root) and str(Path(scan_root).resolve()) not in rlist:
                    rlist.append(str(Path(scan_root).resolve()))
                albums = _cached_albums(rlist)
                if albums is None:
                    albums = _rebuild_albums(rlist)
                    _save_album_cache(albums)
                if not rlist:
                    # 首次启动：未配置任何相册根目录 → 引导页
                    return self._json({
                        'view': 'setup', 'dir': current_dir or '',
                        'roots': [], 'albums': [], 'persons': [], 'roles': [],
                    })
                persons_db = _load_persons().get('persons', {})
                roles_db = _load_roles().get('roles', {})
                all_pids = sorted(persons_db.keys(), key=int)
                persons = [{'id': int(pid), 'count': persons_db[pid].get('count', 0),
                            'color': _person_color(int(pid)),
                            'name': persons_db[pid].get('name', ''),
                            'confirmed': persons_db[pid].get('confirmed', False)}
                           for pid in all_pids]
                roles = [{'id': int(rid), 'count': 0,
                          'color': _person_color(int(rid) + 100),
                          'name': roles_db[str(rid)].get('name', '')}
                         for rid in sorted(roles_db.keys(), key=int)]
                return self._json({
                    'dir': current_dir, 'view': 'albums',
                    'roots': rlist, 'albums': albums,
                    'persons': persons, 'roles': roles,
                })
        except Exception:
            import traceback; traceback.print_exc()
        with _state_lock:
            snap = [[p for p in g] for g in groups]
        pcounts, rcounts = {}, {}
        for g in snap:
            for p in g:
                if p.person_id is not None:
                    pcounts[p.person_id] = pcounts.get(p.person_id, 0) + 1
                if p.role_id is not None:
                    rcounts[p.role_id] = rcounts.get(p.role_id, 0) + 1
        persons_db = _load_persons().get('persons', {})
        roles_db = _load_roles().get('roles', {})
        # 人物：仅当前目录出现的（含本文件夹新聚类与命中的全局人物），不显示其他文件夹的人物
        all_pids = set(pcounts.keys())
        persons = [{'id': pid, 'count': pcounts.get(pid, 0), 'color': _person_color(pid),
                    'name': persons_db.get(str(pid), {}).get('name', ''),
                    'confirmed': persons_db.get(str(pid), {}).get('confirmed', False)}
                   for pid in sorted(all_pids)]
        # 合照类别："含xxx的合照"——合照中含某 coser，按 coser 聚合
        group_counts = {}  # {pid: 合照数}
        for g in snap:
            for p in g:
                if p.is_group and p.group_pids:
                    for gpid in p.group_pids:
                        group_counts[gpid] = group_counts.get(gpid, 0) + 1
        groups_list = [{'pid': gpid, 'count': cnt,
                        'name': '含' + (persons_db.get(str(gpid), {}).get('name', '') or ('人物' + str(gpid))) + '的合照',
                        'color': _person_color(gpid)}
                       for gpid, cnt in sorted(group_counts.items())]
        # 角色：仅当前目录出现的（同样只显示本文件夹涉及的角色）
        all_rids = set(rcounts.keys())
        roles = [{'id': rid, 'count': rcounts.get(rid, 0), 'color': _person_color(rid + 100),
                  'name': roles_db.get(str(rid), {}).get('name', '')}
                 for rid in sorted(all_rids)]
        self._json({
            'dir': current_dir,
            'groups': [[p.to_dict() for p in g] for g in snap],
            'persons': persons, 'roles': roles, 'group_photos': groups_list,
        })

    # ── 调试：返回人脸框和骨架数据 ─────────────────
    def api_debug_photo(self, photo_id):
        if photo_id is None:
            return self._json({'ok': False})
        p = get_photo_by_id(photo_id)
        if p is None:
            return self._json({'ok': False})
        try:
            import numpy as np
            from PIL import Image as _PIL
            lm = _get_face_landmarker()
            pm = _get_pose_landmarker()
            if not lm or not pm:
                return self._json({'ok': False, 'error': 'models not loaded'})
            cache = _cache_dir(Path(scan_root)) / f'thumb_{p.id}.jpg' if scan_root else None
            if cache and cache.exists():
                pil = _PIL.open(cache).convert('RGB')
            else:
                pil = _PIL.open(p.preview).convert('RGB')
            np_img = np.array(pil, dtype=np.uint8)
            import mediapipe as mp
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
            with _mp_lock:
                fr = lm.detect(mp_img)
                pr = pm.detect(mp_img)
            faces = [[{'x': p.x, 'y': p.y, 'z': p.z} for p in f] for f in fr.face_landmarks]
            poses = [[{'x': p.x, 'y': p.y, 'z': p.z} for p in pose] for pose in pr.pose_landmarks]

            return self._json({
                'ok': True, 'w': pil.width, 'h': pil.height,
                'faces': faces, 'poses': poses,
                'score': p.score, 's_sharp': p.score_sharp,
                's_exp': p.score_exp, 's_contrast': p.score_contrast, 's_face': p.score_face
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._json({'ok': False, 'error': str(e)})

    # ── 快速预览：始终生成 800px 小图 ──────────────
    def serve_preview_quick(self, photo_id):
        if photo_id is not None:
            p = get_photo_by_id(photo_id)
            if p is None: return self._placeholder()
            # 优先用缓存的缩略图（250px），秒出
            cf = _cache_path(Path(scan_root), p, 250) if scan_root else None
            if cf and cf.exists():
                data = cf.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            # 无缓存时生成 400px 小图（比 800px 快）
            data = make_thumb(p.preview, 400)
            if not data: return self._placeholder()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._placeholder()

    # ── 缩略图/预览图：优先读缓存 ──────────────────────
    def _send_jpeg(self, data: bytes):
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'max-age=3600')
        self.end_headers()
        self.wfile.write(data)

    def serve_img(self, photo_id, path_str, size):
        if photo_id is not None:
            p = get_photo_by_id(photo_id)
            if p is None:
                return self._placeholder()
            # 缩略图(≤500)优先读缓存；预览图(>500)从原文件生成真正高清图
            if size <= 500:
                data = _read_cache_bytes(scan_root, p, 250)
                if data:
                    return self._send_jpeg(data)
            path = p.preview
        else:
            # 无 id 时仅允许 scan_root 子树内的路径（防穿越）
            safe = _safe_path(scan_root, unquote(path_str or ''))
            if (not safe or not safe.exists()):
                # 相册封面：也允许已配置相册根目录子树内的路径（跨根显示封面）
                for r in _load_roots():
                    s = _safe_path(r, unquote(path_str or ''))
                    if s and s.exists():
                        safe = s
                        break
            if not safe or not safe.exists():
                return self._placeholder()
            path = safe
        if size <= 500:
            data = make_thumb(path, size)
        else:
            data = serve_image(path)  # 预览图：HIF→2400px 高清，JPG→原文件
        if not data:
            return self._placeholder()
        self._send_jpeg(data)

    _PLACEHOLDER = (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00'
        b'\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xdb\x00C\x01\t\t\t\x0c\x0b\x0c\x18\r\r\x182!\x1c!22222222222222222222222222222222222222222222222222\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x00\x00\x00\x00\x01\x02\x03\x11\x04\x12!1\x06\x13Q\x07"aq\x142\x81\x91\xa1\xb1\xc1\xf0#$r3\x823\xb2\xe1\xd1\xff\xc4\x00\x15\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x03\x04\x05\x06\x04\x03\x00\x00\x00\x00\x00\x00\x01\x02\x03\x11\x04\x12!1\x06\x12AQ\x07"\xa1q\x13\x142\x81\x91\xa2\xb1\xc1\xf0#3\xb2\xd1r\xff\xc4\x00\x15\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x11\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xd9'
    )

    def _placeholder(self):
        self.send_response(200)
        self.send_header('Content-Type', 'image/jpeg')
        self.send_header('Cache-Control', 'max-age=3600')
        self.end_headers()
        self.wfile.write(self._PLACEHOLDER)

    # ── 人物/角色管理 API ─────────────────────────────
    def api_persons(self, q):
        """GET /api/persons → 全局人物库；POST 由 do_POST 处理（此为 GET）。"""
        db = _load_persons()
        out = [{'id': int(pid), 'name': p.get('name', ''), 'count': p.get('count', 0),
                'confirmed': p.get('confirmed', False),
                'color': _person_color(int(pid))}
               for pid, p in sorted(db['persons'].items(), key=lambda x: int(x[0]))]
        return self._json({'persons': out, 'next_id': db['next_id']})

    def api_roles(self, q):
        db = _load_roles()
        out = [{'id': int(rid), 'name': r.get('name', ''), 'color': _person_color(int(rid) + 100),
                'exemplar_count': len(r.get('exemplars', []))}
               for rid, r in sorted(db['roles'].items(), key=lambda x: int(x[0]))]
        return self._json({'roles': out, 'next_id': db['next_id']})

    def api_person_photos(self, q):
        """GET /api/person_photos?pid=1 → 跨目录返回该人物在所有相册的照片列表。"""
        pid = q.get('pid', [None])[0]
        if pid is None:
            return self._json({'ok': False, 'error': 'pid required'})
        pid = int(pid)
        albums = _load_albums().get('albums', [])
        results = []
        for alb in albums:
            sp = Path(alb['path']) / '_tg_cache' / 'state.json'
            if not sp.exists():
                continue
            try:
                state = json.loads(sp.read_text(encoding='utf-8'))
                for ppath, v in state.get('photos', {}).items():
                    if v.get('person_id') == pid:
                        results.append({'path': ppath, 'name': Path(ppath).name,
                                        'album': alb['name'], 'keep': v.get('keep', False),
                                        'star': v.get('star', False), 'score': v.get('score', 0)})
            except Exception:
                pass
        return self._json({'pid': pid, 'photos': results, 'count': len(results)})

    def api_merge_suggestions(self):
        return self._json({'suggestions': _compute_merge_suggestions()})

    def api_cleanup(self):
        """GET /api/cleanup → 清洗全局人物库（拆分污染 person）。"""
        return self._json(cleanup_persons())

    def api_reassign(self, body):
        """POST /api/reassign {ids, person_id, action?, target_id?, name?}"""
        action = body.get('action', 'reassign')
        if action == 'create_person':
            with _persons_lock:
                persons = _load_persons()
                pid = persons['next_id']
                persons['persons'][str(pid)] = {'embeddings': [], 'count': 0,
                    'exemplar': '', 'name': body.get('name', '')}
                persons['next_id'] = pid + 1
                _save_persons(persons)
            return self._json({'ok': True, 'id': pid})
        if action == 'rename':
            _rename_person(int(body['id']), body.get('name', ''))
            return self._json({'ok': True})
        if action == 'merge':
            ok = _merge_persons(int(body['target_id']), int(body['id']))
            return self._json({'ok': ok})
        if action == 'delete':
            with _persons_lock:
                persons = _load_persons()
                persons['persons'].pop(str(body['id']), None)
                _save_persons(persons)
            with _state_lock:
                for g in groups:
                    for p in g:
                        if p.person_id == body['id']:
                            p.person_id = None
                save_state(scan_root, groups)
            return self._json({'ok': True})
        # 默认：批量改 person_id
        ids = body.get('ids', [])
        pid = body.get('person_id')
        with _state_lock:
            for tid in ids:
                p = get_photo_by_id(int(tid))
                if p:
                    p.person_id = pid
            save_state(scan_root, groups)
        return self._json({'ok': True, 'count': len(ids)})

    def api_assign_role(self, body):
        """POST /api/assign_role {ids: [photo_id...], role_id: int|null, action?, name?}"""
        action = body.get('action', 'assign')
        if action == 'create':
            with _persons_lock:
                roles = _load_roles()
                rid = roles['next_id']
                roles['roles'][str(rid)] = {'name': body.get('name', ''), 'exemplars': [], 'color': _person_color(rid + 100)}
                roles['next_id'] = rid + 1
                _save_roles(roles)
            return self._json({'ok': True, 'id': rid})
        if action == 'rename':
            _rename_role(int(body['id']), body.get('name', ''))
            return self._json({'ok': True})
        if action == 'delete':
            with _persons_lock:
                roles = _load_roles()
                roles['roles'].pop(str(body['id']), None)
                _save_roles(roles)
            with _state_lock:
                for g in groups:
                    for p in g:
                        if p.role_id == body['id']:
                            p.role_id = None
                save_state(scan_root, groups)
            return self._json({'ok': True})
        # 默认：分配角色
        ids = body.get('ids', [])
        rid = body.get('role_id')
        with _state_lock:
            for tid in ids:
                p = get_photo_by_id(int(tid))
                if p:
                    p.role_id = rid
                    # 标注 exemplar（存路径供角色识别用）
                    if rid is not None:
                        with _persons_lock:
                            roles = _load_roles()
                            rk = str(rid)
                            if rk in roles['roles']:
                                exs = roles['roles'][rk].setdefault('exemplars', [])
                                ppath = str(p.preview)
                                if ppath not in exs:
                                    exs.append(ppath)
                                    if len(exs) > 10:
                                        exs[:] = exs[-10:]
                                _save_roles(roles)
            save_state(scan_root, groups)
        return self._json({'ok': True, 'count': len(ids)})

    def api_toggle(self, body):
        keep = body.get('keep', False)
        pid = body.get('id')
        with _state_lock:
            if pid is not None:
                p = get_photo_by_id(int(pid))
                if p:
                    p.keep = keep
                    save_state(scan_root, groups)
                    return self._json({'ok': True})
            path = body.get('path')
            if path:
                for g in groups:
                    for p in g:
                        if str(p.preview) == path:
                            p.keep = keep
                            save_state(scan_root, groups)
                            return self._json({'ok': True})
        self._json({'ok': False}, 404)

    def api_scan(self, body):
        global groups, current_dir, scan_root, _global_root_cache
        directory = body.get('directory', '')
        if not directory or not Path(directory).is_absolute() or not Path(directory).is_dir():
            return self._json({'ok': False, 'error': '目录不存在或非绝对路径'}, 400)
        scan_root = str(Path(directory).resolve())
        current_dir = str(Path(directory).resolve())
        _global_root_cache = None  # 重置缓存以适配新根
        # 相册根目录：已配置的根或旧启发式识别为根 → 只建索引（前端用 /api/data 拿 persons/roles）
        root_resolved = str(Path(directory).resolve())
        if root_resolved in [str(Path(r).resolve()) for r in _load_roots()] or _is_album_root(Path(directory)):
            groups = []
            self._json({'ok': True, 'dir': current_dir, 'view': 'albums',
                        'groups': [], 'albums': _build_album_index(Path(directory))})
            return
        # 相册子目录：scan 照片（不自动评分）
        with _state_lock:
            groups = scan(directory)
            rebuild_photo_list()
        self._json({
            'ok': True,
            'dir': current_dir, 'view': 'photos',
            'groups': [[p.to_dict() for p in g] for g in groups] if groups else [],
        })

    def api_score(self, body):
        """手动触发评分（缓存+评分+人物批量聚类+角色归类）。
        body.dims: 参与评分的维度子集（sharp/face/exp/contrast），缺省全开。
        人物识别为文件夹内批量聚类 + 全局质心匹配（compute_persons）。"""
        global scoring_busy
        if scoring_busy:
            return self._json({'ok': False, 'error': '评分进行中'})
        if not groups:
            return self._json({'ok': False, 'error': '没有照片'})
        # 校验 dims：只保留合法维度，空则默认全开
        raw_dims = body.get('dims') or []
        dims = [d for d in raw_dims if d in _ALL_DIMS]
        if not dims:
            dims = list(_ALL_DIMS)
        scoring_busy = True
        root_s = str(Path(scan_root).resolve()) if scan_root else ''
        photos_flat = [p for g in groups for p in g]
        def _bg_score():
            global scoring_busy
            try:
                pregenerate_cache(root_s, photos_flat)
                compute_subbursts(root_s, groups)
                compute_persons(root_s, groups)          # 批量聚类人物 + 全局质心匹配
                score_groups(root_s, groups, dims=dims)  # 质量评分（闭眼基线用 person_id）
                compute_roles(root_s, groups)
            except Exception as e:
                print(f"  >> 后台评分异常: {e}")
            finally:
                scoring_busy = False
        threading.Thread(target=_bg_score, daemon=True).start()
        self._json({'ok': True, 'dims': dims})

    def _get_local_person_embeddings(self, local_pid: int) -> list:
        """取某未确认人物（局部 person）在本文件夹内的代表 embedding。
        优先用全局库的 prototype，落空则从照片重新识别。"""
        import numpy as np
        # 全局库 prototype
        with _persons_lock:
            persons = _load_persons()
            pdata = persons.get('persons', {}).get(str(local_pid))
        if pdata:
            proto = pdata.get('prototype')
            if proto:
                return [np.asarray(proto, dtype=np.float32)]
        # 从照片重新识别
        embs = []
        for g in groups:
            for p in g:
                if p.person_id == local_pid:
                    try:
                        image = _load_score_image(scan_root, p)
                        if image is None:
                            continue
                        small = _resize_long(image, _FACE_LONG)
                        faces = _detect_faces(small)
                        if faces:
                            primary = max(faces, key=lambda f: f['area'])
                            e = primary.get('embedding')
                            if e is not None:
                                e = np.asarray(e, dtype=np.float32)
                                n = np.linalg.norm(e)
                                if n > 0:
                                    embs.append(e / n)
                                    if len(embs) >= 4:
                                        break
                    except Exception:
                        continue
        return embs

    def api_resolve_person(self, body):
        """POST /api/resolve_person
        {local_id: <未确认人物ID>, name?: str, target_id?: int|null}
        - 无 target_id：用该人物 prototype 匹配全局 confirmed 人物，返回候选
        - 有 target_id：归并到 target_id（或命名注册），更新本文件夹照片 person_id"""
        import numpy as np
        local_id = body.get('local_id')
        if local_id is None:
            return self._json({'ok': False, 'error': 'local_id required'})
        local_id = int(local_id)
        embs = self._get_local_person_embeddings(local_id)
        if not embs:
            return self._json({'ok': False, 'error': '未找到该人物的照片或 prototype'})
        rep = embs[0]

        target_id = body.get('target_id')
        name = body.get('name', '')
        if target_id is None:
            # 匹配全局 confirmed，返回候选
            pid, sim, _ = _match_person(rep, threshold=_PENDING_SIM)
            candidates = []
            if pid is not None:
                persons_db = _load_persons().get('persons', {})
                candidates = [{'id': pid, 'name': persons_db.get(str(pid), {}).get('name', ''),
                               'avg_sim': round(sim, 3),
                               'confirm': sim >= _AUTO_MERGE_SIM}]
            return self._json({'ok': True, 'local_id': local_id,
                               'candidates': candidates, 'embs': len(embs)})

        # 归并：target_id>0 归并到已有全局人物；否则注册新人物（命名）
        global_pid = None
        if target_id > 0:
            global_pid = int(target_id)
            _update_person(global_pid, rep)
        else:
            global_pid = _register_person(rep, name=name, confirmed=bool(name))
            for e in embs[1:]:
                _update_person(global_pid, e)
        # 更新本文件夹所有该人物的照片 person_id
        with _state_lock:
            for g in groups:
                for p in g:
                    if p.person_id == local_id:
                        p.person_id = global_pid
                        p.merge_pending = False
            save_state(scan_root, groups)
        return self._json({'ok': True, 'local_id': local_id,
                           'global_id': global_pid, 'name': name})

    def api_browse(self):
        import subprocess, os
        try:
            ps = [
                'powershell', '-NoProfile', '-Command',
                'Add-Type -AssemblyName System.Windows.Forms; '
                '$f=New-Object System.Windows.Forms.FolderBrowserDialog; '
                '$f.Description="选择照片目录"; '
                'if($f.ShowDialog() -eq "OK"){$f.SelectedPath}else{""}'
            ]
            r = subprocess.run(ps, capture_output=True, text=True, timeout=60)
            path = r.stdout.strip()
            if path and os.path.isdir(path):
                return self._json({'ok': True, 'path': path})
            return self._json({'ok': False, 'error': '未选择目录或已取消'})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._json({'ok': False, 'error': str(e)})

    def api_albums(self, body):
        """POST /api/albums → 管理相册根目录。
        {action:'add', path} / {action:'remove', path} / {action:'rescan'}
        返回最新 roots + 跨根聚合的相册索引。"""
        global _global_root_cache
        action = body.get('action', '')
        roots = _load_roots()
        if action == 'add':
            p = (body.get('path') or '').strip()
            if not p:
                return self._json({'ok': False, 'error': '缺少路径'})
            rp = Path(p).resolve()
            if not rp.is_dir():
                return self._json({'ok': False, 'error': '目录不存在: ' + p})
            rs = [Path(r).resolve() for r in roots]
            if rp not in rs:
                roots.append(str(rp))
                _save_roots(roots)
        elif action == 'remove':
            p = (body.get('path') or '').strip()
            rp = Path(p).resolve()
            roots = [r for r in roots if Path(r).resolve() != rp]
            _save_roots(roots)
        elif action == 'rescan':
            pass  # 仅重建索引
        else:
            return self._json({'ok': False, 'error': '未知操作: ' + str(action)})
        _global_root_cache = None  # 根目录变化可能影响全局库位置
        albums = _rebuild_albums(roots)
        _save_album_cache(albums)
        return self._json({'ok': True, 'roots': roots, 'albums': albums})

    def api_export(self, body):
        import shutil, datetime as dt
        kept = [p for g in groups for p in g if p.keep and p.raw]
        if not kept:
            return self._json({'ok': False, 'error': '没有选中的 ARW 文件'})
        ts = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        export_dir = Path(current_dir) / f'挑选_{ts}'
        names = [p.raw.name for p in kept]
        confirmed = body.get('confirmed', False)
        if not confirmed:
            # 预览：返回待导出清单 + 已存在于历史导出目录中的文件（提示增量）
            already = [n for n in names if n in _exported_arw_names(Path(current_dir))]
            return self._json({
                'ready': True, 'dir': str(export_dir), 'count': len(names),
                'files': names[:500], 'already': already, 'already_count': len(already),
            })
        # 执行复制：跳过目标目录内已存在的同名文件（幂等，不覆盖）
        export_dir.mkdir(parents=True, exist_ok=True)
        copied, skipped = [], []
        for p in kept:
            dst = export_dir / p.raw.name
            if dst.exists():
                skipped.append(p.raw.name)
                continue
            shutil.copy2(p.raw, dst)
            copied.append(p.raw.name)
        save_state(scan_root, groups)
        if skipped:
            print(f"> 导出 {len(copied)} 个 ARW 到 {export_dir.name}（跳过 {len(skipped)} 个已存在）")
        else:
            print(f"> 导出 {len(copied)} 个 ARW 到 {export_dir.name}")
        self._json({'done': True, 'dir': str(export_dir), 'count': len(copied),
                    'skipped': len(skipped), 'files': copied})

    def api_delete(self, body):
        confirmed = body.get('confirmed', False)
        mode = body.get('mode', 'unselected')
        types = body.get('types', ['JPG', 'HIF', 'ARW'])
        with _state_lock:
            del_list, keep_names = compute_delete_list(groups, mode, types)
        if not confirmed:
            return self._json({
                'ready': True, 'total_previews': sum(len(g) for g in groups),
                'to_delete': [str(f) for f in del_list[:200]],
                'to_delete_count': len(del_list),
                'kept': keep_names, 'kept_count': len(keep_names),
            })
        ok, err, _trash, touched = execute_delete(scan_root, del_list)
        if ok > 0:
            with _state_lock:
                for g in groups[:]:
                    g[:] = [p for p in g if p.preview.exists()]
                groups[:] = [g for g in groups if g]
                rebuild_photo_list()
                save_state(scan_root, groups)
        _cleanup_empty_dirs(scan_root, touched)
        if err: print(f"> 已移入回收站 {ok} 个文件 ({err} 个失败)")
        else: print(f"> 已移入回收站 {ok} 个文件")
        self._json({'done': True, 'deleted': ok, 'errors': err})

# ═══════════════════════════════════════════════════════════
#  HTML 模板
# ═══════════════════════════════════════════════════════════

def _get_html_template():
    """从源文件动态读取 HTML 模板，支持热更新后即时获取新前端代码。"""
    try:
        src = Path(__file__).read_text(encoding='utf-8')
        m = re.search(r'HTML_TEMPLATE\s*=\s*r"""(.*?)"""', src, re.DOTALL)
        if m:
            return m.group(1)
    except Exception:
        pass
    return HTML_TEMPLATE

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>连拍挑选</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'Segoe UI',sans-serif;background:#1a1a1a;color:#e0e0e0;overflow:hidden;height:100vh}
.hdr{position:sticky;top:0;z-index:99;background:#252525;border-bottom:1px solid #333}
.menubar{display:flex;align-items:center;gap:2px;height:32px;padding:0 8px;background:#2b2b2b;border-bottom:1px solid #222}
.app-title{font-size:13px;font-weight:600;color:#fff;padding:0 10px;white-space:nowrap;user-select:none}
.menu{position:relative}
.mi-title{padding:5px 10px;font-size:12px;color:#ccc;border-radius:4px;cursor:default;user-select:none;white-space:nowrap}
.mi-title:hover{background:#3a3a3a;color:#fff}
.menu.open .mi-title{background:#3d6df0;color:#fff}
.menu-drop{display:none;position:absolute;top:100%;left:0;min-width:220px;background:#2e2e2e;border:1px solid #444;border-radius:6px;padding:4px;box-shadow:0 10px 28px rgba(0,0,0,.55);z-index:310}
.menu.open .menu-drop{display:block}
.mi{display:flex;align-items:center;gap:8px;width:100%;text-align:left;padding:6px 10px;font-size:12px;color:#ddd;background:none;border:none;border-radius:4px;cursor:pointer;white-space:nowrap}
.mi:hover:not(:disabled){background:#3d6df0;color:#fff}
.mi:disabled{color:#555;cursor:default}
.mi .kbd{margin-left:auto;font-size:10px;color:#777}
.mi:hover:not(:disabled) .kbd{color:#bbb}
.mi .chk{width:14px;text-align:center;color:#5bdc78;font-size:12px}
.mi-sep{height:1px;background:#444;margin:4px 6px}
.toolbar{display:flex;align-items:center;gap:8px;padding:5px 12px;flex-wrap:wrap;background:#1f1f1f}
.dir-row{display:flex;align-items:center;gap:6px;flex:1 1 300px;min-width:200px}
.dir-row input{flex:1;background:#1a1a1a;border:1px solid #444;border-radius:4px;padding:5px 10px;color:#ddd;font-size:12px;outline:none}
.dir-row input:focus{border-color:#666}
.btn-open{background:#2980b9;color:#fff;padding:5px 10px;font-size:12px;cursor:pointer}
.btn-open:hover{background:#2471a3}
.btn-open:disabled{background:#444;color:#666;cursor:default}
.stats{color:#999;font-size:12px;white-space:nowrap}
.btn{padding:6px 14px;border:none;border-radius:4px;font-size:12px;cursor:pointer;white-space:nowrap}
.del{background:#c0392b;color:#fff}
.del:hover{background:#a93226}
.del:disabled{background:#444;color:#666;cursor:default}
.export{background:#2980b9;color:#fff}
.export:hover{background:#2471a3}
.export:disabled{background:#444;color:#666;cursor:default}
.del-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.del-row label{display:flex;align-items:center;gap:3px;cursor:pointer;color:#bbb;font-size:12px;white-space:nowrap}
.del-row label:hover{color:#ddd}
.del-row select{background:#333;border:1px solid #555;border-radius:3px;color:#ddd;padding:2px 6px;font-size:12px;cursor:pointer;outline:none}
.del-row select:hover{border-color:#777}
.del-row input[type=checkbox]{accent-color:#2980b9;margin:0}
.layout{display:flex;height:calc(100vh - 70px)}
.main{flex:1;overflow-y:auto;min-width:0}
.main.side-open{flex:0 0 calc(100% - 480px)}
.side{width:480px;border-left:1px solid #333;display:flex;flex-direction:column;background:#222;position:relative}
.side.closed{display:none}
.side-img{flex:1;overflow:hidden;position:relative;cursor:grab;background:#111}
.side-img:active{cursor:grabbing}
.side-img .iw{transform-origin:0 0;position:absolute;top:0;left:0}
.side-img .iw img{display:block;max-width:none;max-height:none;pointer-events:none}
.side-bar{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:#2a2a2a;border-top:1px solid #333;font-size:12px;color:#999;gap:12px}
.side-bar .pv-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.side-bar .pv-zoom{color:#bbb;white-space:nowrap}
.side-close{position:absolute;top:8px;right:8px;width:28px;height:28px;border-radius:4px;background:rgba(0,0,0,.6);border:none;color:#fff;font-size:16px;cursor:pointer;z-index:10;display:flex;align-items:center;justify-content:center}
.side-close:hover{background:rgba(200,50,50,.8)}
.side-hint{display:flex;align-items:center;justify-content:center;height:100%;color:#555;font-size:14px}
.grp{padding:12px 16px;border-bottom:1px solid #2a2a2a}
.ghdr{font-size:11px;color:#555;margin-bottom:6px}
.sburst{font-size:10px;color:#888;border-top:1px dashed #3a3a3a;margin:8px 0 4px;padding-top:4px}
.photos{display:flex;gap:8px;flex-wrap:wrap}
.card{position:relative;border:2px solid #444;border-radius:5px;overflow:hidden;cursor:pointer;transition:.12s;background:#222;flex-shrink:0}
.card:hover{border-color:#777}
.card.sel{border-color:#27ae60;background:#142814}
.card.lsel{border-color:#2980b9;box-shadow:0 0 6px rgba(41,128,185,.5)}
#labelPanel{position:fixed;top:70px;left:0;bottom:0;width:200px;background:#1f1f1f;border-right:1px solid #333;z-index:90;display:none;flex-direction:column;overflow-y:auto;padding:8px}
#labelPanel.on{display:flex}
#labelPanel h3{font-size:12px;color:#888;margin:8px 0 4px;border-bottom:1px solid #333;padding-bottom:4px}
#labelPanel .lp-item{display:flex;align-items:center;gap:6px;padding:4px 6px;cursor:pointer;border-radius:4px;font-size:12px;color:#ccc}
#labelPanel .lp-item:hover{background:#2a2a2a}
#labelPanel .lp-dot{width:12px;height:12px;border-radius:3px;flex-shrink:0}
#labelPanel .lp-count{color:#666;font-size:10px;margin-left:auto}
#labelPanel .lp-new{color:#2980b9;cursor:pointer;font-size:12px;padding:6px}
.card .thumb{display:block;max-height:150px;pointer-events:none}
.card .star{position:absolute;top:4px;right:4px;font-size:14px;color:#f1c40f;text-shadow:0 0 3px rgba(0,0,0,.8);z-index:5;pointer-events:none;display:none}
.card .star.on{display:block}
.card .clabel{padding:2px 5px;font-size:9px;color:#666;text-align:center;white-space:nowrap}
.card .tag{color:#fff;font-size:8px;padding:1px 3px;border-radius:2px;text-shadow:0 0 2px rgba(0,0,0,.8);line-height:1.2}
.pbar{display:flex;flex-wrap:wrap;gap:6px;padding:6px 16px;background:#1f1f1f;border-bottom:1px solid #333;align-items:center}
.albums{display:flex;flex-wrap:wrap;gap:12px;padding:16px}
.abanner{width:100%;display:flex;align-items:center;gap:12px;padding:8px 0;margin-bottom:8px}
.abanner h2{font-size:16px;color:#fff;flex:1}
.acard{width:180px;cursor:pointer;transition:.12s;border-radius:8px;overflow:hidden;background:#222;border:1px solid #333}
.acard:hover{border-color:#666;transform:translateY(-2px)}
.acover{width:100%;height:120px;overflow:hidden;display:flex;align-items:center;justify-content:center}
.acover img{width:100%;height:100%;object-fit:cover}
.acover .picon{font-size:32px;color:rgba(255,255,255,.5)}
.ainfo{padding:8px}
.aname{font-size:11px;color:#ddd;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ameta{font-size:10px;color:#666;margin-top:2px}
.adots{display:flex;gap:2px;margin-top:4px}
.adots .pdot{width:8px;height:8px;border-radius:2px}
.aroot{width:100%;margin-top:4px}
.aroot-hdr{display:flex;align-items:center;gap:10px;padding:6px 2px;border-bottom:1px solid #2c2c2c;margin-bottom:4px}
.aroot-name{font-size:13px;color:#eee;font-weight:600;white-space:nowrap}
.aroot-path{font-size:11px;color:#666;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.aroot-cnt{font-size:11px;color:#999;white-space:nowrap}
.aroot-grid{display:flex;flex-wrap:wrap;gap:12px;padding:4px 0}
.aroot-row{display:flex;align-items:center;gap:8px;background:#1a1a1a;border:1px solid #333;border-radius:4px;padding:5px 10px;margin:4px 0;font-size:12px;color:#ccc}
.aroot-row span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.setup{max-width:580px;margin:60px auto 0;padding:24px;background:#1f1f1f;border:1px solid #333;border-radius:10px}
.setup h2{font-size:18px;color:#fff;margin-bottom:8px}
.setup-sub{font-size:13px;color:#aaa;line-height:1.7}
.setup-sub code{background:#111;padding:1px 5px;border-radius:3px;color:#7ec8ff}
#albumRootInput:focus{border-color:#666}
#setupRootInput:focus{border-color:#666}
.pbar .plabel{font-size:14px}
.pbar .pchip{font-size:11px;padding:3px 10px;border:1px solid;border-radius:14px;cursor:pointer;color:#ddd;background:#2a2a2a;transition:.12s}
.pbar .pchip:hover{filter:brightness(1.2)}
.pbar .pchip.on{color:#fff;font-weight:bold}
.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:200;align-items:center;justify-content:center}
.modal.on{display:flex}
.mc{background:#2a2a2a;border-radius:10px;padding:20px;max-width:580px;width:92%;max-height:85vh;overflow-y:auto}
.mc h2{font-size:15px;margin-bottom:10px}
.mc p,.mc label{font-size:13px;color:#bbb;margin-bottom:4px}
#filterTags label{display:inline-flex;align-items:center;gap:4px;margin:2px 8px 2px 0;cursor:pointer;white-space:nowrap}
#filterTags input[type=checkbox]{accent-color:#16a085;margin:0}
#filterPanel select{background:#333;border:1px solid #555;border-radius:3px;color:#ddd;padding:4px 8px;font-size:13px;width:100%}
.ma{margin-top:14px;display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap}
.ma .btn-cancel{background:#444;color:#ccc}
.ma .btn-cancel:hover{background:#555}
.flist{max-height:160px;overflow-y:auto;background:#1a1a1a;padding:6px 8px;border-radius:4px;margin:8px 0;font-size:11px;color:#888;word-break:break-all}
.spin{border:3px solid #333;border-top:3px solid #e74c3c;border-radius:50%;width:26px;height:26px;animation:s 1s linear infinite;margin:0 auto 10px}
@keyframes s{to{transform:rotate(360deg)}}
.note{font-size:11px;color:#555;text-align:center;padding:14px}
#logPanel{position:fixed;bottom:0;left:0;right:0;height:180px;background:#111;color:#0f0;font:11px monospace;overflow-y:auto;padding:6px 10px;z-index:300;display:none;border-top:2px solid #333}
#logPanel.on{display:block}
#logPanel .le{white-space:pre-wrap;word-break:break-all}
#logPanel .lw{color:#fff}
@media(max-width:900px){.side{position:fixed;right:0;top:70px;bottom:0;width:100%;z-index:150}.side.closed{display:none}.main.side-open{flex:1}}
</style>
</head>
<body>
<div class="hdr">
  <div class="menubar">
    <span class="app-title">📷 连拍挑选</span>
    <div class="menu" data-menu="file">
      <span class="mi-title" onclick="toggleMenu(this)">文件</span>
      <div class="menu-drop">
        <button class="mi" onclick="focusDir()">打开目录…<span class="kbd">Ctrl+O</span></button>
        <button class="mi" onclick="browseDir()">浏览文件夹…</button>
        <button class="mi" id="backBtn" onclick="backToAlbums()">返回相册</button>
        <div class="mi-sep"></div>
        <button class="mi" id="exportMi" onclick="exportARW()">导出 ARW<span class="kbd" id="exportMiCnt"></span></button>
        <div class="mi-sep"></div>
        <button class="mi" onclick="openAlbumPanel()">相册管理…<span class="kbd" id="albumRootCnt"></span></button>
      </div>
    </div>
    <div class="menu" data-menu="view">
      <span class="mi-title" onclick="toggleMenu(this)">视图</span>
      <div class="menu-drop">
        <button class="mi" id="filterBtn" onclick="toggleFilterPanel()"><span class="chk"></span>筛选…</button>
        <button class="mi" id="labelBtn" onclick="toggleLabel()"><span class="chk" id="labelChk"></span>标注模式</button>
        <div class="mi-sep"></div>
        <button class="mi" id="debugBtn" onclick="toggleDebug()"><span class="chk" id="debugChk"></span>调试</button>
        <button class="mi" id="logBtn" onclick="toggleLog()"><span class="chk" id="logChk"></span>日志</button>
      </div>
    </div>
    <div class="menu" data-menu="score">
      <span class="mi-title" onclick="toggleMenu(this)">评分</span>
      <div class="menu-drop">
        <button class="mi" id="scoreBtn" onclick="openScorePanel()"><span class="chk"></span>评分配置…</button>
      </div>
    </div>
    <div class="menu" data-menu="help">
      <span class="mi-title" onclick="toggleMenu(this)">帮助</span>
      <div class="menu-drop">
        <button class="mi" id="reloadBtn" onclick="hotReload(event)" title="点击=仅前端热更新，Shift+点击=重启整个服务">热更新前端<span class="kbd">Shift=重启</span></button>
        <button class="mi" onclick="aboutInfo()">使用说明…</button>
      </div>
    </div>
    <span class="stats" id="st" style="margin-left:auto;padding-right:10px">加载中…</span>
  </div>
  <div class="toolbar">
    <div class="dir-row">
      <input id="dirInput" type="text" placeholder="D:\0_camera\..." value="" onkeydown="if(event.key==='Enter')openDir()">
      <button class="btn btn-open" id="openBtn" onclick="openDir()">打开</button>
      <button class="btn" style="background:#555;color:#ccc;padding:5px 10px" onclick="browseDir()">📂 浏览</button>
      <button class="btn" id="homeBtn" onclick="backToAlbums()" style="display:none;background:#27ae60;color:#fff;font-weight:600" title="返回相册主界面">⌂ 主界面</button>
    </div>
    <div class="del-row" style="flex:1 1 auto;min-width:0">
      <span style="color:#999;font-size:12px">🗑️</span>
      <select id="delMode" onchange="updDelBtn()">
        <option value="selected">选中的</option>
        <option value="unselected">未选中的</option>
      </select>
      <span style="color:#555">|</span>
      <label><input type="checkbox" id="tJPG" checked onchange="updDelBtn()">JPG</label>
      <label><input type="checkbox" id="tHIF" onchange="updDelBtn()">HIF/HEIC</label>
      <label><input type="checkbox" id="tRAW" checked onchange="updDelBtn()">ARW</label>
      <span class="stats" id="delPreview" style="font-size:11px;color:#888"></span>
      <span style="flex:1"></span>
      <button class="btn export" id="exportBtn" onclick="exportARW()" disabled>导出ARW</button>
      <button class="btn del" id="delBtn" onclick="execDel()" disabled>删除</button>
    </div>
  </div>
</div>
<div id="mergeBar" style="display:none;background:#1a1a2e;border-bottom:1px solid #333;padding:4px 16px;gap:6px;flex-wrap:wrap;align-items:center"></div>
<div class="layout">
  <div id="labelPanel"><h3>人物 (coser)</h3><div id="lpPersons"></div><div class="lp-new" onclick="createPerson()">➕ 新建人物</div><h3>角色</h3><div id="lpRoles"></div><div class="lp-new" onclick="createRole()">➕ 新建角色</div><div class="lp-new" onclick="clearLabelSelection()">✕ 清除选中</div></div>
  <div class="main" id="mainArea">
    <div id="grp"></div>
    <div class="note" id="noteBar">左键切换保留 · 右键预览大图</div>
  </div>
  <div class="side closed" id="side">
    <button class="side-close" onclick="closePreview()">✕</button>
    <div class="side-img" id="pvArea">
      <div class="side-hint" id="pvHint">点击照片查看大图</div>
      <div class="iw" id="pvWrap" style="display:none"><img id="pvImg"><canvas id="pvCanvas" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;display:none"></canvas></div>
    </div>
    <div class="pv-progress" id="pvBar" style="height:3px;background:#333;border-radius:0">
      <div style="height:100%;width:0%;background:#2980b9;transition:width .3s" id="pvBarInner"></div>
    </div>
    <div class="side-bar">
      <span class="pv-name" id="pvName"></span>
      <span class="pv-zoom" id="pvZoom">100%</span>
    </div>
  </div>
</div>

<div class="modal" id="md"><div class="mc" id="mb"></div></div>

<!-- 评分配置面板 -->
<div class="modal" id="scorePanel">
  <div class="mc" style="max-width:380px">
    <h2>⚙️ 评分配置</h2>
    <p style="color:#888;font-size:12px">选择参与评分的维度（未勾选的不计入总分）</p>
    <label><input type="checkbox" id="dimSharp" checked onchange="updateScoreDims()"> 对焦/清晰度</label><br>
    <label><input type="checkbox" id="dimFace" checked onchange="updateScoreDims()"> 表情/人脸</label><br>
    <label><input type="checkbox" id="dimExp" checked onchange="updateScoreDims()"> 曝光</label><br>
    <label><input type="checkbox" id="dimContrast" checked onchange="updateScoreDims()"> 对比度</label>
    <div class="ma">
      <button class="btn btn-cancel" onclick="closeScorePanel()">取消</button>
      <button class="btn" style="background:#e67e22;color:#fff" onclick="confirmScoring()">确认开始评分</button>
    </div>
  </div>
</div>

<!-- 筛选面板 -->
<div class="modal" id="filterPanel">
  <div class="mc" style="max-width:340px">
    <h2>🎚 视图筛选</h2>
    <p style="color:#888;font-size:12px">按分数区间</p>
    <select id="filterScore" onchange="applyFilter()">
      <option value="">全部分数</option>
      <option value="80">≥ 80</option>
      <option value="60-79">60 - 79</option>
      <option value="40-59">40 - 59</option>
      <option value="lt40">&lt; 40</option>
      <option value="0">0（废片）</option>
    </select>
    <p style="color:#888;font-size:12px;margin-top:10px">按标签</p>
    <div id="filterTags">
      <label><input type="checkbox" value="废片" onchange="applyFilter()"> 废片</label>
      <label><input type="checkbox" value="闭眼" onchange="applyFilter()"> 闭眼</label>
      <label><input type="checkbox" value="欠曝" onchange="applyFilter()"> 欠曝</label>
      <label><input type="checkbox" value="过曝" onchange="applyFilter()"> 过曝</label>
      <label><input type="checkbox" value="wink" onchange="applyFilter()"> wink</label>
      <label><input type="checkbox" value="合照" onchange="applyFilter()"> 合照</label>
      <label><input type="checkbox" value="待确认" onchange="applyFilter()"> 待确认</label>
    </div>
    <div class="ma">
      <button class="btn btn-cancel" onclick="closeFilterPanel()">关闭</button>
      <button class="btn" onclick="resetFilter()">清除筛选</button>
    </div>
  </div>
</div>

<!-- 相册管理面板 -->
<div class="modal" id="albumPanel">
  <div class="mc" style="max-width:560px">
    <h2>🗂 相册管理</h2>
    <p style="color:#888;font-size:12px">添加一个或多个相册根目录（如 D:\0_camera），自动检索底下的照片文件夹。全局人物/角色库保存在第一个相册根目录的 <code>_tg_global</code>。</p>
    <div id="albumRootList"></div>
    <div style="display:flex;gap:6px;margin:10px 0">
      <input id="albumRootInput" type="text" placeholder="相册根目录路径" style="flex:1;background:#1a1a1a;border:1px solid #444;border-radius:4px;padding:5px 10px;color:#ddd;font-size:12px;outline:none" onkeydown="if(event.key==='Enter')addAlbumRoot()">
      <button class="btn" onclick="browseAlbumRoot()">浏览…</button>
      <button class="btn" style="background:#2980b9;color:#fff" onclick="addAlbumRoot()">添加</button>
    </div>
    <div class="ma">
      <button class="btn btn-cancel" onclick="closeAlbumPanel()">关闭</button>
      <button class="btn" onclick="rescanAlbums()">重新扫描</button>
    </div>
  </div>
</div>

<div id="logPanel"></div>

<script>
let DATA=[], GROUPS=[], TP=0, TK=0, CACHE_BUST=0, FILTER_PID=null, FILTER_GROUP=null;
let VIEW='photos'; // 'albums' | 'photos' | 'persons' | 'person_photos'
let ALBUM_ROOT=''; // 相册根目录（打开相册时记住，返回时用）
let LABEL_MODE=false, LABEL_SEL=new Set();

const PV={el:null,wrap:null,img:null,scale:1,tx:0,ty:0,drag:false,dsx:0,dsy:0,dtx:0,dty:0};

async function load(v){const u=v==='albums'?'/api/data?view=albums':'/api/data';
  const _g=document.getElementById('grp');if(_g)_g.innerHTML='<div style="padding:48px"><div class="spin"></div></div>';
  const r=await fetch(u);const d=await r.json();DATA=d;
  VIEW=d.view==='albums'?'albums':(d.view==='setup'?'setup':'photos');
  if(VIEW==='albums') ALBUM_ROOT=DATA.dir;
  render();updateButtons();updRootCnt()}
function updRootCnt(){const rc=document.getElementById('albumRootCnt');if(rc)rc.textContent=(DATA.roots||[]).length?('· '+(DATA.roots||[]).length):''}
function render(){
  if(VIEW==='setup'){renderSetup();return updateButtons()}
  if(VIEW==='albums'){renderAlbums();return updateButtons()}
  if(VIEW==='persons'){renderPersons();return updateButtons()}
  if(VIEW==='person_photos'){renderPersonPhotos();return updateButtons()}
  renderPhotos();updateButtons()}
function _pname(pid){const ps=(DATA.persons||[]).find(x=>x.id===pid);return ps&&ps.name?ps.name:'人物'+pid}
function _rname(rid){const rs=(DATA.roles||[]).find(x=>x.id===rid);return rs&&rs.name?rs.name:'角色'+rid}
function _pcol(pid){const P=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#e91e63','#00bcd4','#f1c40f'];return P[pid%P.length]}
function _rcol(rid){return _pcol(rid+100)}
function renderPhotos(){
  GROUPS=DATA.groups||[];TP=0;GROUPS.forEach(g=>TP+=g.length);
  document.getElementById('dirInput').value=DATA.dir||'';
  // 人物筛选条（双击改名）
  let ph='';
  if(DATA.persons&&DATA.persons.length){
    ph+='<div class="pbar"><span class="plabel">👤</span>';
    ph+=`<span class="pchip ${FILTER_PID===null&&FILTER_GROUP===null?'on':''}" onclick="setFilter(null)" style="border-color:#666">全部</span>`;
    DATA.persons.forEach(ps=>{
      const on=FILTER_PID===ps.id;
      const label=ps.name||('人物'+ps.id);
      const unconf=!ps.name&&!ps.confirmed?'style="border-style:dashed"':'';
      ph+=`<span class="pchip ${on?'on':''}" draggable="true" onclick="setFilter(${ps.id})" ondblclick="renamePerson(${ps.id})" oncontextmenu="event.preventDefault();mergeTo(${ps.id})" ondragstart="DRAG_PID=${ps.id}" ondragover="event.preventDefault()" ondrop="mergeDrop(${ps.id})" style="border-color:${ps.color};${on?'background:'+ps.color:''};${unconf?'border-style:dashed':''}" title="双击改名 · 右键合并 · 拖拽合并">${label} ${!ps.name?'🔶':''}<b>${ps.count}</b></span>`;
    });
    ph+='</div>';
  }
  // 合照类别筛选条（"含xxx的合照"）
  if(DATA.group_photos&&DATA.group_photos.length){
    ph+='<div class="pbar"><span class="plabel">👥</span>';
    DATA.group_photos.forEach(gp=>{
      const on=FILTER_GROUP===gp.pid;
      ph+=`<span class="pchip ${on?'on':''}" onclick="setGroupFilter(${gp.pid})" style="border-color:${gp.color};${on?'background:'+gp.color:''}">${gp.name} <b>${gp.count}</b></span>`;
    });
    ph+='</div>';
  }
  // 角色筛选条
  if(DATA.roles&&DATA.roles.length){
    ph+='<div class="pbar"><span class="plabel">🎭</span>';
    DATA.roles.forEach(rl=>{
      const label=rl.name||('角色'+rl.id);
      ph+=`<span class="pchip" ondblclick="renameRole(${rl.id})" style="border-color:${rl.color}" title="双击改名">${label} <b>${rl.count}</b></span>`;
    });
    ph+='</div>';
  }
  // 标注模式左侧面板
  if(LABEL_MODE) renderLabelPanel();
  let h=ph;GROUPS.forEach((g,gi)=>{
    let vis=g.filter(p=>(FILTER_PID===null||p.person_id===FILTER_PID)&&(FILTER_GROUP===null||(p.is_group&&p.group_pids&&p.group_pids.includes(FILTER_GROUP))||(!p.is_group&&p.person_id===FILTER_GROUP))&&_passScoreFilter(p)&&_passTagFilter(p));
    if(vis.length===0) return;
    h+=`<div class="grp"><div class="ghdr">组${gi+1} — ${g.length}张${(FILTER_PID!==null||FILTER_GROUP!==null)?' (筛选'+vis.length+')':''}</div><div class="photos">`;
    g.forEach((p,pi)=>{
      if(FILTER_PID!==null && p.person_id!==FILTER_PID) return;
      if(FILTER_GROUP!==null && !((p.is_group&&p.group_pids&&p.group_pids.includes(FILTER_GROUP))||(!p.is_group&&p.person_id===FILTER_GROUP))) return;
      if(!_passScoreFilter(p)||!_passTagFilter(p)) return;
      if(pi>0 && p.burst!==g[pi-1].burst){
        h+=`</div><div class="sburst">真连拍 ${p.burst+1}</div><div class="photos">`;
      }
      const b=p.has_raw?'<span class="badge b-r">R</span>':'<span class="badge b-n">J</span>';
      const tagMap={'废片':'#c0392b','闭眼':'#e67e22','欠曝':'#8e44ad','过曝':'#8e44ad','无明显差异':'#7f8c8d','wink':'#27ae60'};
      let tagsHtml=(p.tags||[]).map(t=>`<span class="tag" style="background:${tagMap[t]||'#555'}">${t}</span>`).join('');
      if(p.is_group) tagsHtml='<span class="tag" style="background:#8e44ad">合照</span>'+tagsHtml;
      if(p.merge_pending) tagsHtml='<span class="tag" style="background:#f1c40f">🔶待确认</span>'+tagsHtml;
      const pColor=p.person_id!==null?_pcol(p.person_id):'transparent';
      const rColor=p.role_id!==null?_rcol(p.role_id):'transparent';
      const selCls=p.keep?'sel':'';
      const lsel=LABEL_SEL.has(p.id)?'lsel':'';
      const clickFn=LABEL_MODE?`toggleLabelSel(${p.id})`:`toggleKeep(${p.id})`;
      const ctxMenu=LABEL_MODE?'event.preventDefault()':`event.preventDefault();openPreviewById(${p.id})`;
      const scoreTitle=`组内排名(百分位) · 对焦:${p.s_sharp??0} 曝光:${p.s_exp??0} 对比度:${p.s_contrast??0} 表情:${p.s_face??0}`;
      h+=`<div class="card ${selCls} ${lsel}" onclick="${clickFn}" oncontextmenu="${ctxMenu}" data-id="${p.id}" title="${p.merge_pending?'待确认归属':''}">
        <div class="star ${p.star?'on':''}">★</div>
        <div class="pmark" style="position:absolute;top:0;left:0;width:4px;height:100%;background:${pColor};z-index:6;${p.merge_pending?'opacity:.7':''}"></div>
        <div class="pmark" style="position:absolute;top:0;left:4px;width:3px;height:100%;background:${rColor};z-index:6"></div>
        <div class="score" style="position:absolute;bottom:20px;right:4px;font-size:9px;color:#ddd;background:rgba(0,0,0,.6);padding:1px 4px;border-radius:2px;display:block" title="${scoreTitle}">${p.score}</div>
        <div class="tags" style="position:absolute;top:4px;left:8px;display:flex;gap:2px;z-index:5">${tagsHtml}</div>
        <img class="thumb" src="/thumb?id=${p.id}&v=${CACHE_BUST}" loading="lazy" alt="${p.name}">
        <div class="clabel">${b} ${p.name}</div></div>`;
    });h+=`</div></div>`;
  });
  document.getElementById('grp').innerHTML=h;upd()
}
function _pcol(pid){const P=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22','#e91e63','#00bcd4','#f1c40f'];return P[pid%P.length]}
function setFilter(pid){FILTER_PID=pid;FILTER_GROUP=null;render()}
function setGroupFilter(pid){FILTER_GROUP=FILTER_GROUP===pid?null:pid;FILTER_PID=null;render()}

// ── 相册列表视图（跨根目录聚合）──────────────
function _albumCard(a){
  const pColors=(a.persons||[]).map(pid=>`<span class="pdot" style="background:${_pcol(pid)}"></span>`).join('');
  return `<div class="acard" onclick="openAlbum('${a.path.replace(/\\/g,'\\\\\\\\')}')">
    <div class="acover"><img src="/thumb?p=${encodeURIComponent(a.cover_path||a.path)}&v=${CACHE_BUST}" loading="lazy" onerror="this.style.display='none'"></div>
    <div class="ainfo"><div class="aname">${a.name}</div>
    <div class="ameta">${a.photo_count}张 · ${a.date||''}</div>
    <div class="adots">${pColors}</div></div></div>`;
}
function _rootName(r){const p=r.replace(/\\/g,'/').replace(/\/$/,'');const i=p.lastIndexOf('/');return i>=0?p.substring(i+1):p}
function renderAlbums(){
  document.getElementById('dirInput').value=DATA.dir||'';
  let albums=DATA.albums||[], roots=DATA.roots||[];
  if(!roots.length && albums.length){
    // 兼容无 roots 的数据（如 /api/scan 直返）：按 album.root 字段归组
    roots=[...new Set(albums.map(a=>a.root))];
  }
  let h='<div class="albums">';
  h+=`<div class="abanner"><h2>📷 ${albums.length} 个相册</h2>`;
  h+=`<button class="btn" style="background:#2980b9;color:#fff" onclick="VIEW='persons';loadPersons()">👤 人物库 (${(DATA.persons||[]).length})</button> `;
  h+=`<button class="btn" style="background:#8e44ad;color:#fff" onclick="VIEW='roles';loadRoles()">🎭 角色库 (${(DATA.roles||[]).length})</button> `;
  h+=`<button class="btn" style="background:#27ae60;color:#fff" onclick="openAlbumPanel()">➕ 添加/管理相册</button></div>`;
  if(!albums.length&&!roots.length){
    h+=`<div class="note" style="padding:24px">还没有相册。点击右上角「➕ 添加/管理相册」，添加照片根目录（如 D:\\0_camera）后会自动检索。</div>`;
  }
  roots.forEach(r=>{
    const subs=albums.filter(a=>a.root===r);
    h+=`<div class="aroot"><div class="aroot-hdr"><span class="aroot-name">📁 ${_rootName(r)}</span><span class="aroot-path" title="${r}">${r}</span><span class="aroot-cnt">${subs.reduce((s,a)=>s+a.photo_count,0)}张 · ${subs.length}个相册</span></div>`;
    if(!subs.length){
      h+=`<div class="note" style="text-align:left;padding:6px 2px">该目录下没有找到符合要求的照片文件夹（直接含 JPG/HIF/HEIC）。</div>`;
    }else{
      h+='<div class="aroot-grid">'+subs.map(_albumCard).join('')+'</div>';
    }
    h+='</div>';
  });
  h+='</div>';
  document.getElementById('grp').innerHTML=h;
  document.getElementById('st').textContent=`${albums.length} 个相册 · ${(DATA.persons||[]).length} 人物 · ${(DATA.roles||[]).length} 角色`;
}

// ── 首次启动引导页 ──────────────────────────────
function renderSetup(){
  document.getElementById('dirInput').value='';
  const h=`<div class="setup">
    <h2>📷 欢迎使用 连拍挑选</h2>
    <p class="setup-sub">添加一个或多个<b>相册根目录</b>（如 <code>D:\\0_camera</code>），程序会自动检索底下的照片文件夹并集中展示，方便统一挑选与管理。之后可在「文件 → 相册管理…」中随时增删。</p>
    <div style="display:flex;gap:6px;margin:16px 0">
      <input id="setupRootInput" type="text" placeholder="D:\\0_camera" style="flex:1;background:#1a1a1a;border:1px solid #444;border-radius:4px;padding:7px 12px;color:#ddd;font-size:13px;outline:none" onkeydown="if(event.key==='Enter')addSetupRoot()">
      <button class="btn" onclick="browseSetupRoot()">浏览…</button>
      <button class="btn" style="background:#2980b9;color:#fff" onclick="addSetupRoot()">添加</button>
    </div>
    <div id="setupRootList" style="margin:6px 0"></div>
    <div class="ma"><button class="btn" style="background:#27ae60;color:#fff" onclick="finishSetup()">开始使用 →</button></div>
  </div>`;
  document.getElementById('grp').innerHTML=h;
  document.getElementById('st').textContent='首次启动 · 添加相册根目录';
  renderSetupRootList();
}
function renderSetupRootList(){
  const el=document.getElementById('setupRootList');if(!el)return;
  const roots=DATA.roots||[];
  el.innerHTML=roots.length
    ? roots.map(r=>`<div class="aroot-row"><span title="${r}">📁 ${r}</span><button class="btn" style="background:#c0392b;color:#fff;padding:2px 10px" onclick="removeSetupRoot('${r.replace(/'/g,"\\'")}')">移除</button></div>`).join('')
    : '<div class="note" style="text-align:left;padding:6px 2px;color:#888">还没有相册根目录。</div>';
}
async function addSetupRoot(){
  const inp=document.getElementById('setupRootInput');const p=(inp.value||'').trim();
  if(!p){alert('请输入目录路径');return}
  const d=await apiAlbums({action:'add',path:p});
  if(!d.ok){alert(d.error||'添加失败');return}
  inp.value='';syncAlbumsData(d);renderSetupRootList();
}
async function browseSetupRoot(){
  try{const r=await fetch('/api/browse',{method:'POST'});const d=await r.json();
    if(d.ok&&d.path)document.getElementById('setupRootInput').value=d.path;
  }catch(e){alert('浏览失败: '+e.message)}
}
async function removeSetupRoot(path){
  const d=await apiAlbums({action:'remove',path});
  if(!d.ok){alert(d.error||'移除失败');return}
  syncAlbumsData(d);renderSetupRootList();
}
function finishSetup(){VIEW='albums';load('albums')}

// ── 相册管理（弹窗）─────────────────────────────
async function apiAlbums(body){
  const r=await fetch('/api/albums',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return r.json();
}
function syncAlbumsData(d){
  if(d&&d.ok!==false){
    DATA.roots=d.roots||DATA.roots||[];
    DATA.albums=d.albums||DATA.albums||[];
  }
}
function openAlbumPanel(){
  document.getElementById('albumPanel').classList.add('on');
  renderAlbumRootList();
}
function closeAlbumPanel(){document.getElementById('albumPanel').classList.remove('on')}
function renderAlbumRootList(){
  const el=document.getElementById('albumRootList');if(!el)return;
  const roots=DATA.roots||[];
  el.innerHTML=roots.length
    ? roots.map(r=>`<div class="aroot-row"><span title="${r}">📁 ${r}</span><button class="btn" style="background:#c0392b;color:#fff;padding:2px 10px" onclick="removeAlbumRoot('${r.replace(/'/g,"\\'")}')">移除</button></div>`).join('')
    : '<div class="note" style="text-align:left;padding:8px 2px">尚未添加相册根目录。</div>';
}
async function addAlbumRoot(){
  const inp=document.getElementById('albumRootInput');const p=(inp.value||'').trim();
  if(!p){alert('请输入目录路径');return}
  const d=await apiAlbums({action:'add',path:p});
  if(!d.ok){alert(d.error||'添加失败');return}
  inp.value='';syncAlbumsData(d);
  renderAlbumRootList();render();updRootCnt();
}
async function browseAlbumRoot(){
  try{const r=await fetch('/api/browse',{method:'POST'});const d=await r.json();
    if(d.ok&&d.path)document.getElementById('albumRootInput').value=d.path;
  }catch(e){alert('浏览失败: '+e.message)}
}
async function removeAlbumRoot(path){
  const d=await apiAlbums({action:'remove',path});
  if(!d.ok){alert(d.error||'移除失败');return}
  syncAlbumsData(d);
  renderAlbumRootList();render();updRootCnt();
}
async function rescanAlbums(){
  const d=await apiAlbums({action:'rescan'});
  if(!d.ok){alert(d.error||'扫描失败');return}
  syncAlbumsData(d);
  renderAlbumRootList();render();updRootCnt();
}
async function loadPersons(){
  const d=await(await fetch('/api/persons')).json();
  DATA.persons=d.persons||DATA.persons;
  renderPersons();
}
function renderPersons(){
  const persons=DATA.persons||[];
  let h='<div class="albums"><div class="abanner"><h2>👤 人物库 (${persons.length})</h2>';
  h+=`<button class="btn" onclick="VIEW='albums';load()">← 返回相册</button></div>`;
  persons.forEach(p=>{
    h+=`<div class="acard" onclick="openPersonPhotos(${p.id})" oncontextmenu="event.preventDefault();renamePerson(${p.id},'${(p.name||'').replace(/'/g,'')}')">
      <div class="acover" style="background:${p.color}"><span class="picon">${p.name?('👤'):(p.id+'')}</span></div>
      <div class="ainfo"><div class="aname">${p.name||('人物'+p.id)}</div>
      <div class="ameta">${p.count}张</div></div></div>`;
  });
  h+='</div>';
  document.getElementById('grp').innerHTML=h;
  updateButtons();
}
async function openPersonPhotos(pid){
  VIEW='person_photos';
  const d=await(await fetch('/api/person_photos?pid='+pid)).json();
  window._ppData=d;
  renderPersonPhotos();
}
function renderPersonPhotos(){
  const d=window._ppData;
  if(!d)return;
  let h='<div class="albums"><div class="abanner"><h2>👤 人物${d.pid} (${d.count}张)</h2>';
  h+=`<button class="btn" onclick="VIEW='persons';renderPersons()">← 返回人物库</button></div>`;
  (d.photos||[]).forEach(p=>{
    h+=`<div class="acard" onclick="window.open('${p.path.replace(/\\/g,'/')}', '_blank')">
      <div class="acover"><img src="/thumb?p=${encodeURIComponent(p.path)}&v=${CACHE_BUST}" loading="lazy" onerror="this.style.display='none'"></div>
      <div class="ainfo"><div class="aname">${p.name}</div>
      <div class="ameta">${p.album}</div></div></div>`;
  });
  h+='</div>';
  document.getElementById('grp').innerHTML=h;
  updateButtons();
}
async function loadRoles(){
  const d=await(await fetch('/api/roles')).json();
  DATA.roles=d.roles||DATA.roles;
  renderRoles();
}
function renderRoles(){
  const roles=DATA.roles||[];
  let h='<div class="albums"><div class="abanner"><h2>🎭 角色库 (${roles.length})</h2>';
  h+=`<button class="btn" onclick="VIEW='albums';load()">← 返回相册</button></div></div>`;
  document.getElementById('grp').innerHTML=h;
  updateButtons();
}

// ── 人物合并 ──────────────────────────────────
let DRAG_PID=null;
async function mergeTo(srcId){
  // 右键"合并到..."：弹出人物列表
  const ps=(DATA.persons||[]).filter(p=>p.id!==srcId);
  if(!ps.length){alert('没有其他人物可合并');return}
  const src=(DATA.persons||[]).find(p=>p.id===srcId);
  const srcLabel=src&&src.name?src.name:('人物'+srcId);
  let opts=ps.map(p=>`${p.name||('人物'+p.id)} (${p.count}张)`).join('\n');
  const dstIdx=prompt(`把 "${srcLabel}" 合并到：\n\n${opts}\n\n输入目标序号(1-${ps.length})`);
  if(!dstIdx) return;
  const dst=ps[parseInt(dstIdx)-1];
  if(!dst) return;
  if(!confirm(`确认把 "${srcLabel}" 合并到 "${dst.name||('人物'+dst.id)}"？`)) return;
  await mergePerson(srcId, dst.id);
}
async function mergeDrop(dstId){
  // 拖拽合并：DRAG_PID → dstId
  const srcId=DRAG_PID; DRAG_PID=null;
  if(srcId===null||srcId===dstId) return;
  const src=(DATA.persons||[]).find(p=>p.id===srcId);
  const dst=(DATA.persons||[]).find(p=>p.id===dstId);
  const sLabel=src&&src.name?src.name:('人物'+srcId);
  const dLabel=dst&&dst.name?dst.name:('人物'+dstId);
  if(!confirm(`合并 "${sLabel}" → "${dLabel}"？`)) return;
  await mergePerson(srcId, dstId);
}
async function mergePerson(srcId,dstId){
  const r=await fetch('/api/reassign',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'merge',id:srcId,target_id:dstId})});
  const d=await r.json();
  if(!d.ok){alert('合并失败');return}
  await reloadData();
  await loadMergeSuggestions();
}
async function loadMergeSuggestions(){
  const d=await(await fetch('/api/merge_suggestions')).json();
  const bar=document.getElementById('mergeBar');
  if(!d.suggestions||!d.suggestions.length){bar.style.display='none';return}
  bar.style.display='flex';
  let h='<span style="color:#f1c40f;font-size:12px">⚡合并建议</span> ';
  d.suggestions.forEach(s=>{
    const sL=s.src_name||('人物'+s.src_id);
    const dL=s.dst_name||('人物'+s.dst_id);
    h+=`<span style="font-size:11px;color:#bbb;background:#2a2a2a;padding:2px 8px;border-radius:3px;margin-right:4px">${sL}(${s.src_count}) → ${dL}(${s.dst_count}) <b style="color:#f1c40f">${s.similarity}</b> <a style="color:#27ae60;cursor:pointer" onclick="mergePerson(${s.src_id},${s.dst_id})">合并</a></span>`;
  });
  bar.innerHTML=h;
}

// ── 标注模式 ──────────────────────────────────
function toggleLabel(){
  LABEL_MODE=!LABEL_MODE;
  const chk=document.getElementById('labelChk');
  if(chk)chk.textContent=LABEL_MODE?'✓':'';
  document.getElementById('labelPanel').classList.toggle('on',LABEL_MODE);
  if(!LABEL_MODE) LABEL_SEL.clear();
  render();
}
function renderLabelPanel(){
  let h='';
  (DATA.persons||[]).forEach(ps=>{
    const label=ps.name||('人物'+ps.id);
    h+=`<div class="lp-item" onclick="assignToPerson(${ps.id})"><span class="lp-dot" style="background:${ps.color}"></span>${label}<span class="lp-count">${ps.count}</span></div>`;
  });
  document.getElementById('lpPersons').innerHTML=h;
  let rh='';
  (DATA.roles||[]).forEach(rl=>{
    const label=rl.name||('角色'+rl.id);
    rh+=`<div class="lp-item" onclick="assignToRole(${rl.id})"><span class="lp-dot" style="background:${rl.color}"></span>${label}<span class="lp-count">${rl.count}</span></div>`;
  });
  document.getElementById('lpRoles').innerHTML=rh;
}
function toggleLabelSel(id){
  if(LABEL_SEL.has(id)) LABEL_SEL.delete(id); else LABEL_SEL.add(id);
  const card=document.querySelector('.card[data-id="'+id+'"]');
  if(card) card.classList.toggle('lsel');
  document.getElementById('st').textContent=`标注模式 · 已选 ${LABEL_SEL.size} 张`;
}
async function assignToPerson(pid){
  if(LABEL_SEL.size===0){alert('请先点选照片');return}
  const ids=[...LABEL_SEL];
  await fetch('/api/reassign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:ids,person_id:pid})});
  LABEL_SEL.clear();await reloadData();
}
async function assignToRole(rid){
  if(LABEL_SEL.size===0){alert('请先点选照片');return}
  const ids=[...LABEL_SEL];
  await fetch('/api/assign_role',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:ids,role_id:rid})});
  LABEL_SEL.clear();await reloadData();
}
async function createPerson(){
  const name=prompt('新建人物名称（留空则自动编号）');
  if(name===null) return;
  // 创建新人物：注册一个空 embedding 人物（用 reassign 的人当前 person_id 设到一个新 id）
  // 简化：直接给选中照片分配到"新建"——后端没有直接新建空人物的 API，用 roles 的 create 模式类似
  // 这里用 reassign 到一个不存在的 id 触发后端注册（但 _merge_persons 不支持）
  // 实际：新建人物需后端支持，这里用 /api/reassign action=create
  const r=await fetch('/api/reassign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'create_person',name:name||''})});
  const d=await r.json();
  if(d.ok){await reloadData()}
}
async function createRole(){
  const name=prompt('新建角色名称（如：绝区零-妮可）');
  if(name===null) return;
  const r=await fetch('/api/assign_role',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'create',name:name})});
  const d=await r.json();
  if(d.ok){await reloadData()}
}
function clearLabelSelection(){LABEL_SEL.clear();render()}
async function renamePerson(pid,currentName){
  const name=prompt('修改人物名称',currentName||('人物'+pid));
  if(name===null) return;
  await fetch('/api/reassign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'rename',id:pid,name:name})});
  await reloadData();
}
async function renameRole(rid){
  const r=(DATA.roles||[]).find(x=>x.id===rid);
  const name=prompt('修改角色名称',r&&r.name?r.name:('角色'+rid));
  if(name===null) return;
  await fetch('/api/assign_role',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'rename',id:rid,name:name})});
  await reloadData();
}
async function reloadData(){
  const d=await(await fetch('/api/data')).json();DATA=d;render();
}

function toggleKeep(id){
  for(const g of GROUPS)for(const p of g){
    if(p.id!==id)continue;
    p.keep=!p.keep;
    const card=document.querySelector('.card[data-id="'+id+'"]');
    if(card)card.classList.toggle('sel');
    fetch('/api/toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:p.id,keep:p.keep})}).catch(()=>{});
    upd();
    return
  }
}

function upd(){
  let k=0;GROUPS.forEach(g=>g.forEach(p=>{if(p.keep)k++}));TK=k;
  document.getElementById('st').textContent=`${TP}张 · ${GROUPS.length}组 · 已选${k}张保留`;
  updDelBtn()
}

async function openDir(){
  const inp=document.getElementById('dirInput'),btn=document.getElementById('openBtn');
  const dir=inp.value.trim();if(!dir)return;
  btn.disabled=true;btn.textContent='扫描中…';
  document.getElementById('st').textContent='扫描目录中…';
  try{
    const r=await fetch('/api/scan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({directory:dir})});
    const d=await r.json();
    if(!d.ok){alert('目录扫描失败: '+d.error);return}
    CACHE_BUST=Date.now();DATA=d;closePreview();
    if(d.view==='albums'){
      VIEW='albums';ALBUM_ROOT=d.dir;
      // 补充获取 persons/roles（scan 只返回了 albums）
      try{const d2=await(await fetch('/api/data')).json();DATA=d2}catch(e){}
    }else{VIEW='photos'}
    render();
    updateButtons();
  }catch(e){alert('请求失败: '+e.message)}
  finally{btn.disabled=false;btn.textContent='打开'}
}
function openAlbum(path){
  // 记住相册根目录（打开相册时，DATA.dir 是根目录）
  if(DATA.dir && DATA.view==='albums') ALBUM_ROOT=DATA.dir;
  document.getElementById('dirInput').value=path;
  openDir();
}
function backToAlbums(){
  // 回到相册首页（跨根聚合）；未配置根时后端返回引导页
  ALBUM_ROOT='';
  VIEW='albums';
  load('albums');
}
function updateButtons(){
  const inPhotos = VIEW==='photos';
  const atHome = VIEW==='albums'||VIEW==='setup';
  // 主界面按钮：进入相册/人物/角色后显示，首页隐藏
  const hb=document.getElementById('homeBtn');
  if(hb)hb.style.display=atHome?'none':'inline-block';
  const b=document.getElementById('backBtn');
  if(b)b.disabled=atHome;
  ['scoreBtn','filterBtn'].forEach(id=>{
    const el=document.getElementById(id);
    if(el)el.disabled=!inPhotos;
  });
}

// ── 评分配置面板 ──────────────────────────
function openScorePanel(){
  // 回显上次选择
  if(window._scoreDims){
    document.getElementById('dimSharp').checked=window._scoreDims.includes('sharp');
    document.getElementById('dimFace').checked=window._scoreDims.includes('face');
    document.getElementById('dimExp').checked=window._scoreDims.includes('exp');
    document.getElementById('dimContrast').checked=window._scoreDims.includes('contrast');
  }
  document.getElementById('scorePanel').classList.add('on');
}
function closeScorePanel(){document.getElementById('scorePanel').classList.remove('on')}
function updateScoreDims(){/* 占位，无需实时动作 */}
function getSelectedDims(){
  const d=[];
  if(document.getElementById('dimSharp').checked)d.push('sharp');
  if(document.getElementById('dimFace').checked)d.push('face');
  if(document.getElementById('dimExp').checked)d.push('exp');
  if(document.getElementById('dimContrast').checked)d.push('contrast');
  return d;
}
async function confirmScoring(){
  const dims=getSelectedDims();
  if(!dims.length){alert('请至少选择一个评分维度');return}
  window._scoreDims=dims;
  closeScorePanel();
  try{
    const r=await fetch('/api/score',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dims:dims})});
    const d=await r.json();
    if(!d.ok){alert('评分启动失败: '+d.error);return}
    pollScores()
  }catch(e){alert('评分失败: '+e.message)}
}

// ── 筛选面板 ──────────────────────────
let FILTER_SCORE='', FILTER_TAGS=new Set();
function toggleFilterPanel(){
  const p=document.getElementById('filterPanel');
  p.classList.toggle('on');
}
function closeFilterPanel(){document.getElementById('filterPanel').classList.remove('on')}
function applyFilter(){
  FILTER_SCORE=document.getElementById('filterScore').value;
  FILTER_TAGS=new Set();
  document.querySelectorAll('#filterTags input:checked').forEach(c=>FILTER_TAGS.add(c.value));
  render();
}
function resetFilter(){
  FILTER_SCORE='';FILTER_TAGS=new Set();
  document.getElementById('filterScore').value='';
  document.querySelectorAll('#filterTags input').forEach(c=>c.checked=false);
  render();
}
function _passScoreFilter(p){
  if(!FILTER_SCORE) return true;
  const s=p.score;
  if(FILTER_SCORE==='80') return s>=80;
  if(FILTER_SCORE==='60-79') return s>=60&&s<=79;
  if(FILTER_SCORE==='40-59') return s>=40&&s<=59;
  if(FILTER_SCORE==='lt40') return s<40&&s>0;
  if(FILTER_SCORE==='0') return s===0;
  return true;
}
function _passTagFilter(p){
  if(FILTER_TAGS.size===0) return true;
  const tags=p.tags||[];
  for(const t of FILTER_TAGS){
    if(t==='待确认'&&p.merge_pending) return true;
    if(tags.includes(t)) return true;
  }
  return false;
}
async function startScoring(){openScorePanel()}

async function pollScores(){
  for(let i=0;i<600;i++){
    let s;try{s=await(await fetch('/api/status')).json()}catch(e){return}
    if(s.scoring_busy){
      document.getElementById('st').textContent=`${TP}张 · ${GROUPS.length}组 · 评分中…(${i*3}s)`;
      await new Promise(r=>setTimeout(r,3000))
    }else{
      try{const d=await(await fetch('/api/data')).json();DATA=d;render();updateButtons()}catch(e){}
      return
    }
  }
}

async function browseDir(){
  try{
    const r=await fetch('/api/browse',{method:'POST'});
    const d=await r.json();
    if(d.ok && d.path){
      document.getElementById('dirInput').value=d.path;
      openDir()
    }
  }catch(e){alert('浏览失败: '+e.message)}
}

async function exportARW(){
  const md=document.getElementById('md'),mb=document.getElementById('mb');
  md.classList.add('on');
  mb.innerHTML='<div class="spin"></div><p style="text-align:center;color:#999">正在统计待导出 ARW…</p>';
  try{
    const r=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmed:false})});
    const d=await r.json();
    if(!d.ok){mb.innerHTML='<p>'+esc(d.error)+'</p><div class="ma"><button class="btn btn-cancel" onclick="md.classList.remove(\'on\')">关闭</button></div>';return}
    let fl=(d.files||[]).map(f=>'<div>'+esc(f)+'</div>').join('');
    if(d.files.length<d.count)fl+='<div style="color:#555">…还有'+(d.count-d.files.length)+'个</div>';
    const alreadyNote=(d.already_count>0)?`<p style="font-size:12px;color:#e67e22">⚠ 其中 ${d.already_count} 个已在历史导出目录中出现过</p>`:'';
    mb.innerHTML=`<h2 style="color:#2980b9">导出 ARW</h2>
      <p>将复制 <b>${d.count}</b> 个 ARW 到:</p>
      <p style="font-size:12px;color:#888;word-break:break-all">${esc(d.dir)}</p>
      ${alreadyNote}
      <div class="flist">${fl}</div>
      <div class="ma">
        <button class="btn btn-cancel" onclick="md.classList.remove('on')">取消</button>
        <button class="btn export" onclick="exportARWConfirm()">确认导出</button>
      </div>`
  }catch(e){mb.innerHTML='<p>导出失败: '+esc(e.message)+'</p><div class="ma"><button class="btn btn-cancel" onclick="md.classList.remove(\'on\')">关闭</button></div>'}
}

async function exportARWConfirm(){
  const md=document.getElementById('md'),mb=document.getElementById('mb');
  mb.innerHTML='<div class="spin"></div><p style="text-align:center;color:#999">正在导出 ARW…</p>';
  try{
    const r=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmed:true})});
    const d=await r.json();
    if(!d.ok){mb.innerHTML='<p>'+esc(d.error)+'</p><div class="ma"><button class="btn btn-cancel" onclick="md.classList.remove(\'on\')">关闭</button></div>';return}
    mb.innerHTML=`<h2 style="color:#27ae60">导出完成</h2>
      <p>已复制 ${d.count} 个 ARW${d.skipped?`，跳过 ${d.skipped} 个已存在`:''}到:</p>
      <p style="font-size:12px;color:#888;word-break:break-all">${esc(d.dir)}</p>
      <div class="ma"><button class="btn refresh" onclick="md.classList.remove('on')">关闭</button></div>`
  }catch(e){mb.innerHTML='<p>导出失败: '+esc(e.message)+'</p><div class="ma"><button class="btn btn-cancel" onclick="md.classList.remove(\'on\')">关闭</button></div>'}
}

function openPreviewById(id){
  var p=null;
  for(var gi=0;gi<GROUPS.length;gi++)for(var pi=0;pi<GROUPS[gi].length;pi++){
    if(GROUPS[gi][pi].id===id){p=GROUPS[gi][pi];break}
  }
  if(!p)return;
  document.getElementById('side').classList.remove('closed');
  document.getElementById('mainArea').classList.add('side-open');
  document.getElementById('pvHint').style.display='none';
  var wrap=document.getElementById('pvWrap');wrap.style.display='block';
  var img=document.getElementById('pvImg');
  var bar=document.getElementById('pvBar');
  var barInner=document.getElementById('pvBarInner');
  if(bar)bar.style.display='block';
  if(barInner)barInner.style.width='5%';
  img.style.display='none';
  img.setAttribute('data-photo-id', p.id);
  document.getElementById('pvName').textContent='loading... '+p.name;

  PV.scale=1;PV.tx=0;PV.ty=0;PV.wrap=wrap;PV.img=img;PV.el=document.getElementById('pvArea');
  wrap.style.transform='translate(0,0) scale(1)';
  document.getElementById('pvZoom').textContent='';

  // 第一步：加载快速预览（250px 缓存，秒出并立即显示）
  dlog('PV start id='+p.id+' name='+p.name);
  var areaEl=document.getElementById('pvArea');
  var rect=areaEl.getBoundingClientRect();
  dlog('pvArea rect: w='+rect.width+' h='+rect.height);
  img.onload=function(){
    dlog('quick onload: nw='+img.naturalWidth+' nh='+img.naturalHeight);
    if(barInner)barInner.style.width='50%';
    document.getElementById('pvName').textContent=p.name;
    fitPreview();
    img.style.display='block';
    dlog('quick shown, scale='+PV.scale.toFixed(3));
    if(DEBUG_MODE) drawDebug();
    // 第二步：后台加载高清图（独立 Image 对象，不打断低清显示）
    if(barInner)barInner.style.width='70%';
    var fullImg=new Image();
    fullImg.onload=function(){
      dlog('fullImg loaded: nw='+fullImg.naturalWidth+' nh='+fullImg.naturalHeight);
      var r=areaEl.getBoundingClientRect();
      var panelW=r.width, panelH=r.height;
      var newW=fullImg.naturalWidth, newH=fullImg.naturalHeight;
      var oldW=img.naturalWidth||1, oldH=img.naturalHeight||1;
      // 保持低清图的视觉缩放比例：新缩放 = 旧缩放 × (旧宽/新高)
      var s=PV.scale*oldW/newW;
      // 屏幕中心点在低清图上的相对坐标（0..1）
      var relX=((panelW/2-PV.tx)/PV.scale)/oldW;
      var relY=((panelH/2-PV.ty)/PV.scale)/oldH;
      // 高清图上同一位置
      var cx=relX*newW, cy=relY*newH;
      // 新平移：让高清图的 (cx,cy) 落在屏幕中心
      var tx=panelW/2-cx*s, ty=panelH/2-cy*s;
      dlog('seamless: oldW='+oldW+' newW='+newW+' oldScale='+PV.scale.toFixed(3)+' newScale='+s.toFixed(3)+' tx='+tx.toFixed(0)+' ty='+ty.toFixed(0));
      // 替换 img.onload 为高清处理器，再设 src（浏览器缓存命中，秒切）
      img.onload=function(){
        dlog('full swapped in: nw='+img.naturalWidth);
        if(barInner)barInner.style.width='100%';
        setTimeout(function(){if(bar)bar.style.display='none';},400);
        PV.scale=s;PV.tx=tx;PV.ty=ty;
        applyPV();
        img.style.display='block';
        if(DEBUG_MODE) drawDebug();
      };
      img.src=fullImg.src;
    };
    fullImg.onerror=function(){dlog('!! fullImg onerror')};
    var fullSrc='/preview?id='+p.id+'&t='+Date.now();
    dlog('loading fullImg: '+fullSrc.substring(0,50));
    fullImg.src=fullSrc;
  };
  img.onerror=function(){
    dlog('!! quick onerror');
    document.getElementById('pvName').textContent='failed: '+p.name;
    img.style.display='none'
  };
  var quickSrc='/preview_quick?id='+p.id+'&t='+Date.now();
  dlog('setting img.src to quick: '+quickSrc.substring(0,50));
  img.src=quickSrc;
}

function fitPreview(){
  const wrap=PV.wrap,el=PV.el;if(!wrap||!el)return;
  const r=el.getBoundingClientRect();
  const iw=PV.img.naturalWidth||PV.img.width,ih=PV.img.naturalHeight||PV.img.height;
  if(!iw||!ih)return;
  const sx=r.width/iw,sy=r.height/ih;
  PV.scale=Math.min(sx,sy);PV.tx=0;PV.ty=0;
  applyPV();PV.img.style.display='block'
}

var DEBUG_MODE = false;
var LOG_MODE = false;
function toggleLog(){
  LOG_MODE = !LOG_MODE;
  var lp=document.getElementById('logPanel');
  lp.classList.toggle('on',LOG_MODE);
  const chk=document.getElementById('logChk');
  if(chk)chk.textContent = LOG_MODE ? '✓' : '';
}
function dlog(msg){
  var ts=new Date().toLocaleTimeString('en-US',{hour12:false})+'.'+String(Date.now()%1000).padStart(3,'0');
  var line='['+ts+'] '+msg;
  if(typeof console!=='undefined') console.log(line);
  if(LOG_MODE){
    var lp=document.getElementById('logPanel');
    if(lp){var d=document.createElement('div');d.className='le';d.textContent=line;lp.appendChild(d);lp.scrollTop=lp.scrollHeight}
  }
  // 同步落盘到服务端 pv_debug.log（fire-and-forget）
  try{fetch('/api/log?msg='+encodeURIComponent(msg))}catch(e){}
}
function toggleDebug(){
  DEBUG_MODE = !DEBUG_MODE;
  const chk=document.getElementById('debugChk');
  if(chk)chk.textContent = DEBUG_MODE ? '✓' : '';
  document.getElementById('pvCanvas').style.display = DEBUG_MODE ? 'block' : 'none';
  if(DEBUG_MODE && PV.img) drawDebug();
}
async function drawDebug(){
  var id = PV.img.getAttribute('data-photo-id');
  if(!id) return;
  try{
    var r = await fetch('/api/debug_photo?id=' + id);
    var d = await r.json();
    if(!d.ok) return;
    var canvas = document.getElementById('pvCanvas');
    canvas.width = d.w; canvas.height = d.h;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // 人脸框
    for(var fi = 0; fi < d.faces.length; fi++){
      var pts = d.faces[fi];
      ctx.strokeStyle = '#00ff00'; ctx.lineWidth = 2;
      // 找脸的范围
      var minX = 1, maxX = 0, minY = 1, maxY = 0;
      for(var pi = 0; pi < pts.length; pi++){
        if(pts[pi].x < minX) minX = pts[pi].x;
        if(pts[pi].x > maxX) maxX = pts[pi].x;
        if(pts[pi].y < minY) minY = pts[pi].y;
        if(pts[pi].y > maxY) maxY = pts[pi].y;
      }
      ctx.strokeRect(minX * d.w, minY * d.h, (maxX-minX) * d.w, (maxY-minY) * d.h);
      ctx.fillStyle = '#00ff00'; ctx.font = '12px sans-serif';
      ctx.fillText('Face ' + (fi+1), minX * d.w, minY * d.h - 4);
    }
    // 人体骨架 (PoseLandmarker 连接)
    var connections = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,23],[12,24],[11,13],[13,15],[15,17],[17,19],[19,21],[15,21],[12,14],[14,16],[16,18],[18,20],[16,20],[23,24],[23,25],[25,27],[27,29],[29,31],[27,31],[24,26],[26,28],[28,30],[30,32],[28,32]];
    for(var pi = 0; pi < d.poses.length; pi++){
      var pts = d.poses[pi];
      for(var ci = 0; ci < connections.length; ci++){
        var p1 = pts[connections[ci][0]], p2 = pts[connections[ci][1]];
        if(!p1 || !p2) continue;
        ctx.strokeStyle = '#ff4444'; ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(p1.x * d.w, p1.y * d.h);
        ctx.lineTo(p2.x * d.w, p2.y * d.h);
        ctx.stroke();
      }
      // 画关键点
      for(var pi2 = 0; pi2 < pts.length; pi2++){
        ctx.fillStyle = '#ff0'; ctx.beginPath();
        ctx.arc(pts[pi2].x * d.w, pts[pi2].y * d.h, 3, 0, Math.PI*2);
        ctx.fill();
      }
    }
  }catch(e){}
}

function closePreview(){
  document.getElementById('side').classList.add('closed');
  document.getElementById('mainArea').classList.remove('side-open');
  document.getElementById('pvHint').style.display='flex';
  document.getElementById('pvWrap').style.display='none';
  if(PV.img)PV.img.src=''
}

function applyPV(){
  PV.wrap.style.transform=`translate(${PV.tx}px,${PV.ty}px) scale(${PV.scale})`;
  document.getElementById('pvZoom').textContent=Math.round(PV.scale*100)+'%'
}

document.addEventListener('DOMContentLoaded',function(){
  const el=document.getElementById('pvArea');
  el.addEventListener('wheel',function(e){
    if(!PV.wrap||PV.wrap.style.display==='none')return;
    e.preventDefault();
    const r=el.getBoundingClientRect();
    const mx=e.clientX-r.left,my=e.clientY-r.top;
    const step=e.deltaY>0?0.88:1/0.88;
    const ns=Math.min(20,Math.max(0.15,PV.scale*step));
    const ix=(mx-PV.tx)/PV.scale,iy=(my-PV.ty)/PV.scale;
    PV.tx=mx-ix*ns;PV.ty=my-iy*ns;PV.scale=ns;
    applyPV()
  },{passive:false});
  el.addEventListener('mousedown',function(e){
    if(!PV.wrap||PV.wrap.style.display==='none')return;
    PV.drag=true;PV.dsx=e.clientX;PV.dsy=e.clientY;PV.dtx=PV.tx;PV.dty=PV.ty;
    el.style.cursor='grabbing'
  })
});
document.addEventListener('mousemove',function(e){
  if(!PV.drag)return;
  PV.tx=PV.dtx+(e.clientX-PV.dsx);PV.ty=PV.dty+(e.clientY-PV.dsy);
  applyPV()
});
document.addEventListener('mouseup',function(){
  PV.drag=false;
  if(PV.el)PV.el.style.cursor='grab'
});

let DEL_OPTS={};

function updDelBtn(){
  try{
    const mode=document.getElementById('delMode').value;
    let types=[];
    if(document.getElementById('tJPG').checked)types.push('JPG');
    if(document.getElementById('tHIF').checked)types.push('HIF');
    if(document.getElementById('tRAW').checked)types.push('ARW');
    let counts={JPG:0,HIF:0,ARW:0}, total=0;
    for(let g of GROUPS)for(let p of g){
      const match=(mode==='unselected'&&!p.keep)||(mode==='selected'&&p.keep);
      if(!match)continue;
      if(types.includes('ARW')&&p.has_raw){counts.ARW++;total++}
      if(types.includes(p.fmt)){counts[p.fmt]++;total++}
    }
    const btn=document.getElementById('delBtn');
    btn.disabled=(total===0);
    btn.textContent=total>0?'删除 '+total+' 个':'删除 (0)';
    let arwCount=0;
    for(let g of GROUPS)for(let p of g){if(p.keep && p.has_raw)arwCount++}
    const expBtn=document.getElementById('exportBtn');
    expBtn.disabled=(arwCount===0);
    expBtn.textContent=arwCount>0?`导出ARW · ${arwCount}`:'导出ARW';
    const expMi=document.getElementById('exportMi');
    if(expMi)expMi.disabled=(arwCount===0);
    const expMiCnt=document.getElementById('exportMiCnt');
    if(expMiCnt)expMiCnt.textContent=arwCount>0?String(arwCount):'';
    let parts=[];
    if(counts.JPG)parts.push('JPG '+counts.JPG);
    if(counts.HIF)parts.push('HIF '+counts.HIF);
    if(counts.ARW)parts.push('ARW '+counts.ARW);
    document.getElementById('delPreview').textContent=parts.length?'将删除: '+parts.join(' + '):'点击照片标记后执行删除';
  }catch(e){
    document.getElementById('delPreview').textContent='错误: '+e.message;
  }
}

async function execDel(){
  try{
    DEL_OPTS.mode=document.getElementById('delMode').value;
    DEL_OPTS.types=[];
    if(document.getElementById('tJPG').checked)DEL_OPTS.types.push('JPG');
    if(document.getElementById('tHIF').checked)DEL_OPTS.types.push('HIF');
    if(document.getElementById('tRAW').checked)DEL_OPTS.types.push('ARW');
  const md=document.getElementById('md'),mb=document.getElementById('mb');
  md.classList.add('on');
  mb.innerHTML='<div class="spin"></div><p style="text-align:center;color:#999">计算中…</p>';
  const r=await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmed:false,mode:DEL_OPTS.mode,types:DEL_OPTS.types})});
  const d=await r.json();
  if(!d.to_delete_count){mb.innerHTML='<p>没有需要删除的文件。</p><div class="ma"><button class="btn btn-cancel" onclick="md.classList.remove(\'on\')">关闭</button></div>';return}
  let fl=d.to_delete.map(f=>'<div>'+esc(f)+'</div>').join('');
  if(d.to_delete.length<d.to_delete_count)fl+=`<div style="color:#555">…还有${d.to_delete_count-d.to_delete.length}个文件</div>`;
  mb.innerHTML=`<h2 style="color:#e74c3c">即将删除 ${d.to_delete_count} 个文件</h2>
    <p>保留 ${d.kept_count} 张: ${esc(d.kept.join(', '))}</p>
    <div class="flist">${fl}</div>
    <p style="font-size:12px;color:#e74c3c">⚠ 此操作不可撤销！</p>
    <div class="ma">
      <button class="btn btn-cancel" onclick="md.classList.remove('on')">取消</button>
      <button class="btn del" onclick="execDelConfirm()">确认删除</button>
    </div>`
  }catch(e){
    document.getElementById('delPreview').textContent='执行错误: '+e.message;
    document.getElementById('md').classList.remove('on');
  }
}

async function execDelConfirm(){
  const mb=document.getElementById('mb');
  mb.innerHTML='<div class="spin"></div><p style="text-align:center;color:#999">正在删除…</p>';
  const r=await fetch('/api/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmed:true,mode:DEL_OPTS.mode,types:DEL_OPTS.types})});
  const d=await r.json();
  mb.innerHTML=`<h2 style="color:#27ae60">完成</h2>
    <p>删除了 ${d.deleted} 个文件${d.errors?`，${d.errors}个失败`:''}</p>
    <div class="ma"><button class="btn refresh" onclick="location.reload()">刷新页面</button></div>`;
  document.getElementById('delBtn').disabled=true
}

function esc(s){const d=document.createElement('div');d.appendChild(document.createTextNode(s));return d.innerHTML}

async function hotReload(e){
  var btn=document.getElementById('reloadBtn');
  var shift=e&&e.shiftKey;
  btn.disabled=true;btn.textContent=shift?'🔄 重启中…':'🔄 重载中…';
  try{
    var r=await fetch('/api/reload'+(shift?'?restart=1':''));
    var d=await r.json();
    if(d.ok){
      if(d.restarting){
        btn.textContent='🔄 重启中…等待服务恢复';
        // 轮询等待服务恢复
        var tries=0;
        var iv=setInterval(function(){
          tries++;
          fetch('/api/status').then(function(r){return r.json()}).then(function(s){
            clearInterval(iv);btn.textContent='🔄 已重启!';btn.disabled=false;
            setTimeout(function(){location.reload()},300)
          }).catch(function(){
            if(tries>30){clearInterval(iv);btn.textContent='🔄 超时';btn.disabled=false}
          })
        },1000)
      }else{
        btn.textContent='🔄 已更新!';
        setTimeout(function(){location.reload()},300)
      }
    }else{
      btn.textContent='🔄 失败';btn.disabled=false;
      alert('热更新失败: '+(d.error||'未知错误'))
    }
  }catch(ex){
    btn.textContent='🔄 失败';btn.disabled=false;
    alert('热更新请求失败: '+ex.message)
  }
}

// ── 菜单栏交互 ──────────────────────────
function toggleMenu(titleEl){
  const menu=titleEl.closest('.menu');
  const wasOpen=menu.classList.contains('open');
  closeMenus();
  if(!wasOpen)menu.classList.add('open');
}
function closeMenus(){document.querySelectorAll('.menu.open').forEach(m=>m.classList.remove('open'))}
document.addEventListener('click',function(e){
  const menu=e.target.closest('.menu');
  if(menu && e.target.closest('.mi')){
    menu.classList.remove('open');   // 点击菜单项后收起
  }else if(!menu){
    closeMenus();                    // 点击菜单外收起
  }
});
document.addEventListener('keydown',function(e){
  if((e.ctrlKey||e.metaKey)&&(e.key==='o'||e.key==='O')){e.preventDefault();focusDir()}
  if(e.key==='Escape')closeMenus();
});
function focusDir(){
  const inp=document.getElementById('dirInput');
  if(inp){inp.focus();inp.select()}
}
function aboutInfo(){
  const md=document.getElementById('md'),mb=document.getElementById('mb');
  md.classList.add('on');
  mb.innerHTML=`<h2>📷 连拍挑选</h2>
    <p style="font-size:13px">轻量级连拍照片挑选工具。</p>
    <p style="font-size:12px;color:#888">· 左键照片切换保留（绿框）<br>· 右键照片打开大图预览<br>· 顶部菜单栏分类管理 评分/筛选/标注 等设置<br>· 工具栏切换删除范围，执行删除 / 导出 ARW</p>
    <div class="ma"><button class="btn btn-cancel" onclick="md.classList.remove('on')">关闭</button></div>`
}

load()
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════

def main():
    global groups, current_dir, scan_root
    ap = argparse.ArgumentParser(description='连拍照片挑选工具')
    ap.add_argument('directory', nargs='?', default='', help='照片目录（留空则在界面中输入）')
    ap.add_argument('--port', type=int, default=0, help='端口（0=自动）')
    args = ap.parse_args()

    if args.directory:
        scan_root = args.directory
        if _is_album_root(Path(args.directory)):
            # 相册根目录：不扫照片，只建索引
            current_dir = str(Path(args.directory).resolve())
            groups = []
            print(f"=> 相册根目录: {current_dir}")
        else:
            groups = scan(args.directory)
            current_dir = str(Path(args.directory).resolve())
            rebuild_photo_list()
            if not groups:
                print("没有需要处理的照片。")
                return

    host = '127.0.0.1'
    port = args.port or 0
    srv = ThreadingHTTPServer((host, port), Handler)
    port = srv.server_address[1]
    url = f'http://{host}:{port}'

    print(f"=> 服务器已启动: {url}")
    if not _load_roots():
        print("=> 首次启动：浏览器中会引导添加相册根目录（如 D:/0_camera）")
        print("   已配置的相册根保存在 ~/.tgphoto/roots.json，可随时在界面中修改")
    else:
        print("=> 在浏览器中打开，首页默认显示全部相册；顶部可输入路径扫描单个目录")
    print(f"   按 Ctrl+C 停止服务\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == '__main__':
    main()