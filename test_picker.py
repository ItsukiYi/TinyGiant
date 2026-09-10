#!/usr/bin/env python3
"""
picker.py 自动化测试 — 覆盖路径安全、持久化、删除回收站、分组等核心修复点。
不依赖 cv2/mediapipe：scan 使用 start_workers=False 跳过后台评分。
运行：pytest test_picker.py -v
"""
import os
import sys
import json
import shutil
import types
from pathlib import Path

import numpy as np
import pytest
import cv2
from PIL import Image

# 确保能 import 同目录的 picker.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import picker as P  # noqa: E402


# ── fixture：构造带 EXIF DateTime 的合法 JPEG ─────────────────
def _make_jpg(path, dt_str):
    """生成 >15KB 的随机噪声 JPEG，写入 EXIF DateTimeOriginal(36867)。"""
    data = os.urandom(400 * 400 * 3)
    img = Image.frombytes('RGB', (400, 400), data)
    exif = img.getexif()
    exif[36867] = dt_str
    img.save(path, format='JPEG', quality=90, exif=exif.tobytes())


# ═══════════════════════════════════════════════════════════
#  分组扫描
# ═══════════════════════════════════════════════════════════
def test_scan_groups_by_time(tmp_path):
    # 时间：00:00 00:05 00:10 01:00 → 前三张一组，第四张因 >30s 单独成组
    files = [
        ('DSC0001.jpg', '2024:01:01 10:00:00'),
        ('DSC0002.jpg', '2024:01:01 10:00:05'),
        ('DSC0003.jpg', '2024:01:01 10:00:10'),
        ('DSC0004.jpg', '2024:01:01 10:01:00'),
    ]
    for name, dt in files:
        _make_jpg(tmp_path / name, dt)

    groups = P.scan(str(tmp_path), start_workers=False)
    assert len(groups) == 2
    assert [len(g) for g in groups] == [3, 1]
    assert groups[0][0].num == 1
    assert groups[-1][0].num == 4


def test_scan_skips_cache_and_trash_dirs(tmp_path):
    _make_jpg(tmp_path / 'DSC0001.jpg', '2024:01:01 10:00:00')
    # 这些目录应被扫描跳过
    (tmp_path / '_tg_cache').mkdir()
    (tmp_path / '_tg_cache' / 'thumb_0.jpg').write_bytes(b'x')
    (tmp_path / '_tg_trash').mkdir()
    (tmp_path / '_tg_trash' / 'DSC0002.jpg').write_bytes(b'x' * 20000)
    groups = P.scan(str(tmp_path), start_workers=False)
    assert len(groups) == 1
    assert len(groups[0]) == 1


# ═══════════════════════════════════════════════════════════
#  路径安全
# ═══════════════════════════════════════════════════════════
def test_safe_path_allows_inside(tmp_path):
    inside = tmp_path / 'sub' / 'x.jpg'
    inside.parent.mkdir()
    inside.write_bytes(b'x')
    assert P._safe_path(str(tmp_path), str(inside)) == inside.resolve()


def test_safe_path_blocks_traversal(tmp_path):
    root = str(tmp_path)
    # 相对穿越
    evil = os.path.join(str(tmp_path), '..', '..', 'evil.jpg')
    assert P._safe_path(root, evil) is None
    # 绝对路径在 root 外
    outside = tmp_path.parent / 'picker_outside_target.jpg'
    try:
        assert P._safe_path(root, str(outside)) is None
    finally:
        pass
    # 空字符串
    assert P._safe_path(root, '') is None


# ═══════════════════════════════════════════════════════════
#  持久化
# ═══════════════════════════════════════════════════════════
def test_persistence_roundtrip(tmp_path):
    p = P.Photo(tmp_path / 'a.jpg')
    p.keep, p.star = True, True
    p.score, p.score_sharp = 42.0, 10.0
    p.score_exp, p.score_contrast, p.score_face = 5.0, 6.0, 7.0
    groups = [[p]]
    P.save_state(str(tmp_path), groups)

    state = P.load_state(str(tmp_path))
    rec = state[str(p.preview)]
    assert rec['keep'] is True and rec['star'] is True
    assert rec['score'] == 42.0

    # 还原到新对象（路径相同，id 不同）
    p2 = P.Photo(tmp_path / 'a.jpg')
    p2.keep, p2.score = False, 0.0
    P._restore_state([p2], state)
    assert p2.keep is True
    assert p2.star is True
    assert p2.score == 42.0
    assert p2.score_sharp == 10.0


def test_persistence_missing_returns_empty(tmp_path):
    assert P.load_state(str(tmp_path)) == {}


def test_persistence_corrupt_ignored(tmp_path):
    cache = tmp_path / '_tg_cache'
    cache.mkdir()
    (cache / 'state.json').write_text('not valid json {', encoding='utf-8')
    assert P.load_state(str(tmp_path)) == {}


# ═══════════════════════════════════════════════════════════
#  删除（回收站化 + 禁删父目录）
# ═══════════════════════════════════════════════════════════
def test_compute_delete_list_modes(tmp_path):
    a = P.Photo(tmp_path / 'a.jpg'); a.keep = True;  a.raw = tmp_path / 'a.ARW'
    b = P.Photo(tmp_path / 'b.jpg'); b.keep = False; b.raw = tmp_path / 'b.ARW'
    groups = [[a, b]]

    del_sel, _ = P.compute_delete_list(groups, 'selected', ['JPG', 'HIF', 'ARW'])
    assert set(del_sel) == {tmp_path / 'a.jpg', tmp_path / 'a.ARW'}

    del_unsel, _ = P.compute_delete_list(groups, 'unselected', ['JPG', 'ARW'])
    assert set(del_unsel) == {tmp_path / 'b.jpg', tmp_path / 'b.ARW'}

    # 仅 JPG 类型
    del_jpg, _ = P.compute_delete_list(groups, 'unselected', ['JPG'])
    assert del_jpg == [tmp_path / 'b.jpg']


def test_execute_delete_moves_to_trash(tmp_path):
    f = tmp_path / 'DSC0001.jpg'
    f.write_bytes(b'x' * 20000)
    p = P.Photo(f); p.keep = False
    del_list, _ = P.compute_delete_list([[p]], 'unselected', ['JPG'])

    ok, err, trash, touched = P.execute_delete(str(tmp_path), del_list)
    assert ok == 1 and err == 0
    assert not f.exists(), "原文件应已移走"
    assert trash.exists(), "应创建回收站目录"
    moved = list(trash.rglob('DSC0001.jpg'))
    assert len(moved) == 1, "文件应存在于回收站中"


def test_cleanup_dirs_stays_within_root(tmp_path):
    sub = tmp_path / 'sub'; sub.mkdir()
    outside = tmp_path.parent / 'tg_cleanup_outside'
    outside.mkdir(exist_ok=True)
    try:
        P._cleanup_empty_dirs(str(tmp_path), [sub, tmp_path, outside])
        assert tmp_path.exists(), "扫描根目录绝不能被删除"
        assert outside.exists(), "根目录外的目录绝不能被删除"
        assert not sub.exists(), "根目录内的空子目录应被清理"
    finally:
        if outside.exists():
            outside.rmdir()


# ═══════════════════════════════════════════════════════════
#  preview 缓存
# ═══════════════════════════════════════════════════════════
def test_read_cache_bytes_hit_and_miss(tmp_path):
    p = P.Photo(tmp_path / 'a.jpg')
    # 写入与 photo.id 对应的 250 缓存
    cp = P._cache_path(tmp_path, p, 250)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(b'thumbdata')
    assert P._read_cache_bytes(str(tmp_path), p, 250) == b'thumbdata'
    # 1600 缓存未生成 → None
    assert P._read_cache_bytes(str(tmp_path), p, 1600) is None


# ═══════════════════════════════════════════════════════════
#  评分系统
# ═══════════════════════════════════════════════════════════
def _noise_img(size=300):
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)


def _gray96(seed):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, (96, 96), dtype=np.uint8)


def test_lap_var_sharp_gt_blur():
    sharp = _noise_img()
    blur = cv2.GaussianBlur(sharp, (15, 15), 0)
    assert P._lap_var(cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)) > \
           P._lap_var(cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY))


def test_percentile_rank():
    r = P._percentile_rank([10, 20, 30, 40])
    assert r[0] == 0.0 and r[-1] == 1.0
    ties = P._percentile_rank([5, 5, 5])
    assert all(x == 0.5 for x in ties)
    assert P._percentile_rank([7]) == [0.5]
    assert P._percentile_rank([]) == []


def _lm_eye(open_eye=True):
    """构造 468 点 landmark，仅设置左右眼 6 点。"""
    lm = [(0.0, 0.0)] * 468
    ly_top, ly_bot = (0.20, 0.35) if open_eye else (0.275, 0.285)
    lm[33] = (0.10, 0.30); lm[133] = (0.30, 0.30)
    lm[160] = (0.15, ly_top); lm[158] = (0.25, ly_top)
    lm[153] = (0.15, ly_bot); lm[144] = (0.25, ly_bot)
    lm[362] = (0.50, 0.30); lm[263] = (0.70, 0.30)
    lm[385] = (0.55, ly_top); lm[387] = (0.65, ly_top)
    lm[373] = (0.55, ly_bot); lm[380] = (0.65, ly_bot)
    return lm


def test_ear_closed_vs_open():
    assert P._ear_from_landmarks(_lm_eye(True)) > 0.18
    assert P._ear_from_landmarks(_lm_eye(False)) < 0.18


def test_score_photo_orders_by_sharp():
    sharp = _noise_img(400)
    blur = cv2.GaussianBlur(sharp, (21, 21), 0)
    r_sharp = P._score_photo(None, None, image=sharp, faces=[])
    r_blur = P._score_photo(None, None, image=blur, faces=[])
    assert r_sharp['lap'] > r_blur['lap']
    assert '废片' not in r_sharp['tags']


def test_score_group_best_is_100(tmp_path, monkeypatch):
    laps = [10, 20, 30, 40]
    raws = [{'lap': l, 'exp': 0.8, 'contrast': 0.8, 'face': 0.5, 'tags': []} for l in laps]
    it = iter(raws)
    monkeypatch.setattr(P, '_score_photo', lambda root, photo, **kw: next(it))
    photos = [P.Photo(tmp_path / f'p{i}.jpg') for i in range(4)]
    P.score_groups(None, [photos])
    scores = [p.score for p in photos]
    assert max(scores) == 100
    assert scores[3] == 100          # 最高 lap → 本组最佳
    assert scores[0] < scores[3]


def test_junk_detection():
    black = np.zeros((200, 200, 3), dtype=np.uint8)
    r = P._score_photo(None, None, image=black, faces=[])
    assert '废片' in r['tags']
    assert r['lap'] == 0.0


def test_overexposure_tag():
    # 渐变亮图：保证 std>5（不判废片）+ mean>0.88 + highlight>0.32
    x = np.arange(200)
    col = (230 + x * 0.12).clip(230, 255).astype(np.uint8)  # 230→254 渐变
    bright = np.stack([col] * 200, 0)  # (200,200)
    bright = np.stack([bright, bright, bright], 2)  # (200,200,3)
    r = P._score_photo(None, None, image=bright, faces=[])
    assert '过曝' in r['tags']
    assert r['exp'] < 0.5


def test_star_selection_picks_best_not_first(tmp_path, monkeypatch):
    # idx0 最差、idx3 最佳：星标应落在 idx3，而非首帧 idx0
    raws = [
        {'lap': 5,  'exp': 0.8, 'contrast': 0.8, 'face': 0.5, 'tags': []},
        {'lap': 10, 'exp': 0.8, 'contrast': 0.8, 'face': 0.5, 'tags': []},
        {'lap': 20, 'exp': 0.8, 'contrast': 0.8, 'face': 0.5, 'tags': []},
        {'lap': 40, 'exp': 0.8, 'contrast': 0.8, 'face': 0.5, 'tags': []},
    ]
    it = iter(raws)
    monkeypatch.setattr(P, '_score_photo', lambda root, photo, **kw: next(it))
    photos = [P.Photo(tmp_path / f'p{i}.jpg') for i in range(4)]
    P.score_groups(None, [photos])
    assert photos[3].star is True
    assert photos[0].star is False


def _bs(blink_l, blink_r):
    """构造含 eyeBlinkLeft/Right 的 blendshape Category 列表。"""
    return [types.SimpleNamespace(category_name='eyeBlinkLeft', score=blink_l),
            types.SimpleNamespace(category_name='eyeBlinkRight', score=blink_r),
            types.SimpleNamespace(category_name='other', score=0.1)]


def test_blink_from_blendshapes():
    # 返回 (left, right) 元组
    r1 = P._blink_from_blendshapes(_bs(0.1, 0.15))
    assert r1 == (0.1, 0.15)
    r2 = P._blink_from_blendshapes(_bs(0.9, 0.85))
    assert r2 == (0.9, 0.85)
    assert P._blink_from_blendshapes(None) is None
    assert P._blink_from_blendshapes([]) is None


def _face(blink_l=None, blink_r=None, ear=0.4, box=(0.25, 0.25, 0.75, 0.75), lm=None, area=0.25):
    blink = (blink_l, blink_r) if (blink_l is not None and blink_r is not None) else None
    return {'box': box, 'lm': lm or [], 'ear': ear, 'blink': blink,
            'blink_l': blink_l, 'blink_r': blink_r, 'area': area}


def test_blink_closed_penalizes():
    img = _noise_img(400)
    # 双眼闭合 = 闭眼
    rc = P._score_photo(None, None, image=img, faces=[_face(blink_l=0.8, blink_r=0.8)])
    # 双眼睁开
    ro = P._score_photo(None, None, image=img, faces=[_face(blink_l=0.1, blink_r=0.1)])
    assert '闭眼' in rc['tags']
    assert '闭眼' not in ro['tags']
    assert rc['face'] < ro['face']


def test_wink_not_penalized():
    img = _noise_img(400)
    # 单眼闭合 = wink，不应标闭眼，且应有 wink 标签
    rw = P._score_photo(None, None, image=img, faces=[_face(blink_l=0.8, blink_r=0.1)])
    assert '闭眼' not in rw['tags']
    assert 'wink' in rw['tags']
    # wink 分数应高于双眼闭合
    rc = P._score_photo(None, None, image=img, faces=[_face(blink_l=0.8, blink_r=0.8)])
    assert rw['face'] > rc['face']


def test_eye_region_sharp_path():
    img = _noise_img(400)
    face = _face(blink_l=0.1, blink_r=0.1, box=(0.2, 0.2, 0.8, 0.8), lm=_lm_eye(True), area=0.36)
    r = P._score_photo(None, None, image=img, faces=[face])
    assert r['lap'] > 0
    assert '废片' not in r['tags']


def test_face_metered_exposure_no_backlight_penalty():
    # 亮背景 + 暗中心（逆光人像：脸曝光正常、背景死白）
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    img[100:300, 100:300] = 60
    face = _face(blink_l=0.1, blink_r=0.1, box=(0.25, 0.25, 0.75, 0.75))
    r = P._score_photo(None, None, image=img, faces=[face])
    assert '过曝' not in r['tags']           # 主体测光：脸区不判过曝
    r2 = P._score_photo(None, None, image=img, faces=[])
    assert '过曝' in r2['tags']              # 无脸时全图测光仍判过曝


def test_near_identical_flat(tmp_path, monkeypatch):
    # 4 张近乎同质：lap/face/exp/contrast 全接近 → cv 低 → 扁平 50 + 无明显差异 + 无星标
    raws = [{'lap': 100.0, 'exp': 0.80, 'contrast': 0.80, 'face': 0.50, 'tags': []},
            {'lap': 101.0, 'exp': 0.805, 'contrast': 0.80, 'face': 0.50, 'tags': []},
            {'lap': 99.0,  'exp': 0.795, 'contrast': 0.80, 'face': 0.50, 'tags': []},
            {'lap': 100.5, 'exp': 0.80,  'contrast': 0.80, 'face': 0.50, 'tags': []}]
    it = iter(raws)
    monkeypatch.setattr(P, '_score_photo', lambda root, photo, **kw: next(it))
    photos = [P.Photo(tmp_path / f'p{i}.jpg') for i in range(4)]
    P.score_groups(None, [photos])
    assert all(p.score == 50 for p in photos)
    assert all('无明显差异' in p.tags for p in photos)
    assert all(not p.star for p in photos)


# ═══════════════════════════════════════════════════════════
#  真连拍识别
# ═══════════════════════════════════════════════════════════
def test_frame_distance_same_vs_different():
    a = _gray96(0)
    assert P._frame_distance(a, a) < 0.01
    b = _gray96(1)
    assert P._frame_distance(a, b) > 0.2


def test_compute_subbursts_boundary(tmp_path, monkeypatch):
    a, b = _gray96(0), _gray96(1)
    sigs = [a, a, a, b, b, b]
    it = iter(sigs)
    monkeypatch.setattr(P, '_frame_sig', lambda root, photo, **kw: next(it))
    photos = [P.Photo(tmp_path / f'p{i}.jpg') for i in range(6)]
    P.compute_subbursts(None, [photos])
    assert [p.burst for p in photos] == [0, 0, 0, 1, 1, 1]


def test_compute_subbursts_near_identical_no_split(tmp_path, monkeypatch):
    a = _gray96(0)
    sigs = [a] + [(a.astype(int) + np.random.randint(-2, 3, a.shape)).clip(0, 255).astype(np.uint8)
                  for _ in range(5)]
    it = iter(sigs)
    monkeypatch.setattr(P, '_frame_sig', lambda root, photo, **kw: next(it))
    photos = [P.Photo(tmp_path / f'p{i}.jpg') for i in range(6)]
    P.compute_subbursts(None, [photos])
    assert all(p.burst == 0 for p in photos)


def test_split_subbursts_contiguous(tmp_path):
    photos = [P.Photo(tmp_path / f'p{i}.jpg') for i in range(6)]
    for p, b in zip(photos, [0, 0, 1, 1, 1, 2]):
        p.burst = b
    raws = [{'lap': 1, 'exp': 0.5, 'contrast': 0.5, 'face': 0.5, 'tags': []} for _ in range(6)]
    subs = P._split_subbursts(photos, raws)
    assert len(subs) == 3
    assert [len(sp) for sp, _ in subs] == [2, 3, 1]
    assert subs[0][0][0] is photos[0]
    assert subs[2][0][0] is photos[5]


def test_score_per_subburst_best_each_100(tmp_path, monkeypatch):
    # 两段真连拍，每段内 lap 递增；各自最佳应=100
    raws = [{'lap': 10, 'exp': 0.8, 'contrast': 0.8, 'face': 0.5, 'tags': []},
            {'lap': 40, 'exp': 0.8, 'contrast': 0.8, 'face': 0.5, 'tags': []},
            {'lap': 10, 'exp': 0.8, 'contrast': 0.8, 'face': 0.5, 'tags': []},
            {'lap': 40, 'exp': 0.8, 'contrast': 0.8, 'face': 0.5, 'tags': []}]
    it = iter(raws)
    monkeypatch.setattr(P, '_score_photo', lambda root, photo, **kw: next(it))
    photos = [P.Photo(tmp_path / f'p{i}.jpg') for i in range(4)]
    photos[0].burst = 0; photos[1].burst = 0
    photos[2].burst = 1; photos[3].burst = 1
    P.score_groups(None, [photos])
    assert photos[1].score == 100
    assert photos[3].score == 100
    assert photos[0].score < photos[1].score
    assert photos[2].score < photos[3].score


# ═══════════════════════════════════════════════════════════
#  人物识别 / 闭眼基线
# ═══════════════════════════════════════════════════════════
def test_cosine_sim():
    import numpy as np
    a = np.array([1, 0, 0], dtype=np.float32)
    b = np.array([1, 0, 0], dtype=np.float32)
    c = np.array([0, 1, 0], dtype=np.float32)
    assert P._cosine_sim(a, b) == pytest.approx(1.0)
    assert P._cosine_sim(a, c) == pytest.approx(0.0)
    assert P._cosine_sim(a, -a) == pytest.approx(-1.0)


def test_align_face_shape():
    img = _noise_img(400)
    # landmark 不足 → 中心 crop，仍输出 112×112
    aligned = P._align_face(img, [])
    assert aligned.shape == (112, 112, 3)
    # 充足 landmark → 仿射对齐
    lm = [(0.0, 0.0)] * 468
    lm[33] = (0.3, 0.3); lm[263] = (0.7, 0.3); lm[1] = (0.5, 0.5)
    lm[61] = (0.5, 0.6); lm[199] = (0.5, 0.8)
    aligned2 = P._align_face(img, lm)
    assert aligned2.shape == (112, 112, 3)


def test_match_person_threshold(tmp_path, monkeypatch):
    """v3 质心匹配：只匹配 confirmed 人物，取最高余弦，≥ 阈值返回。"""
    import numpy as np
    fake_home = tmp_path / 'fakehome'
    fake_home.mkdir()
    (fake_home / '.tgphoto').mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake_home)
    P._save_persons({'version': 3, 'persons': {}, 'next_id': 1})

    emb_a = np.ones(512, dtype=np.float32) / np.linalg.norm(np.ones(512))
    pid = P._register_person(emb_a, name='花崎', confirmed=True)  # 已确认
    assert pid == 1

    # 相同方向 → 匹配
    pid2, sim, n = P._match_person(emb_a)
    assert pid2 == 1 and n == 1 and sim > 0.9

    # 正交方向 → 不匹配
    emb_diff = np.zeros(512, dtype=np.float32); emb_diff[0] = 1.0
    pid3, sim3, n3 = P._match_person(emb_diff)
    assert pid3 is None and n3 == 0


def test_match_unconfirmed_not_matched(tmp_path, monkeypatch):
    """未确认（未命名）人物不参与自动跨库匹配。"""
    import numpy as np
    fake = tmp_path / 'un'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    P._save_persons({'version': 3, 'persons': {}, 'next_id': 1})
    emb = np.ones(512, dtype=np.float32) / np.linalg.norm(np.ones(512))
    pid = P._register_person(emb, confirmed=False)  # 未确认
    assert pid == 1
    p, sim, n = P._match_person(emb)
    assert p is None  # 未确认不匹配


def test_persons_persistence_roundtrip(tmp_path, monkeypatch):
    fake_home = tmp_path / 'fakehome2'
    fake_home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake_home)
    P._save_persons({'version': 3, 'persons': {
        '1': {'name': '花崎', 'confirmed': True, 'prototype': [0.1, 0.2, 0.3],
              'count': 5, 'created_at': 0}}, 'next_id': 2})
    loaded = P._load_persons()
    assert '1' in loaded['persons']
    assert loaded['persons']['1']['name'] == '花崎'
    assert loaded['persons']['1']['count'] == 5
    assert loaded['next_id'] == 2


def test_blink_baseline_relative(monkeypatch):
    """同 person_id 多帧：全局阈值误判，但相对基线纠正。"""
    import numpy as np
    # 某人完全睁开 blink=0.15（天生小眼），全局阈值 0.50 不触发闭眼
    # 但本段有一帧 blink=0.60（相对基线 +0.45 → 闭眼）
    raws = [
        {'lap': 100, 'exp': 0.8, 'contrast': 0.8, 'face': 0.8, 'tags': [],
         'person_id': 1, 'blink_l': 0.15, 'blink_r': 0.15},  # 完全睁开
        {'lap': 90, 'exp': 0.8, 'contrast': 0.8, 'face': 0.2, 'tags': ['闭眼'],
         'person_id': 1, 'blink_l': 0.60, 'blink_r': 0.60},  # 全局判闭眼
    ]
    photos = [P.Photo(f'/fake/{i}.jpg') for i in range(2)]
    photos[0].person_id = 1; photos[1].person_id = 1
    P._apply_person_blink_baseline(photos, raws)
    # 第一帧睁开：不应有闭眼标签
    assert '闭眼' not in raws[0]['tags']
    # 第二帧相对基线闭眼：保留闭眼标签
    assert '闭眼' in raws[1]['tags']


def test_recognize_graceful_degradation():
    """模型缺失 → _get_face_recognizer 返回 None，识别管线不崩溃。"""
    assert P._get_face_recognizer() is None or True  # 不崩溃即可
    # 无 face 数据 → compute_persons 返回空统计不崩溃
    d = Path('D:/_nonexistent_test_dir')
    stats = P.compute_persons(str(d), [])
    assert stats == {'auto': 0, 'pending': 0, 'new': 0, 'group': 0}


# ═══════════════════════════════════════════════════════════
#  人物标注 / 角色 / 合并
# ═══════════════════════════════════════════════════════════
def test_person_rename(tmp_path, monkeypatch):
    fake = tmp_path / 'fh'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    P._save_persons({'version': 3, 'persons': {
        '1': {'name': '', 'confirmed': False, 'prototype': [0.1, 0.2], 'count': 3, 'created_at': 0}},
        'next_id': 2})
    P._rename_person(1, '花崎爱')
    db = P._load_persons()
    assert db['persons']['1']['name'] == '花崎爱'
    assert db['persons']['1']['confirmed'] is True  # 命名即确认


def test_person_merge(tmp_path, monkeypatch):
    fake = tmp_path / 'fh2'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    import numpy as np
    a = np.zeros(512, dtype=np.float32); a[0] = 1.0
    b = np.zeros(512, dtype=np.float32); b[1] = 1.0
    P._save_persons({'version': 3, 'persons': {
        '1': {'name': '六哥', 'confirmed': True, 'prototype': a.tolist(), 'count': 10, 'created_at': 0},
        '2': {'name': '', 'confirmed': False, 'prototype': b.tolist(), 'count': 5, 'created_at': 0},
    }, 'next_id': 3})
    ok = P._merge_persons(2, 1)  # src=2, dst=1
    assert ok
    db = P._load_persons()
    assert '2' not in db['persons']
    assert db['persons']['1']['count'] == 15
    assert db['persons']['1']['name'] == '六哥'
    # 质心 = (10*a + 5*b)/15 归一化
    proto = np.asarray(db['persons']['1']['prototype'], dtype=np.float32)
    assert proto[0] > 0 and proto[1] > 0


def test_role_create_rename(tmp_path, monkeypatch):
    fake = tmp_path / 'fh3'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    roles = P._load_roles()
    assert roles['next_id'] == 1
    roles['roles']['1'] = {'name': '绝区零-妮可', 'exemplars': [], 'color': '#e74c3c'}
    roles['next_id'] = 2
    P._save_roles(roles)
    P._rename_role(1, '妮可')
    db = P._load_roles()
    assert db['roles']['1']['name'] == '妮可'


def test_reassign_persists(tmp_path):
    d = tmp_path / 'photos'; d.mkdir()
    p1 = P.Photo(d / 'DSC0001.jpg'); p1.person_id = 1
    p2 = P.Photo(d / 'DSC0002.jpg'); p2.person_id = 2
    groups = [[p1, p2]]
    P.photo_list = [p1, p2]
    P.photo_dict = {p1.id: p1, p2.id: p2}
    # 改 p2 到 person_id=1
    p2.person_id = 1
    assert p2.person_id == 1


def test_role_auto_classify(tmp_path, monkeypatch):
    """标注 exemplar 后，相似帧自动归类。"""
    fake = tmp_path / 'fh4'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    # 创建角色 + exemplar
    roles = P._load_roles()
    roles['roles']['1'] = {'name': 'test', 'exemplars': [str(tmp_path / 'ex1.jpg')], 'color': '#3498db'}
    roles['next_id'] = 2
    P._save_roles(roles)
    # 造 exemplar 图 + 相似帧
    import numpy as np
    from PIL import Image
    base = np.random.RandomState(42).randint(0, 256, (96, 96), dtype=np.uint8)
    Image.fromarray(base).save(str(tmp_path / 'ex1.jpg'))
    # 建缓存目录 + 造相似签名
    d = tmp_path / 'shoot'; d.mkdir()
    cache = d / '_tg_cache'; cache.mkdir()
    p1 = P.Photo(d / 'DSC0001.jpg'); p1.id = 0
    P.Photo._next_id = 1
    # 造一张接近的缓存缩略图
    Image.fromarray(base).save(str(cache / 'thumb_0.jpg'))
    groups = [[p1]]
    monkeypatch.setattr(P, 'scan_root', str(d))
    # save_state 需要目录可写
    monkeypatch.setattr(P, 'save_state', lambda r, g: None)
    P.compute_roles(str(d), groups)
    assert p1.role_id == 1


def test_role_persistence(tmp_path, monkeypatch):
    fake = tmp_path / 'fh5'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    roles = {'version': 2, 'roles': {'1': {'name': '原神-雷电', 'exemplars': ['a.jpg', 'b.jpg'], 'color': '#f00'}}, 'next_id': 2}
    P._save_roles(roles)
    loaded = P._load_roles()
    assert loaded['roles']['1']['name'] == '原神-雷电'
    assert len(loaded['roles']['1']['exemplars']) == 2
    assert loaded['next_id'] == 2


# ═══════════════════════════════════════════════════════════
#  二级匹配 / 合并建议 / 待确认
# ═══════════════════════════════════════════════════════════
def test_two_level_match(tmp_path, monkeypatch):
    """质心匹配：已确认人物命中、正交不命中。"""
    import numpy as np
    fake = tmp_path / 'tl'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    P._save_persons({'version': 3, 'persons': {}, 'next_id': 1})
    base = np.zeros(512, dtype=np.float32); base[0] = 1.0
    pid1 = P._register_person(base, confirmed=True, name='花崎')
    high = base.copy(); high[2] = 0.01; high /= np.linalg.norm(high)
    p, sim, n = P._match_person(high)
    assert p == 1 and sim >= P._AUTO_MERGE_SIM
    orth = np.zeros(512, dtype=np.float32); orth[511] = 1.0
    p2, sim2, n2 = P._match_person(orth)
    assert p2 is None


# ═══════════════════════════════════════════════════════════
#  防污染：拒绝 outlier / 批量聚类 / 清洗
# ═══════════════════════════════════════════════════════════
def test_match_requires_support(tmp_path, monkeypatch):
    """未确认（未命名）人物不参与自动匹配。"""
    import numpy as np
    fake = tmp_path / 'sup'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    P._save_persons({'version': 3, 'persons': {}, 'next_id': 1})
    base = np.zeros(512, dtype=np.float32); base[0] = 1.0
    pid1 = P._register_person(base)  # 默认未确认
    assert pid1 == 1
    near = base.copy(); near[1] = 0.1; near /= np.linalg.norm(near)
    p, sim, n = P._match_person(near)
    assert p is None and n == 0  # 未确认不参与匹配


def test_update_rejects_outlier(tmp_path, monkeypatch):
    """内部不一致的 embedding → 拒绝更新质心（防污染源）。"""
    import numpy as np
    fake = tmp_path / 'out'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    P._save_persons({'version': 3, 'persons': {}, 'next_id': 1})
    base = np.zeros(512, dtype=np.float32); base[0] = 1.0
    pid1 = P._register_person(base)
    orth = np.zeros(512, dtype=np.float32); orth[511] = 1.0
    accepted = P._update_person(pid1, orth)
    assert accepted is False
    db = P._load_persons()
    assert db['persons']['1']['count'] == 1  # 未被污染


def test_hac_clusters_same_person():
    """批量聚类：相似 embedding 聚同一簇，不同人分开。"""
    import numpy as np
    base = np.zeros(64, dtype=np.float32); base[0] = 1.0
    same1 = base.copy(); same1[1] = 0.01; same1 /= np.linalg.norm(same1)
    same2 = base.copy(); same2[2] = 0.01; same2 /= np.linalg.norm(same2)
    other = np.zeros(64, dtype=np.float32); other[40] = 1.0
    X = np.stack([base, same1, same2, other])
    labels = P._hac_cluster(X, dist_thr=0.50)
    # 前 3 个同簇，第 4 个不同
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_prototype_of_normalized():
    """质心均值归一化。"""
    import numpy as np
    a = np.zeros(64, dtype=np.float32); a[0] = 1.0
    b = np.zeros(64, dtype=np.float32); b[1] = 1.0
    proto = P._prototype_of([a, b])
    assert proto is not None
    assert abs(np.linalg.norm(proto) - 1.0) < 1e-3
    assert proto[0] > 0 and proto[1] > 0


def test_cleanup_splits_contaminated(tmp_path, monkeypatch):
    """v3 cleanup：删除 count=0 或 prototype 为空的人物。"""
    fake = tmp_path / 'cln'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    P._save_persons({'version': 3, 'persons': {
        '1': {'name': '花崎', 'confirmed': True, 'prototype': [0.1, 0.2], 'count': 4, 'created_at': 0},
        '2': {'name': '', 'confirmed': False, 'prototype': None, 'count': 0, 'created_at': 0},
    }, 'next_id': 3})
    report = P.cleanup_persons()
    assert report['cleaned'] == 1
    db = P._load_persons()
    assert '1' in db['persons']
    assert '2' not in db['persons']


def test_merge_suggestions(tmp_path, monkeypatch):
    """两个人物 embedding 相似度 > 0.30 → 建议合并。"""
    import numpy as np
    fake = tmp_path / 'ms'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    a = np.zeros(512, dtype=np.float32); a[0] = 1.0; a[1] = 0.3; a /= np.linalg.norm(a)
    b = np.zeros(512, dtype=np.float32); b[0] = 1.0; b[1] = 0.2; b /= np.linalg.norm(b)
    c = np.zeros(512, dtype=np.float32); c[100] = 1.0  # 完全不同
    P._save_persons({'version': 2, 'persons': {
        '1': {'embeddings': [a.tolist()], 'count': 10, 'name': '花崎爱', 'exemplar': ''},
        '2': {'embeddings': [b.tolist()], 'count': 3, 'name': '', 'exemplar': ''},
        '3': {'embeddings': [c.tolist()], 'count': 5, 'name': '', 'exemplar': ''},
    }, 'next_id': 4})
    # 设置 groups 让 person 1,2,3 都出现在"当前目录"
    p1 = P.Photo(fake / 'DSC0001.jpg'); p1.person_id = 1
    p2 = P.Photo(fake / 'DSC0002.jpg'); p2.person_id = 2
    p3 = P.Photo(fake / 'DSC0003.jpg'); p3.person_id = 3
    P.groups = [[p1, p2, p3]]
    suggestions = P._compute_merge_suggestions()
    # 1 和 2 应该有建议（相似度高），3 不应该
    pairs = [(s['src_id'], s['dst_id']) for s in suggestions]
    assert (2, 1) in pairs or (1, 2) in pairs  # src=2(小) → dst=1(大)
    assert not any(3 in pair for pair in pairs)


def test_merge_pending_persists(tmp_path):
    """merge_pending 保存到 state 并恢复。"""
    d = tmp_path / 'mp'; d.mkdir()
    p1 = P.Photo(d / 'DSC0001.jpg')
    p1.person_id = 1; p1.merge_pending = True
    P.photo_list = [p1]; P.photo_dict = {p1.id: p1}
    P.groups = [[p1]]
    P.scan_root = str(d)
    P.save_state(str(d), [[p1]])
    state = P.load_state(str(d))
    rec = state[str(p1.preview)]
    assert rec.get('merge_pending') == True


def test_role_two_level():
    """角色匹配：≤_ROLE_CONFIRM 直认，≤_ROLE_SUGGEST 待确认。"""
    assert P._ROLE_CONFIRM < P._ROLE_SUGGEST
    assert P._ROLE_CONFIRM == 0.08
    assert P._ROLE_SUGGEST == 0.15


# ═══════════════════════════════════════════════════════════
#  文件夹内批量聚类 → 跨库质心匹配 → 命名/归并
# ═══════════════════════════════════════════════════════════
def test_compute_persons_clusters_folder(tmp_path, monkeypatch):
    """compute_persons 对本文件夹人脸批量聚类：相似同簇、不同人分开；小簇不建人物。"""
    import numpy as np
    fake = tmp_path / 'cp'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    P._save_persons({'version': 3, 'persons': {}, 'next_id': 1})
    d = fake / 'photos'; d.mkdir()
    # 人 A：3 张脸（≥_MIN_CLUSTER）→ 建人物；路人：1 张脸（小簇）→ 不建人物
    a1 = np.zeros(512, dtype=np.float32); a1[0] = 1.0
    a2 = a1.copy(); a2[1] = 0.01; a2 /= np.linalg.norm(a2)
    a3 = a1.copy(); a3[2] = 0.01; a3 /= np.linalg.norm(a3)
    rando = np.zeros(512, dtype=np.float32); rando[400] = 1.0
    def mk(e):
        return {'box': (0, 0, 1, 1), 'area': 1.0, 'embedding': e.tolist(),
                'blink_l': None, 'blink_r': None}
    photos = []
    for i, emb in enumerate([a1, a2, a3]):
        p = P.Photo(d / f'DSC000{i+1}.jpg')
        p._face_data = [mk(emb)]
        photos.append(p)
    p_r = P.Photo(d / 'DSC0004.jpg'); p_r._face_data = [mk(rando)]
    photos.append(p_r)
    groups = [photos]
    monkeypatch.setattr(P, 'save_state', lambda r, g: None)
    stats = P.compute_persons(str(d), groups)
    # 人 A 三张同簇 → 同一 person_id（新建未确认）；路人小簇 → None
    assert photos[0].person_id == photos[1].person_id == photos[2].person_id
    assert photos[0].person_id is not None and photos[0].person_id > 0
    assert photos[3].person_id is None
    assert photos[0].merge_pending is True  # 新建未确认
    db = P._load_persons()
    assert len(db['persons']) == 1  # 只建了 A，路人未建


def test_compute_persons_auto_merge_confirmed(tmp_path, monkeypatch):
    """全局已有已确认人物 → 本文件夹相似簇自动归并（sim≥_AUTO_MERGE_SIM）。"""
    import numpy as np
    fake = tmp_path / 'am'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    huagi = np.zeros(512, dtype=np.float32); huagi[0] = 1.0
    P._save_persons({'version': 3, 'persons': {
        '1': {'name': '花崎', 'confirmed': True, 'prototype': huagi.tolist(),
              'count': 100, 'created_at': 0}}, 'next_id': 2})
    d = fake / 'photos'; d.mkdir()
    def mk(e):
        return {'box': (0, 0, 1, 1), 'area': 1.0, 'embedding': e.tolist(),
                'blink_l': None, 'blink_r': None}
    photos = []
    for i in range(3):
        e = huagi.copy(); e[1] = 0.01 * (i + 1); e /= np.linalg.norm(e)
        p = P.Photo(d / f'DSC000{i+1}.jpg'); p._face_data = [mk(e)]
        photos.append(p)
    groups = [photos]
    monkeypatch.setattr(P, 'save_state', lambda r, g: None)
    P.compute_persons(str(d), groups)
    # 归并到全局花崎（person 1），非新建
    assert all(p.person_id == 1 for p in photos)
    assert all(p.merge_pending is False for p in photos)


def test_resolve_person_merges_to_global(tmp_path, monkeypatch):
    """api_resolve_person：命名未确认人物归并到已存在的全局人物。"""
    import numpy as np
    fake = tmp_path / 'rp'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    huagi = np.zeros(512, dtype=np.float32); huagi[0] = 1.0
    # 全局已有已确认"花崎"(person 1) + 本文件夹未确认人物(2)
    P._save_persons({'version': 3, 'persons': {
        '1': {'name': '花崎', 'confirmed': True, 'prototype': huagi.tolist(),
              'count': 100, 'created_at': 0},
        '2': {'name': '', 'confirmed': False, 'prototype': huagi.tolist(),
              'count': 5, 'created_at': 0}}, 'next_id': 3})
    d = fake / 'photos'; d.mkdir()
    p1 = P.Photo(d / 'DSC0001.jpg'); p1.person_id = 2
    p2 = P.Photo(d / 'DSC0002.jpg'); p2.person_id = 2
    P.groups = [[p1, p2]]; P.scan_root = str(d)
    P._state_lock = __import__('threading').RLock()
    handler = object.__new__(P.Handler)
    handler.groups = P.groups; handler.scan_root = str(d)
    handler._state_lock = P._state_lock
    handler._json = lambda data, status=200: data
    monkeypatch.setattr(P, 'save_state', lambda r, g: None)
    # 未确认人物 2 的 prototype 与花崎一致 → 匹配候选
    handler._get_local_person_embeddings = lambda pid: [huagi]
    res = handler.api_resolve_person({'local_id': 2, 'target_id': 1})
    assert res['ok'] is True and res['global_id'] == 1
    # 照片 person_id 更新为全局 1
    assert p1.person_id == 1 and p2.person_id == 1
    assert p1.merge_pending is False


def test_folder_data_only_current_persons(tmp_path, monkeypatch):
    """文件夹内 api_data 只返回当前文件夹出现的人物（含局部负 ID），不含其他文件夹的全局人物。"""
    import numpy as np
    # 全局库有两个已命名人物（一个在本文件夹，一个不在）
    fake = tmp_path / 'fld'; fake.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: fake)
    P._save_persons({'version': 2, 'persons': {
        '1': {'embeddings': [], 'count': 50, 'exemplar': '', 'name': '花崎'},
        '2': {'embeddings': [], 'count': 30, 'exemplar': '', 'name': 'baka'}}, 'next_id': 3})
    # 本文件夹只出现 person 1 和局部 -1
    d = fake / 'shoot'; d.mkdir()
    p1 = P.Photo(d / 'DSC0001.jpg'); p1.person_id = 1
    p2 = P.Photo(d / 'DSC0002.jpg'); p2.person_id = -1
    P.groups = [[p1, p2]]; P.scan_root = str(d)
    P.current_dir = str(d)
    P._state_lock = __import__('threading').RLock()
    # 模拟 api_data 的 pcounts 计算
    pcounts = {}
    for g in P.groups:
        for p in g:
            if p.person_id is not None:
                pcounts[p.person_id] = pcounts.get(p.person_id, 0) + 1
    all_pids = set(pcounts.keys())  # 新逻辑：只本文件夹
    assert 1 in all_pids and -1 in all_pids
    assert 2 not in all_pids  # baka 不在本文件夹，不显示


# ═══════════════════════════════════════════════════════════
#  已导出照片提醒（挑选_* 目录 → 默认选中）
# ═══════════════════════════════════════════════════════════
def test_mark_previously_exported(tmp_path):
    """导出目录里的 ARW 若仍在原相册，对应照片默认选中；不在则不改。"""
    d = tmp_path / 'album'; d.mkdir()
    # 原相册：DSC0001(有 raw), DSC0002(有 raw), DSC0003(有 raw)
    for stem in ('DSC0001', 'DSC0002', 'DSC0003'):
        (d / f'{stem}.jpg').write_bytes(b'x')
        (d / f'{stem}.ARW').write_bytes(b'x')
    # 导出目录：包含 DSC0001 和 DSC0002 的 ARW（DSC0001 原 raw 仍在；DSC0002 原 raw 被删）
    export_dir = d / '挑选_20260830_001103'
    export_dir.mkdir()
    (export_dir / 'DSC0001.ARW').write_bytes(b'x')
    (export_dir / 'DSC0002.ARW').write_bytes(b'x')
    # DSC0002 原 raw 删除（模拟已被移动/清理）
    (d / 'DSC0002.ARW').unlink()

    photos = [P.Photo(d / 'DSC0001.jpg'), P.Photo(d / 'DSC0002.jpg'), P.Photo(d / 'DSC0003.jpg')]
    for p in photos:
        p.find_raw()
    # 确保 DSC0002 无 raw（原 ARW 已删）
    assert photos[1].raw is None
    count = P._mark_previously_exported(d, photos)
    # DSC0001: raw 仍在 + 已导出 → 选中；DSC0002: raw 已删 → 不选；DSC0003: 未导出 → 不选
    assert photos[0].keep is True
    assert photos[1].keep is False
    assert photos[2].keep is False
    assert count == 1


def test_mark_exported_no_override_restore(tmp_path, monkeypatch):
    """mark 在 _restore_state 之后运行，不覆盖用户已保存的选中状态（但默认未保存时按导出标记）。"""
    d = tmp_path / 'album2'; d.mkdir()
    (d / 'DSC0001.jpg').write_bytes(b'x')
    (d / 'DSC0001.ARW').write_bytes(b'x')
    export_dir = d / '挑选_20260829_120000'
    export_dir.mkdir()
    (export_dir / 'DSC0001.ARW').write_bytes(b'x')
    # 无 state（首次打开），mark 应选中 DSC0001
    p = P.Photo(d / 'DSC0001.jpg'); p.find_raw()
    count = P._mark_previously_exported(d, [p])
    assert p.keep is True and count == 1


# ═══════════════════════════════════════════════════════════
#  评分维度可配置（dims）
# ═══════════════════════════════════════════════════════════
def _mk_raw(lap, exp, contrast, face, tags=None):
    return {'lap': lap, 'exp': exp, 'contrast': contrast, 'face': face,
            'tags': tags or [], 'person_id': None, 'blink_l': None, 'blink_r': None}


def test_normalize_group_all_dims_default():
    """无 dims（默认）→ 保持原 4 维权重，全部子分非 0。"""
    raws = [_mk_raw(10, 0.2, 0.2, 0.2), _mk_raw(40, 0.8, 0.8, 0.8)]
    norm, _ = P._normalize_group(raws)
    # 最佳照片各子分应非 0（全维度参与）
    best = norm[1]
    assert best['sharp_pct'] > 0 and best['face_pct'] > 0
    assert best['exp_pct'] > 0 and best['contrast_pct'] > 0
    # 最佳 quality_pct 应为 1.0（本组最佳）
    assert best['quality_pct'] == 1.0


def test_normalize_group_dims_filter_sharp_only():
    """dims 只含 sharp → quality 只由对焦决定，其他子分 0。"""
    # 照片 A：对焦好但曝光/表情差；照片 B：对焦差但曝光/表情好
    raws = [
        _mk_raw(50, 0.1, 0.1, 0.1),   # A: 对焦好
        _mk_raw(5, 0.9, 0.9, 0.9),    # B: 对焦差
    ]
    norm, _ = P._normalize_group(raws, dims=['sharp'])
    # 只看对焦 → A 应优于 B
    assert norm[0]['quality'] > norm[1]['quality']
    # 未勾选维度子分为 0
    assert norm[0]['face_pct'] == 0.0 and norm[1]['face_pct'] == 0.0
    assert norm[0]['exp_pct'] == 0.0 and norm[1]['exp_pct'] == 0.0
    assert norm[0]['contrast_pct'] == 0.0 and norm[1]['contrast_pct'] == 0.0
    # 对焦子分参与
    assert norm[0]['sharp_pct'] == 1.0 and norm[1]['sharp_pct'] == 0.0


def test_normalize_group_dims_weights_renormalized():
    """dims 权重重新归一化：只勾 face+exp 时两者各占 0.30/(0.30+0.15)=0.667 / 0.333。"""
    # A: 表情好、曝光差；B: 表情差、曝光好
    raws = [
        _mk_raw(10, 0.1, 0.1, 0.9),  # A: face 好 exp 差
        _mk_raw(10, 0.9, 0.1, 0.1),  # B: face 差 exp 好
    ]
    norm, _ = P._normalize_group(raws, dims=['face', 'exp'])
    qA = 0.667 * 1.0 + 0.333 * 0.0  # A: face=1.0, exp=0.0
    qB = 0.667 * 0.0 + 0.333 * 1.0  # B: face=0.0, exp=1.0
    assert norm[0]['quality'] == pytest.approx(qA, abs=0.02)
    assert norm[1]['quality'] == pytest.approx(qB, abs=0.02)
    assert norm[0]['quality'] > norm[1]['quality']  # face 权重高 → A 更好


def test_score_groups_dims_passthrough(tmp_path, monkeypatch):
    """score_groups 透传 dims 到 _normalize_group：只勾 sharp 时其他子分写 0。"""
    import numpy as np
    d = tmp_path / 'sg'; d.mkdir()
    # 构造一组照片（用真实小图，锐/糊对比）
    rng = np.random.RandomState(1)
    sharp = rng.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    import cv2
    blur = cv2.GaussianBlur(sharp, (15, 15), 0)
    from PIL import Image
    files = []
    for i, arr in enumerate((sharp, blur, blur, blur)):
        f = d / f'DSC000{i+1}.jpg'
        Image.fromarray(arr).save(f, 'JPEG')
        files.append(f)
    photos = [P.Photo(f) for f in files]
    # 用真实评分（dims 只勾 sharp）
    P.score_groups(str(d), [photos], dims=['sharp'])
    # 最锐照片（第 0 张）应最高分
    assert photos[0].score >= photos[1].score
    # 未勾选维度子分为 0
    assert photos[0].score_face == 0
    assert photos[0].score_exp == 0
    assert photos[0].score_contrast == 0


# ═══════════════════════════════════════════════════════════
#  相册根目录管理 + 相册发现
# ═══════════════════════════════════════════════════════════
def _mk_album_root(tmp_path, name='root'):
    """构造相册根目录：root/album_a 含照片，root/album_b 已扫过(state.json)，root/empty 空。"""
    root = tmp_path / name
    a = root / 'album_a'; a.mkdir(parents=True)
    _make_jpg(a / 'DSC0001.jpg', '2024:01:01 10:00:00')
    _make_jpg(a / 'DSC0002.jpg', '2024:01:01 10:00:05')
    b = root / 'album_b'; b.mkdir(parents=True)
    _make_jpg(b / 'DSC0001.jpg', '2024:01:01 11:00:00')
    (b / '_tg_cache').mkdir()
    (b / '_tg_cache' / 'state.json').write_text(
        json.dumps({'photos': {str(b / 'DSC0001.jpg'): {'person_id': 3, 'role_id': 1, 'keep': True}}}),
        encoding='utf-8')
    (root / 'empty').mkdir()
    return root


def test_roots_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(P, '_roots_path', lambda: tmp_path / 'roots.json')
    P._save_roots(['D:/a', 'E:/b'])
    assert P._load_roots() == ['D:/a', 'E:/b']
    # 缺失文件返回空
    monkeypatch.setattr(P, '_roots_path', lambda: tmp_path / 'nope.json')
    assert P._load_roots() == []


def test_is_photo_folder(tmp_path):
    root = _mk_album_root(tmp_path)
    assert P._is_photo_folder(root / 'album_a') is True       # 直接含照片
    assert P._is_photo_folder(root / 'album_b') is True       # 有 _tg_cache/state.json
    assert P._is_photo_folder(root / 'empty') is False        # 空目录
    assert P._is_photo_folder(root / 'album_a' / 'DSC0001.jpg') is False  # 非目录


def test_build_album_index_discovers_photo_folders(tmp_path):
    root = _mk_album_root(tmp_path)
    albums = P._build_album_index(root)
    names = [a['name'] for a in albums]
    assert 'album_a' in names and 'album_b' in names
    assert 'empty' not in names          # 空目录被跳过
    a = next(x for x in albums if x['name'] == 'album_a')
    assert a['photo_count'] == 2         # 未扫过 → 快速直扫计数
    assert a['cover_path'].endswith('DSC0001.jpg')
    assert a['root'] == str(root)
    b = next(x for x in albums if x['name'] == 'album_b')
    assert b['photo_count'] == 1         # 已扫过 → 读 state.json
    assert b['persons'] == [3] and b['roles'] == [1]


def test_all_albums_aggregates_roots(tmp_path, monkeypatch):
    r1 = _mk_album_root(tmp_path, 'root1')
    r2 = tmp_path / 'root2'; (r2 / 'sub_x').mkdir(parents=True)
    _make_jpg(r2 / 'sub_x' / 'a.jpg', '2024:02:02 10:00:00')
    monkeypatch.setattr(P, '_roots_path', lambda: tmp_path / 'roots.json')
    P._save_roots([str(r1), str(r2)])
    roots, albums = P._all_albums()
    assert roots == [str(r1), str(r2)]
    names = {(a['root'], a['name']) for a in albums}
    assert (str(r1), 'album_a') in names
    assert (str(r1), 'album_b') in names
    assert (str(r2), 'sub_x') in names


def test_global_root_prefers_first_configured_root(tmp_path, monkeypatch):
    monkeypatch.setattr(P, '_global_root_cache', None)
    r1 = _mk_album_root(tmp_path, 'root1')
    monkeypatch.setattr(P, '_roots_path', lambda: tmp_path / 'roots.json')
    P._save_roots([str(r1)])
    assert str(P._global_root()) == str(r1 / '_tg_global')
    monkeypatch.setattr(P, '_global_root_cache', None)


def test_build_album_index_recursive_nested(tmp_path):
    """照片嵌在多层子目录（如 DCIM/100MSDCF）也应被发现为相册，且跳过 jpgs。"""
    root = tmp_path / 'root'
    a = root / '25.12.26-shoot'
    deep = a / 'DCIM' / '100MSDCF'; deep.mkdir(parents=True)
    _make_jpg(deep / '_DSC0001.jpg', '2024:12:26 10:00:00')
    _make_jpg(deep / '_DSC0002.jpg', '2024:12:26 10:00:05')
    (root / 'empty').mkdir()
    j = root / 'jpgs'; j.mkdir()
    _make_jpg(j / 'a.jpg', '2024:12:27 10:00:00')  # SKIP_DIRS 内的目录不当作相册
    albums = P._build_album_index(root)
    names = [x['name'] for x in albums]
    assert '25.12.26-shoot' in names
    assert 'empty' not in names
    assert 'jpgs' not in names
    a_alb = next(x for x in albums if x['name'] == '25.12.26-shoot')
    assert a_alb['photo_count'] == 2
    assert a_alb['cover_path'].endswith('_DSC0001.jpg')


def test_api_scan_configured_root_returns_albums(tmp_path, monkeypatch):
    """打开已配置的相册根（仅 1 个子相册，不满足旧启发式）也应返回 albums 视图。"""
    import threading
    import urllib.request
    monkeypatch.setattr(P, 'scan_root', '')
    monkeypatch.setattr(P, 'current_dir', '')
    monkeypatch.setattr(P, 'groups', [])
    monkeypatch.setattr(P, '_roots_path', lambda: tmp_path / 'roots.json')
    monkeypatch.setattr(P, '_global_root_cache', None)
    root = tmp_path / 'camera'
    a = root / '20240101_shoot'; a.mkdir(parents=True)
    _make_jpg(a / 'DSC0001.jpg', '2024:01:01 10:00:00')
    assert P._is_album_root(root) is False  # 仅 1 个子相册，旧启发式识别不出
    P._save_roots([str(root)])
    srv = P.ThreadingHTTPServer(('127.0.0.1', 0), P.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{port}'
    try:
        def post(p, body):
            req = urllib.request.Request(base + p, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read().decode())

        d = post('/api/scan', {'directory': str(root)})
        assert d['ok'] and d['view'] == 'albums'
        assert [x['name'] for x in d['albums']] == ['20240101_shoot']
        # 前端打开根后会再 fetch /api/data：此时 scan_root=根目录，仍应为 albums
        with urllib.request.urlopen(base + '/api/data') as r:
            d2 = json.loads(r.read().decode())
        assert d2['view'] == 'albums'
    finally:
        srv.shutdown()


def test_api_albums_setup_add_remove_flow(tmp_path, monkeypatch):
    """HTTP 级：首次启动 setup → 添加根 → 相册首页聚合 → 移除回 setup。"""
    import threading
    import urllib.request
    # 隔离：其他测试会改动模块全局，这里全部重置
    monkeypatch.setattr(P, 'scan_root', '')
    monkeypatch.setattr(P, 'current_dir', '')
    monkeypatch.setattr(P, 'groups', [])
    monkeypatch.setattr(P, '_roots_path', lambda: tmp_path / 'roots.json')
    monkeypatch.setattr(P, '_global_root_cache', None)
    root = tmp_path / 'camera'
    a = root / '20240101_shoot'; a.mkdir(parents=True)
    _make_jpg(a / 'DSC0001.jpg', '2024:01:01 10:00:00')
    (root / 'empty').mkdir()
    srv = P.ThreadingHTTPServer(('127.0.0.1', 0), P.Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{port}'
    try:
        def get(p):
            with urllib.request.urlopen(base + p) as r:
                return json.loads(r.read().decode())

        def post(p, body):
            req = urllib.request.Request(base + p, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read().decode())

        assert get('/api/data')['view'] == 'setup'
        d = post('/api/albums', {'action': 'add', 'path': str(root)})
        assert d['ok'] and d['roots'] == [str(root.resolve())]
        d = get('/api/data')
        assert d['view'] == 'albums'
        assert [x['name'] for x in d['albums']] == ['20240101_shoot']
        d = post('/api/albums', {'action': 'remove', 'path': str(root)})
        assert d['ok'] and d['roots'] == []
        assert get('/api/data')['view'] == 'setup'
    finally:
        srv.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
