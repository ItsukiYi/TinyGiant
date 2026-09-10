# TinyGiant 连拍挑选工具

轻量级连拍照片挑选工具。本地启动一个 HTTP 服务，浏览器打开即可挑选、评分、删除、导出。
单文件实现（[picker.py](picker.py)），零配置，专为摄影连拍工作流设计。

> **注意**：仓库内的 `tgphoto/` 目录为早期废弃的 PySide6 桌面方案，**不再维护、不被任何入口引用**，请勿使用。实际运行的是本文件描述的 `picker.py`。

## 核心功能

| 功能 | 说明 |
|------|------|
| 📷 多格式支持 | JPEG、HEIF/HIF（含 HEIC）预览，同名 ARW 原始文件自动配对 |
| 🗂 相册管理 | 首次启动引导添加相册根目录（如 `D:\0_camera`），自动检索其中的照片文件夹，跨根统一展示，可随时增删/重扫 |
| ⭐ 自动评分 | 清晰度（中心加权 Laplacian）、曝光、对比度、MediaPipe 人脸/睁眼/姿态 |
| 🏷️ 智能推荐 | 每组连拍按分数排序，跳过相邻帧保证多样性，自动标星 2-3 张 |
| 🧹 连拍分组 | 按拍摄编号间隔 + EXIF 时间间隔自动成组 |
| 🗑️ 安全删除 | 删除走 `_tg_trash` 回收站（可恢复），仅清理 root 内空目录 |
| 📤 导出 | 选中照片的 ARW 原始文件复制到 `挑选_<时间戳>/` |
| 💾 状态持久化 | `keep/star/score` 落盘到 `_tg_cache/state.json`，重启不丢 |
| 🔒 本地安全 | 仅绑定 127.0.0.1，校验 Host 头防跨源，路径校验防穿越 |

## 技术栈

- **Python 3.12+**
- **http.server**（`ThreadingHTTPServer`）— 后端 + 内嵌 HTML 前端
- **Pillow + pillow-heif** — 图像解码 / 缩放 / EXIF / HIF 支持
- **OpenCV + NumPy** — 清晰度、曝光、对比度评分
- **MediaPipe** — 人脸关键点、姿态检测（可选，缺失则跳过人脸评分）
- **PowerShell** — 文件夹选择对话框（仅 Windows）

## 快速开始

```bash
# 1. 安装依赖
pip install pillow pillow-heif opencv-python mediapipe numpy

# 2. 启动（不传目录 = 相册管理模式，首选）
python picker.py --port 8888
# 或用启动脚本
启动挑选工具.bat

# 3. 浏览器打开终端打印的地址
#    首次启动会引导添加相册根目录（如 D:\0_camera），自动检索底下的照片文件夹；
#    之后首页默认显示全部相册，顶部「文件 → 相册管理…」可随时增删/重新扫描。
#    也可以直接指定单个拍摄目录扫描：
#    python picker.py "D:\0_camera\某次拍摄" --port 8888
```

无 cv2/mediapipe 时仍可用，仅跳过自动评分（仅做挑选/删除/导出）。

## 项目结构

```
TinyGiant/
├── picker.py            # 全部后端 + 内嵌前端（HTTP 服务、扫描、评分、删除、导出）
├── run_server.py         # 无界面环境启动包装（禁用 webbrowser 自动打开）
├── test_picker.py        # pytest 自动化测试（路径安全/持久化/删除/分组/缓存）
├── face_landmarker.task  # MediaPipe 人脸模型（可选）
├── pose_landmarker_lite.task  # MediaPipe 姿态模型（可选）
├── 启动挑选工具.bat       # Windows 启动脚本
├── README.md / 架构图.md   # 文档
└── tgphoto/              # ⚠️ 早期废弃方案，勿用
```

## HTTP 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | 主页面（HTML+JS） |
| GET  | `/api/data` | 当前分组与照片数据；`?view=albums` 强制相册首页 |
| GET  | `/api/status` | 后台评分进度 `{scoring_busy}` |
| GET  | `/thumb?id=N` | 250px 缩略图（读缓存） |
| GET  | `/preview?id=N` | 1600px 预览（读缓存） |
| GET  | `/preview_quick?id=N` | 快速 250px 预览 |
| GET  | `/api/debug_photo?id=N` | 人脸/姿态调试数据 |
| POST | `/api/browse` | 弹出文件夹选择框 |
| POST | `/api/scan` | 扫描目录（须绝对路径） |
| POST | `/api/toggle` | 切换保留并持久化 |
| POST | `/api/delete` | 回收站式删除（需 confirmed） |
| POST | `/api/export` | 导出选中 ARW |
| POST | `/api/albums` | 相册根管理：`{action:'add'\|'remove'\|'rescan', path}` |

## 交互

- **相册首页**：首次启动引导添加相册根目录；之后打开即显示跨根聚合的全部相册（按根分组，含封面/张数/人物），点击卡片进入挑选
- **左键**照片：切换保留（绿框）
- **右键**照片：侧栏大图预览（缩略图→高清渐进加载，滚轮缩放、拖动平移）
- 顶部目录栏输入绝对路径 → 打开；📂 浏览 弹窗选择
- 🗑️ 删除栏：选「选中/未选中」+ 勾选 JPG/HIF/ARW → 删除（移入回收站）
- 导出 ARW：把标记保留且含 ARW 的照片复制到子目录
- 「文件 → 相册管理…」：增删相册根目录、重新扫描

## 测试

```bash
pip install pytest
pytest test_picker.py -v
```

覆盖：连拍分组、缓存/回收站目录跳过、路径穿越拦截、状态读写往返与损坏容错、删除回收站化与禁删父目录、preview 缓存命中、相册根管理（增删/发现/聚合）与首次引导流程。

## 数据与安全说明

- 相册根目录配置保存在 `~/.tgphoto/roots.json`；全局人物/角色库优先保存在第一个相册根目录的 `_tg_global/`，未配置时回退 `~/.tgphoto/`。
- 所有缓存/状态/回收站均写入扫描目录下的 `_tg_cache/` 与 `_tg_trash/`，扫描自动跳过这两个目录。
- 仅绑定 `127.0.0.1`，且校验 `Host` 头为回环地址，拦截来自浏览器恶意页面的跨源请求。
- 图片路径参数经 `_safe_path` 校验必须位于扫描根子树（或已配置相册根）内，拒绝路径穿越。
- 删除不直接 `unlink`，而是 `shutil.move` 到 `_tg_trash/<时间戳>/`，误删可手动恢复。
