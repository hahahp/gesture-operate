# MediaPipe Gesture Playground

一个可直接调用电脑摄像头的 MediaPipe 手势识别最小项目，也顺手做成了一个小型“隔空涂鸦板”。

现在还包含一个更偏电影/VR 风格的 **Spatial Canvas**：通过食指定位、拇指与食指捏合来抓取屏幕中的空间卡片，并支持双手缩放和旋转。

## Spatial Canvas 体验版

```bash
cd /Users/ping/ping_ws/gesture-operate
.venv/bin/python spatial_demo.py
```

交互方式：

- 移动食指：移动空间指针
- 拇指与食指捏合：抓住卡片
- 保持捏合并移动：拖动卡片
- 两只手同时捏合：缩放、旋转卡片
- 轻捏后松开：切换卡片内容状态
- 快速移动后松开：带惯性抛出卡片
- `F`：切换全屏，`R`：重置场景，`Q` / `Esc`：退出

如果还没有给 Terminal 或 Codex 相机权限，可以先用鼠标预览视觉和交互；按住鼠标左键等价于捏合：

```bash
.venv/bin/python spatial_demo.py --mouse
```

全屏启动：

```bash
.venv/bin/python spatial_demo.py --fullscreen
```

这个 Demo 只操作自身窗口中的卡片，不会控制其他 App，因此不需要辅助功能或屏幕录制权限，只需要相机权限。后续可以在它之上增加一个需要用户明确授权的“系统鼠标模式”。

## 能做什么

- 实时识别双手、绘制 21 个手部关键点和骨架
- 显示手势名称、置信度、左右手与 FPS
- `Pointing_Up`（食指向上）：用食指尖隔空画线
- `Closed_Fist`（握拳）：清空画布
- `Victory`（V 手势）：切换画笔颜色
- `Open_Palm`（张开手掌）：暂停绘画
- `Q` / `Esc` 退出，`C` 清屏，空格换色

预训练模型还支持 `Thumb_Up`、`Thumb_Down` 和 `ILoveYou`，可以很容易继续绑定自己的玩法。

## 运行

项目已经使用本机 Python 3.9 创建了独立虚拟环境 `.venv`：

```bash
cd /Users/ping/ping_ws/gesture-operate
source .venv/bin/activate
python app.py
```

也可以不激活环境，直接运行：

```bash
.venv/bin/python app.py
```

只检查依赖和模型、不开摄像头：

```bash
.venv/bin/python app.py --check
```

只读取一帧来检查摄像头、不开预览窗口：

```bash
.venv/bin/python app.py --camera-check
```

如果默认摄像头不是编号 0：

```bash
.venv/bin/python app.py --camera 1
```

首次使用摄像头时，macOS 可能弹出权限请求。若没有弹出或无法打开，请到“系统设置 → 隐私与安全性 → 相机”，允许当前终端或 Codex 使用相机，然后重新运行。

## 项目结构

```text
.
├── .venv/                         # 项目独立 Python 环境
├── models/gesture_recognizer.task # Google 官方预训练模型
├── app.py                         # 摄像头、识别、骨架与涂鸦逻辑
├── spatial_demo.py                # 捏合、拖动、缩放、旋转空间卡片
├── requirements.txt               # Python 依赖
└── README.md
```

模型缺失时，`app.py` 会从 Google 官方地址自动下载一次；之后识别在本机执行，不需要持续联网。

## 调研结论

本项目使用当前推荐的 **MediaPipe Tasks Gesture Recognizer**，而不是已停止维护的 Legacy Hands API：

- [Python 安装指南](https://developers.google.com/edge/mediapipe/solutions/setup_python)：支持 Windows、macOS、Linux 及 Python 3.9+
- [Python 手势识别指南](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python)：摄像头场景使用 `LIVE_STREAM`，结果通过异步回调返回；OpenCV 可负责采集摄像头帧
- [Gesture Recognizer 概览与模型](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)：官方模型输出手势、左右手信息、图像坐标和世界坐标下的 21 个关键点

适合继续玩的方向：手势控制幻灯片、虚拟按钮/打地鼠、捏合调节音量、用自己采集的数据训练自定义手势。官方内置分类器主要识别静态手势；需要识别“挥手”等连续动作时，应在关键点时间序列之上增加状态机或时序模型。
