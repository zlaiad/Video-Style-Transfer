# 🎨 Video Style Transfer – CLI Version  
**Style Transfer for Videos using VGG19 + RAFT Optical Flow (Server-Friendly, No GUI)**

一个基于 **VGG19 风格迁移 + RAFT 光流时序一致性 + HSV 色彩增强 + 锐化处理** 的视频风格迁移工具。  
完全命令行（CLI）版本，适合服务器环境、SSH 执行、自动化脚本与批量视频处理。

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
  pip install -r requirements.txt
```

### 2️⃣ Basic Usage
```bash
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4
```

---

## 🧩 Features

- 🎨 **VGG19 Style Transfer**（逐帧优化，每帧约 400 次迭代）
- 🌊 **RAFT Optical Flow**（高精度时序一致性）
- ✨ **HSV Color Enhancement**
- 🔪 **Unsharp Mask Edge Sharpening**
- 📏 **Two Resolution Modes**：256×256 fixed 或 original 全分辨率
- ⚡ GPU 加速支持（自动检测）

---

## 🎛 Command-line Arguments

| 参数 | 简写 | 必需 | 描述 | 默认值 |
|------|------|------|------|---------|
| `--video` | `-v` | ✔ | 输入视频路径 | - |
| `--style` | `-s` | ✔ | 风格图像路径 | - |
| `--output` | `-o` | ✔ | 输出视频路径 | - |
| `--resolution` | — | ✖ | `fixed`（256×256）或 `original` | `fixed` |
| `--raft` | — | ✖ | RAFT 模型大小：`small` / `large` | `small` |
| `--device` | — | ✖ | `auto` / `cuda` / `cpu` | `auto` |
| `--quiet` | `-q` | ✖ | 静默模式（仅输出错误） | False |
| `--verbose` | `-V` | ✖ | 输出详细日志 | False |

---

## 📖 Usage Examples

### 🌈 Basic stylization
```bash
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4
```

### 🚀 Use RAFT-large
```bash
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4 --raft large
```

### 📏 Output in original resolution
```bash
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4 --resolution original
```

### 🔍 Verbose mode
```bash
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4 --verbose
```

### 🤫 Quiet mode
```bash
  python cv_off.py --video input.mp4 --style style.jpg --output output.mp4 --quiet
```

---

## 🧠 How It Works

### 🔹 1. VGG19-based Style Transfer  
每一帧通过多种 Loss（Content、Style、Color、Edge、TV）进行优化，得到稳定的风格化结果。

### 🔹 2. RAFT Optical Flow  
用于计算相邻帧的精准光流，并将上一帧 warp 到当前帧进行时序一致性约束，有效减少闪烁。

### 🔹 3. Post-processing  
包括：

- HSV 饱和度/明度增强  
- HSV 跨帧 Hue 融合（保证色彩稳定）  
- Unsharp Mask 锐化（改善纹理与边缘）

最终输出的视频更清晰、鲜明、稳定。

---

## 📂 Project Structure

```
.
├── cv_off.py           # 主程序（CLI版本）
├── requirements.txt
└── README.md
```


---

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- opencv-python
- numpy, pillow, tqdm

适配服务器环境：  
✔ 不需要 GUI  
✔ 支持 SSH / 后台运行  

---

## 📜 License

```
MIT License
Copyright (c) 2025
```

---

## 👥 Authors（Group Members）

| 姓名  | Email |
|------|------------------------------|
| **Pu Tianyi**    | tpuac@connect.ust.hk |
| **Wang Xinyi**   | xwangla@connect.ust.hk |
| **Wu Xinze**     | xwudo@connect.ust.hk |
| **Lai Zhiyuan**  | zlaiad@connect.ust.hk |

感谢使用本项目！  
如需二次开发或商业用途，请遵循 MIT License。