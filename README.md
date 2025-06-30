# r/vision - Visual Effects Tool

<div align="center">

![r/vision Logo](https://img.shields.io/badge/r%2Fvision-v2.0.0-orange?style=for-the-badge&logo=python)

**🎬 Add Tracking Effects to Your Videos**

[📱 **See Demo on Instagram**](https://www.instagram.com/reel/DLhWQweyOcy/) | [📂 **Get the Code**](https://github.com/ferrary7/r-vision) | [🌐 **Try Live Demo**](#web-ui)

</div>

## What It Does

r/vision adds colorful tracking boxes around objects in your videos. Perfect for:

- **Social Media** - Eye-catching effects for TikTok, Instagram, YouTube

## Quick Start

### 🌐 Web UI (Easiest)

Try the live demo in your browser:

```bash
streamlit run app.py
```

- **Upload videos** up to 200MB
- **Real-time progress** with frame-by-frame updates
- **Instant preview** and download
- **Customizable settings** with live sliders

### 💻 Command Line

```bash
# Clone the repo
git clone https://github.com/ferrary7/r-vision.git
cd r-vision

# Install requirements
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python r_vision.py your_video.mp4

# Save with custom name
python r_vision.py input.mp4 --output enhanced.mp4

# More sensitive tracking (detects more objects)
python r_vision.py video.mp4 --yolo-confidence 0.1

# Less sensitive (only main objects)
python r_vision.py video.mp4 --yolo-confidence 0.7
```

## Features

- **🌐 Web Interface** - Streamlit UI with real-time progress and video info
- **Color-coded boxes** - Different colors for people, cars, electronics, etc.
- **Thin, modern design** - Clean HUD-style overlays
- **Live stats** - Shows confidence and FPS on each box
- **Real-time progress** - Frame-by-frame processing updates
- **Customizable** - Adjust sensitivity and colors

## Options

| Option | Description | Example |
|--------|-------------|---------|
| `--output` | Output filename | `--output cool_video.mp4` |
| `--yolo-confidence` | Sensitivity (0.1-1.0) | `--yolo-confidence 0.3` |
| `--no-yolo` | Only track people | `--no-yolo` |
| `--no-pose` | Only track objects | `--no-pose` |

## Requirements

- Python 3.8+
- 4GB+ RAM

## Colors

- 🟢 **Green**: People
- 🔴 **Red**: Vehicles (cars, bikes, planes)
- 🟣 **Purple**: Animals
- 🔵 **Blue**: Electronics
- 🟠 **Orange**: Food & drinks
- 🟡 **Yellow**: Furniture & objects

## For Developers

Want to customize? Edit the colors in `r_vision.py`:

```python
self.specific_colors = {
    'person': (0, 255, 255),    # Cyan for people
    'car': (255, 0, 255),       # Pink for cars
    'laptop': (0, 255, 0),      # Green for laptops
}
```

## License

MIT License - use it however you want!

---

<div align="center">
Made with ❤️ for content creators
</div>
