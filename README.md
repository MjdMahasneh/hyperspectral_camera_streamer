# Hyperspectral Pushbroom Streamer (Shared Memory, Real-Time 2D View)

A high-speed, low-latency **hyperspectral line-scanner streamer** implemented in Python. It captures real-time spectral lines from a pushbroom camera, stores them in shared memory, and builds a continuously updating 2D grayscale view for visualization or downstream processing.

---

## Features

- **Threaded acquisition and visualization** — continuous camera streaming and live display.
- **Zero-copy shared memory** — capture and visualization threads share the same 3D NumPy cube (no serialization overhead).
- **Ring buffer architecture** — fixed-size memory that overwrites old frames, avoiding delays or queue buildup.
- **Real-time 2D reconstruction** — spectral averaging of bands to produce grayscale visualization.
- **Optional saving** — export `.npy` cubes and `.png` 2D projections on keypress.

---

## Architecture Overview

```
[ Capture Thread ]                  [ Main Thread ]
------------------                  ----------------
- Reads camera lines                - Builds 2D grayscale view
- Writes into shared cube           - Displays live image
- Uses lock for safety              - Handles user input / saving
```

The camera capture never stops while the main thread visualizes the latest data slice.

---

## Key Concepts

| Concept | Description |
|----------|-------------|
| **Pushbroom capture** | Each frame is a single spectral line (e.g., 640 × 213). |
| **Shared memory** | Both threads operate directly on the same NumPy array. |
| **Locking** | Ensures safe concurrent read/write operations. |
| **Ring buffer** | Continuously overwrites oldest lines with newest ones. |
| **2D reconstruction** | Joins recent lines and averages spectral bands for grayscale view. |

---

## Installation

Clone the repository or install directly from GitHub:

```bash
git clone https://github.com/MjdMahasneh/hyperspectral_camera_streamer.git
cd hyperspectral_camera_streamer
```

---

## Requirements

```bash
pip install numpy opencv-python
```

You must also install your **camera's Python API or SDK**.  
For example, for HAIP Black Industry cameras:

```python
from HAIP_BlackIndustry import HAIP_BlackIndustry
```

Place the provided `HAIP_BlackIndustry.py` file (or your own camera API wrapper) in the same directory.

---

## Usage

1. Connect your hyperspectral camera (e.g., HAIP) via GigE or USB.
2. Adjust parameters in the configuration section:

```python
IP_ADDRESS = "192.168.7.1"
SPATIAL_PIXELS = 640
SPECTRAL_BANDS = 213
MAX_LINES = 512
BAND_AVG_WINDOW = 10
SAVE_DIR = "output"
```

3. Run the streamer:

```bash
python hyperspectral_streamer.py
```

Controls:
- **`s`** → Save current cube (`.npy`) and grayscale view (`.png`)
- **`q`** → Quit gracefully

---

## Output Structure

```
output/
 ├── cube_1698696000.npy       # 3D HSI cube (spatial × spectral × temporal)
 └── view2D_1698696000.png     # Grayscale projection (spatial × temporal)
```

---

## Example Integration (HAIP Camera)

HAIP pushbroom sensors stream ~450 FPS spectral lines via GigE.  
Example minimal integration:

```python
cam = HAIP_BlackIndustry()
cam.init("192.168.7.1")
cam.startCameraStream()
frame = cam.getImage()  # returns a (640, 213) spectral line
```

The streamer wraps this logic in a multi-threaded shared-memory pipeline, enabling near-zero latency visualization.








