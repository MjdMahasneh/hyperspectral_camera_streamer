import os
import numpy as np
import cv2
import threading
import time
from HAIP_BlackIndustry import HAIP_BlackIndustry

# ==============================================================
# CONFIG
# ==============================================================
IP_ADDRESS = "192.168.7.1"
CAMERA_MODE = 0
FPS_TARGET = 450

EXPOSURE = 2200  # microseconds
GAIN = 0        # dB

SPATIAL_PIXELS = 640      # vertical (height)
SPECTRAL_BANDS = 213      # spectral channels
MAX_LINES = 1024          # horizontal (width)
BAND_AVG_WINDOW = 10      # spectral bands to average for grayscale

SAVE_HSI = True
SAVE_2D = True
SAVE_DIR = "output"




# ==============================================================
# CAPTURE THREAD (shared memory)
# ==============================================================
class CaptureThread(threading.Thread):
    """Captures lines from HAIP camera and writes into shared cube."""
    def __init__(self, cube, write_index, stop_event, lock):
        super().__init__(daemon=True) # Daemon thread (auto-exit on main thread end)
        self.cube = cube
        self.write_index = write_index
        self.stop_event = stop_event
        self.lock = lock
        self.camera = HAIP_BlackIndustry()

    def run(self):
        print(f"[INFO] Connecting to HAIP at {IP_ADDRESS} ...")
        cam = self.camera
        cam.init(IP_ADDRESS)
        cam.setMode(CAMERA_MODE)
        cam.setGain(GAIN)
        cam.setExposure(EXPOSURE)
        cam.setFPS(FPS_TARGET)
        cam.startCameraStream()
        print("[INFO] Streaming started (shared memory).")

        # Wait until the first valid frame is available
        while cam.getImage() is None and not self.stop_event.is_set():
            time.sleep(0.05)
        print("[INFO] Camera stream initialized and returning frames.")

        try:
            while not self.stop_event.is_set():
                frame = cam.getImage()  # (640, 213)
                if frame is None:
                    time.sleep(0.001)
                    continue

                frame = frame.astype(np.float32)
                with self.lock:
                    idx = self.write_index[0] % MAX_LINES # get current index with wrap-around
                    self.cube[:, :, idx] = frame
                    self.write_index[0] += 1

                # Optional small sleep to reduce CPU load without affecting FPS (e.g, if camera is slower, to prevent over-polling the getImage() and wasting CPU cycles)
                # time.sleep(0.001)
        finally:
            # if anything crashes mid-capture, this ensures the stream closes cleanly and the camera doesn’t hang.
            cam.stopCameraStream()
            print("[INFO] Capture thread stopped.")


# ==============================================================
# MAIN PIPELINE
# ==============================================================
def main():
    cube = np.zeros((SPATIAL_PIXELS, SPECTRAL_BANDS, MAX_LINES), dtype=np.float32)
    write_index = [0]
    lock = threading.Lock()
    stop_event = threading.Event()

    capture_thread = CaptureThread(cube, write_index, stop_event, lock)
    capture_thread.start()

    print("[INFO] Press 's' to save or 'q' to quit.")
    t0 = time.time()

    try:
        while not stop_event.is_set():
            with lock:
                idx = write_index[0]

            if idx < MAX_LINES:
                # Wait until at least one full 2D frame is formed
                time.sleep(0.01)
                continue

            # 1. Take last MAX_LINES (1024) slices from cube
            cube_section = cube[:, :, (idx - MAX_LINES) % MAX_LINES : idx % MAX_LINES]
            # Handle wrap-around in ring buffer
            if cube_section.shape[2] != MAX_LINES:
                cube_section = np.concatenate(
                    (cube[:, :, (idx - MAX_LINES) % MAX_LINES:], cube[:, :, : idx % MAX_LINES]),
                    axis=2,
                )

            # 2. Average selected bands -> 2D grayscale
            mid = SPECTRAL_BANDS // 2
            bands = slice(mid - BAND_AVG_WINDOW // 2, mid + BAND_AVG_WINDOW // 2)
            view_2d = np.mean(cube_section[:, bands, :], axis=1)  # (640, 1024)

            # Normalize to 0–255 uint8
            view_2d = np.clip(view_2d / np.max(view_2d) * 255, 0, 255).astype(np.uint8)

            # Convert to BGR for display
            view_bgr = cv2.cvtColor(view_2d, cv2.COLOR_GRAY2BGR)


            # =========================================================
            # Display live 2D image
            # =========================================================
            cv2.imshow("Hyperspectral 2D Stream", view_bgr)
            key = cv2.waitKey(1) & 0xFF

            # =========================================================
            # Optional save
            # =========================================================
            if key == ord("s"):
                os.makedirs(SAVE_DIR, exist_ok=True)
                if SAVE_HSI:
                    np.save(f"{SAVE_DIR}/cube_{int(time.time())}.npy", cube)
                    print("[INFO] Saved HSI cube.")
                if SAVE_2D:
                    cv2.imwrite(f"{SAVE_DIR}/view2D_{int(time.time())}.png", view_2d)
                    print("[INFO] Saved 2D grayscale image.")

            if key == ord("q"):
                stop_event.set()
                break

    finally:
        # =========================================================
        # Shutdown
        # =========================================================
        stop_event.set() # place in finally to signal capture thread to stop cleanly in case of error (or normal exit)
        elapsed = time.time() - t0
        print(f"[INFO] Lines captured ≈ {write_index[0]} "
              f"({write_index[0]/elapsed:.1f} FPS)")
        capture_thread.join(timeout=2) # wait for capture thread to end
        cv2.destroyAllWindows()
        print("[INFO] Stream ended.")





if __name__ == "__main__":
    main()
