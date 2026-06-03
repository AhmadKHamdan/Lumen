"""Minimal camera probe — no YOLO, no navigation. Just: can OpenCV read live
frames right now? Uses MSMF (required for MediaFoundation/MIPI cams like the
HP 5MP) with an init delay and cold-open retries. Reports brightness and saves
the frame to cam_probe.jpg.

Run:  python scripts/cam_probe.py
"""
import time
import cv2


def open_cam(index=0, attempts=4):
    for attempt in range(attempts):
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap.isOpened():
            time.sleep(1.5)  # let the pipeline start
            for _ in range(10):
                ok, f = cap.read()
                if ok and f is not None and float(f.mean()) > 5:
                    print(f"opened on attempt {attempt + 1}")
                    return cap, f
                time.sleep(0.1)
        cap.release()
        time.sleep(3.0)  # MF device needs a few seconds to fully release
    return None, None


cap, last = open_cam()
if cap is None:
    print("RESULT: camera black/blocked in THIS session")
    raise SystemExit(1)

maxmean = float(last.mean())
for i in range(20):
    ok, f = cap.read()
    if ok and f is not None:
        m = float(f.mean())
        maxmean = max(maxmean, m)
        last = f
    time.sleep(0.1)

print("MAX brightness:", round(maxmean, 1))
cv2.imwrite("cam_probe.jpg", last)
print("RESULT: CAMERA WORKS (saved cam_probe.jpg)")
cap.release()
