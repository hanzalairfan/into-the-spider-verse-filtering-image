"""
ITSV-style comic filter — final pipeline, ported from itsv.ipynb.

Uses your exact tuned parameters:
  - linework: medianBlur(7) -> adaptiveThreshold(blockSize=15, C=3) -> MORPH_CLOSE(2x2)
  - color:    bilateralFilter(d=12) x2 -> level-reduce quantize(levels=10)
  - halftone: 6px grid, threshold=150, max_radius=4
  - combine:  multiply blend (edges * halftone * color)

Run modes:
  python comic_filter.py photo.jpg           -> saves output/comic_result.png
  python comic_filter.py --webcam            -> live filtered webcam feed
"""

import sys
import cv2
import numpy as np


def apply_filter(frame,
                  blur_ksize=7, block_size=15, C=3,
                  bilateral_d=12, sigma_color=200, sigma_space=7, smooth_iters=2,
                  levels=10,
                  halftone_spacing=6, halftone_threshold=150, halftone_max_radius=4):
    """
    Takes one BGR frame in, returns one BGR comic-filtered frame out.
    Nothing here depends on anything outside this function -- safe to call
    repeatedly on a stream of different-sized frames.
    """

    # ---- 1. Linework ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mb = cv2.medianBlur(gray, blur_ksize)
    edges = cv2.adaptiveThreshold(
        mb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        block_size, C
    )
    kernel = np.ones((2, 2), np.uint8)
    clean_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # ---- 2. Color quantization ----
    smooth = frame.copy()
    for _ in range(smooth_iters):
        smooth = cv2.bilateralFilter(smooth, d=bilateral_d,
                                      sigmaColor=sigma_color, sigmaSpace=sigma_space)
    factor = 256 // levels
    quantized = ((smooth // factor) * factor).astype(np.uint8)

    # ---- 3. Halftone shading ----
    h, w = gray.shape
    blank_canvas = np.full((h, w), 255, dtype=np.uint8)
    for y in range(0, h, halftone_spacing):
        for x in range(0, w, halftone_spacing):
            brightness = gray[y, x]
            if brightness < halftone_threshold:
                radius = max(1, int((1 - brightness / halftone_threshold) * halftone_max_radius))
                cv2.circle(blank_canvas, (x, y), radius, 0, -1)

    # ---- 4. Combine (multiply blend) ----
    edges_3 = cv2.cvtColor(clean_edges, cv2.COLOR_GRAY2BGR)
    halftone_3 = cv2.cvtColor(blank_canvas, cv2.COLOR_GRAY2BGR)

    float_edges = (edges_3 / 255).astype(np.float32)
    float_halftone = (halftone_3 / 255).astype(np.float32)
    float_quantized = (quantized / 255).astype(np.float32)

    multiplied = float_edges * float_halftone * float_quantized
    result = (multiplied * 255).astype(np.uint8)

    return result


def run_on_image(path, out_path=r"D:\python\New folder (3)\images.jpg"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    result = apply_filter(img)
    cv2.imwrite(out_path, result)
    print(f"Saved: {out_path}")


def run_on_webcam(camera_index=0, downscale_width=480):
    """
    Live webcam loop. downscale_width shrinks each frame before filtering
    (the halftone loop is a pure-Python nested loop, so full HD frames will
    lag badly -- shrinking keeps it responsive).
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera_index / permissions.")

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, stopping.")
            break

        # Downscale for speed, keep aspect ratio
        h, w = frame.shape[:2]
        if w > downscale_width:
            scale = downscale_width / w
            frame = cv2.resize(frame, (downscale_width, int(h * scale)))

        filtered = apply_filter(frame)
        cv2.imshow("ITSV Comic Filter", filtered)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--webcam":
        run_on_webcam()
    elif len(sys.argv) > 1:
        run_on_image(sys.argv[1])
    else:
        print("Usage:")
        print("  python comic_filter.py <image_path>   # run on a photo")
        print("  python comic_filter.py --webcam        # live webcam filter")