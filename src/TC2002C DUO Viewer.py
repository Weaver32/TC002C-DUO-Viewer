# -*- coding: utf-8 -*-
#Author: Weaver32. Github: https://github.com/Weaver32/Weaver32

import av
import numpy as np
import cv2
import platform
import traceback


W, H = 256, 392
FPS = "25"
devTemp = -1


if platform.system() == "Windows":
    #For Windows machines:
    cFormat = "dshow"
    DEVICE = "video=UVC Camera"
elif platform.system() == "Linux":
    #For Linux machines:
    DEVICE = "/dev/video3" #could be video1 if you don't have another webcam installed...
    cFormat = "V4L2"
else:
    print("OS not supported.")
    raise SystemExit()

container = None
try:
    container = av.open(
        DEVICE,
        format=cFormat,
        options={
            "video_size": f"{W}x{H}",
            "pixel_format": "yuyv422",
            "framerate": FPS,
            "rtbufsize": "100M",
        },
    )
except:
    traceback.print_exc()
    print("TC002C DUO not found.\n Under Linux, check which /dev/video(x) is the correct one.\n\n")
    raise SystemExit
stream = container.streams.video[0]

def decode_raw16_top(top_bytes):
    Y0 = top_bytes[:, 0::4].astype(np.uint16)
    U0 = top_bytes[:, 1::4].astype(np.uint16)
    Y1 = top_bytes[:, 2::4].astype(np.uint16)
    V0 = top_bytes[:, 3::4].astype(np.uint16)

    raw16 = np.empty((top_bytes.shape[0], W), dtype=np.uint16)
    raw16[:, 0::2] = (U0 << 8) | Y0
    raw16[:, 1::2] = (V0 << 8) | Y1
    return raw16

def percentile_scale(img, p_lo=2.0, p_hi=98.0):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)
    return (img * 255).astype(np.uint8)

def center_roi(a, margin=24):
    return a[margin:-margin, margin:-margin]

def manual_scale_to_u8(tempC, tmin, tmax):
    if tmax <= tmin:
        tmax = tmin + 0.1
    x = (tempC - tmin) / (tmax - tmin)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)

def draw_label_block(img, lines, org=(12, 28), line_h=22):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 1
    x, y = org

    widths = [cv2.getTextSize(s, font, scale, thick)[0][0] for s in lines]
    w = max(widths) + 16 if widths else 120
    h = len(lines) * line_h + 12

    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - 8, y - 18),
        (x - 8 + w, y - 18 + h),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    yy = y
    for s in lines:
        cv2.putText(img, s, (x, yy), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        yy += line_h

def draw_crosshair(img):
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2
    cv2.drawMarker(
        img,
        (cx, cy),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        24,
        1,
        cv2.LINE_AA,
    )

def get_available_colormaps():
    """
    Build a stable list of OpenCV colormaps available in this build.
    Returns: list of (pretty_name, code_int), ordered by code_int.
    """
    pairs = []
    for k in dir(cv2):
        if k.startswith("COLORMAP_"):
            v = getattr(cv2, k, None)
            if isinstance(v, int):
                pairs.append((k, v))

    # Deduplicate by code (some builds may alias names)
    by_code = {}
    for k, v in pairs:
        by_code.setdefault(v, k)

    # Sort by numeric code for stable slider behavior
    out = []
    for code in sorted(by_code.keys()):
        name = by_code[code].replace("COLORMAP_", "").replace("_", " ").title()
        out.append((name, code))
    return out

# UI
CTRL_WIN = "Thermal Controls"
THERM_WIN = "TC002CDuo Thermal image viewer"
cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)

cv2.createTrackbar("AutoScale", CTRL_WIN, 1, 1, lambda v: None)
cv2.createTrackbar("ScaleMin_x10", CTRL_WIN, 200, 1000, lambda v: None)
cv2.createTrackbar("ScaleMax_x10", CTRL_WIN, 400, 1000, lambda v: None)

# Colormap slider (defaults to Inferno if available)
COLORMAPS = get_available_colormaps()

cv2.createTrackbar("Colormap", CTRL_WIN, 0, len(COLORMAPS) - 1, lambda v: None)
#Set default colormap to INFERNO (because it looks cool)
cv2.setTrackbarPos("Colormap", CTRL_WIN, 14)

while True:
    for packet in container.demux(stream):
        for fr in packet.decode():
            b = bytes(fr.planes[0])
            ls = fr.planes[0].line_size
            yuyv = np.frombuffer(b, np.uint8).reshape(H, ls)[:, : W * 2]

            top = yuyv[0:192, :]          # absolute temperature payload
            bottom = yuyv[196:387, :]     # grayscale relative image (unused here)
            metadata = yuyv[192:196, :60] # metadata, containing the device temperature

            devTemp = ((int(metadata[0, 3]) << 8) | int(metadata[0, 2])) / 44.0-3 #Calculate device temperature in °C (scaling factors by own measurements)

            raw16 = decode_raw16_top(top)
            tempC = raw16.astype(np.float32) / 32.0 - 171 +devTemp*1.1 #Calculate and absolute temperatures and compansate for device temperature (scaling factors by own measurements

            # ROI stats
            troi = center_roi(tempC)
            roi_min = float(troi.min())
            roi_max = float(troi.max())

            # Center temperature
            hh, ww = tempC.shape
            center_temp = float(tempC[hh // 2, ww // 2])

            # Scaling
            auto = cv2.getTrackbarPos("AutoScale", CTRL_WIN) == 1
            if auto:
                view = percentile_scale(tempC, 1, 99)
                scale_text = "Scale: AUTO"
            else:
                smin = cv2.getTrackbarPos("ScaleMin_x10", CTRL_WIN) / 10.0
                smax = cv2.getTrackbarPos("ScaleMax_x10", CTRL_WIN) / 10.0
                view = manual_scale_to_u8(tempC, smin, smax)
                scale_text = f"Scale: {smin:.1f} to {smax:.1f} C"

            # Colormap selection
            cmap_idx = cv2.getTrackbarPos("Colormap", CTRL_WIN)
            cmap_idx = int(np.clip(cmap_idx, 0, len(COLORMAPS) - 1))
            cmap_name, cmap_code = COLORMAPS[cmap_idx]
            
            #Rotate the picture so that the cable of the TC002CDUO points downwards and scale the view to 384x512
            view2 = cv2.applyColorMap(view, cmap_code)
            nView = cv2.rotate(view2, cv2.ROTATE_90_COUNTERCLOCKWISE)
            nView = cv2.resize(nView, (384, 512), interpolation=cv2.INTER_NEAREST)

            draw_label_block(
                nView,
                [
                    f"Center: {center_temp:.2f} C",
                    f"Min: {roi_min:.2f} C",
                    f"Max: {roi_max:.2f} C",
                    scale_text,
                    f"Colormap: {cmap_name}",
                ],
            )

            draw_crosshair(nView)

            cv2.imshow(THERM_WIN, nView)

            # Press Escape to close the software
            if cv2.waitKey(1) == 27:
                container.close()
                cv2.destroyAllWindows()
                raise SystemExit

            # OpenCV doesn't provide an onClose event, so poll window visibility, in case user closed the window
            if (cv2.getWindowProperty(THERM_WIN, cv2.WND_PROP_VISIBLE) < 1 or
                cv2.getWindowProperty(CTRL_WIN, cv2.WND_PROP_VISIBLE) < 1):
                container.close()
                cv2.destroyAllWindows()
                raise SystemExit
        break