"""
Title: Concreto en Juego
Authors: Milo Miranda , Saskia Benitez, Ida Osten, Luis Esteban Rodriguez
Year: 2025
Location: Amsterdam
Technique: Digital video art with code intervention and computer vision
Maintainer: Luis Esteban Rodriguez <rodriguezjluis0@gmail.com> <https://github.com/metalerk>
"""

import cv2
import numpy as np


# Define HSV Color Ranges
COLOR_RANGES = {
    "yellow": {
        "lower": np.array([20, 100, 100]),
        "upper": np.array([30, 255, 255])
    },
    "orange": {
        "lower": np.array([10, 100, 100]),
        "upper": np.array([20, 255, 255])
    },
    "green": {
        "lower": np.array([40, 40, 40]),
        "upper": np.array([80, 255, 255])
    },
    "neon_yellow": {
        "lower": np.array([10, 80, 80]),
        "upper": np.array([20, 255, 255])
    },
    "neon_green": {
        "lower": np.array([40, 150, 150]),
        "upper": np.array([85, 255, 255])
    },
    "lime_neon": {
        "lower": np.array([30, 80, 200]),
        "upper": np.array([40, 130, 255])
},
}

# Video Settings Structure: (video_name, [overlay_images], scale, [colors])
# args:
# (video_filename, [image1, image2, ...], scale_size, [tracking_colour1, tracking_colour2, ...])
video_settings = (
    ("01", ["F", "F", "F"], 1.5, ["green", "yellow", "lime_neon"]),
    ("02", ["A", "A"], 1.5, ["green", "lime_neon"]),
    ("03", ["B"], 1.5, ["orange"]),
    ("04", ["C"], 1.5, ["orange"]),
    ("05", ["D", "D"], 1.5, ["green", "lime_neon"]),
)


def preprocess_overlay(image_name, scale):
    """Load and scale overlay image + create masks."""
    overlay = cv2.imread(f"{image_name}.png")
    orig_h, orig_w = overlay.shape[:2]
    scaled_w = int(orig_w * scale)
    scaled_h = int(orig_h * scale)
    overlay_scaled = cv2.resize(overlay, (scaled_w, scaled_h))

    # Masks
    gray_overlay = cv2.cvtColor(overlay_scaled, cv2.COLOR_BGR2GRAY)
    _, overlay_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
    overlay_mask_inv = cv2.bitwise_not(overlay_mask)

    return overlay_scaled, overlay_mask, overlay_mask_inv


def apply_gamma_correction(image, gamma=1.0):
    """Optional gamma correction to fix exposure/brightness shifts."""
    look_up_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
    corrected = cv2.LUT(image, look_up_table)
    return corrected


def overlay_img_in_video(video_path, overlay_names, img_scale, color_names):
    print(f"🔄 Processing video: {video_path}")

    # Load Video
    cap = cv2.VideoCapture(f"{video_path}.MOV")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # MP4 H264-safe
    out = cv2.VideoWriter(f"rendered/{video_path}_rendered.mp4", fourcc, fps, (frame_width, frame_height))

    # Preprocess overlays per color
    overlays = {}
    for overlay_name, color_name in zip(overlay_names, color_names):
        if color_name not in COLOR_RANGES:
            print(f"❌ Color '{color_name}' not defined. Skipping.")
            continue
        overlays[color_name] = preprocess_overlay(overlay_name, img_scale)

    frame_count = 0

    # Process Each Frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        original_frame = frame.copy()  # Preserve original frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # For each color
        for color_name, color_info in COLOR_RANGES.items():
            if color_name not in overlays:
                continue  # Skip unused

            overlay_scaled, overlay_mask, overlay_mask_inv = overlays[color_name]
            lower_color = color_info["lower"]
            upper_color = color_info["upper"]

            mask = cv2.inRange(hsv, lower_color, upper_color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < 500:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Position overlay centered
                scaled_w, scaled_h = overlay_scaled.shape[1], overlay_scaled.shape[0]
                x_offset = x + (w - scaled_w) // 2
                y_offset = y + (h - scaled_h) // 2

                # Frame boundary checks
                x1 = max(0, x_offset)
                y1 = max(0, y_offset)
                x2 = min(x1 + scaled_w, frame_width)
                y2 = min(y1 + scaled_h, frame_height)

                overlay_crop = overlay_scaled[0:y2 - y1, 0:x2 - x1]
                mask_crop = overlay_mask[0:y2 - y1, 0:x2 - x1]
                mask_inv_crop = overlay_mask_inv[0:y2 - y1, 0:x2 - x1]

                roi = original_frame[y1:y2, x1:x2]

                bg = cv2.bitwise_and(roi, roi, mask=mask_inv_crop)
                fg = cv2.bitwise_and(overlay_crop, overlay_crop, mask=mask_crop)

                combined = cv2.add(bg, fg)
                original_frame[y1:y2, x1:x2] = combined

        # Optional Gamma Correction
        gamma_corrected = apply_gamma_correction(original_frame, gamma=1.0)
        out.write(gamma_corrected)

        # Preview Window
        cv2.imshow("Preview", gamma_corrected)
        key = cv2.waitKey(1) & 0xFF  # 1ms delay for real-time speed

        # Press 'q' to quit early
        if key == ord('q'):
            print("🚪 Exit requested. Stopping early...")
            break

        # Progress print
        print(f"Frame {frame_count}/{total_frames}", end="\r")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'\n✅ {video_path}: Overlays applied and saved (color preserved).')


def main():
    print("🧑‍💻 Starting Multi-Color Multi-Overlay Rendering...")

    for setting in video_settings:
        overlay_img_in_video(*setting)

    print("\n✨ DONE! For even better control, you can run:")
    print("ffmpeg -i rendered/your_video_rendered.mp4 -vcodec libx264 -crf 18 -preset slow final_output.mp4")


if __name__ == "__main__":
    main()
