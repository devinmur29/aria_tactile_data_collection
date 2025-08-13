import argparse
import subprocess
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run aria_mps on a specified folder.")
parser.add_argument('--foldername', required=True, help='Name of the folder inside ariarecordings.')
parser.add_argument('--small', action='store_true', help='Use the small glove config if set.')
parser.add_argument('--requestmps', action='store_true', help='Request mps if set.')
parser.add_argument('--aria_ts_ns', type=int, help='Aria timestamp in ns to align tactile data to (skips QR search if provided).')
args = parser.parse_args()

# Construct the input path
input_path = os.path.join("ariarecordings", args.foldername)
expected_path_hand = os.path.join("ariarecordings", args.foldername, f"mps_{args.foldername}_vrs", "hand_tracking","summary.json")
expected_path_slam = os.path.join("ariarecordings", args.foldername, f"mps_{args.foldername}_vrs", "slam","summary.json")
# Build the command
command = ["aria_mps", "single", "-i", input_path]

# Run the command
try:
    if not args.requestmps:
        print("MPS Not Requested - Skipping")
    elif os.path.exists(expected_path_hand) and os.path.exists(expected_path_slam):
        print("MPS Already Finished - Skipping")
    else:
        subprocess.run(command, check=True)
        print(f"✅ Successfully ran aria_mps on {input_path}")
except subprocess.CalledProcessError as e:
    print(f"❌ Error while running aria_mps: {e}")

import cv2
import numpy as np
import h5py
import json
from tqdm import tqdm
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import ast
import re

from projectaria_tools.core import data_provider
from projectaria_tools.core.mps.utils import get_nearest_hand_tracking_result
from projectaria_tools.core.sensor_data import TimeQueryOptions, TimeDomain
import projectaria_tools.core.mps as mps

# === CONFIG ===
foldername = args.foldername
base_path = os.path.join("ariarecordings", foldername)
vrs_path = os.path.join(base_path, f"{foldername}.vrs")
hand_tracking_path = os.path.join(base_path, f"mps_{foldername}_vrs", "hand_tracking", "hand_tracking_results.csv")
left_h5_path = os.path.join(base_path, "leftPressure.hdf5")
right_h5_path = os.path.join(base_path, "rightPressure.hdf5")
aligned_left_path = os.path.join(base_path, "leftPressure_aligned.hdf5")
aligned_right_path = os.path.join(base_path, "rightPressure_aligned.hdf5")

mapping_json_path = "point_weight_mappings_small.json" if args.small else "point_weight_mappings_large.json"
svg_file_path = "voronoi_regions_small.svg" if args.small else "voronoi_regions_large.svg"
output_path = os.path.join(base_path, f"{foldername}.mp4")


# Strip the np.float64(...) wrappers
def sanitize_qr_string(data):
    return re.sub(r"np\.float64\(([\d\.Ee+-]+)\)", r"\1", data)


def find_qr_and_timestamp(vrs_path, camera_name="camera-rgb"):
    print(f"[🔍] Scanning VRS file: {vrs_path}")
    provider = data_provider.create_vrs_data_provider(vrs_path)
    stream_id = provider.get_stream_id_from_label(camera_name)
    num_frames = provider.get_num_data(stream_id)

    qr_detector = cv2.QRCodeDetector()
    halfway_index = int(num_frames * 0.95)
    qr_found = False

    for idx in tqdm(range(num_frames), desc="Scanning frames for QR"):
        img_data = provider.get_image_data_by_index(stream_id, idx)
        img_np = img_data[0].to_numpy_array()
        ts_ns = img_data[1].capture_timestamp_ns

        if img_np is None or img_np.size == 0:
            continue
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)

        try:
            data, bbox, _ = qr_detector.detectAndDecode(img_np)
        except cv2.error as e:
            print(f"Skipping frame {idx} due to OpenCV error: {e}")
            continue

        if not data:
            if idx == halfway_index and not qr_found:
                raise RuntimeError("[⚠️] No QR detected in first half of frames! Make sure QR is in frame")
            continue

        print(f"QR code found at frame {idx}, timestamp {ts_ns}")
        try:
            clean_data = sanitize_qr_string(data)
            qr_json = ast.literal_eval(clean_data)
            displayed_sec = float(qr_json["gen"])
            print(f"[✓] Found QR code at index {idx} → Aria timestamp: {ts_ns} ns")
            qr_found = True
            return ts_ns, displayed_sec, provider, stream_id
        except (ValueError, KeyError, json.JSONDecodeError) as E:
            print(f"[!] Error decoding QR payload: {E}")
            continue
    raise RuntimeError("No QR code with 'displayed' field found in VRS.")


# === Timestamp selection logic ===
if args.aria_ts_ns:
    aria_ts_ns = args.aria_ts_ns
    print(f"[⏱] Using provided aria_ts_ns: {aria_ts_ns}")
    if os.path.exists(left_h5_path):
        with h5py.File(left_h5_path, 'r') as f:
            displayed_sec = float(f['ts'][0])
            print(f"[✓] Using first tactile timestamp as displayed_sec: {displayed_sec}")
    else:
        raise FileNotFoundError(f"Cannot find {left_h5_path} to get displayed_sec.")
    provider = data_provider.create_vrs_data_provider(vrs_path)
    stream_id = provider.get_stream_id_from_label("camera-rgb")
else:
    aria_ts_ns, displayed_sec, provider, stream_id = find_qr_and_timestamp(vrs_path)


def align_glove(h5_path, displayed_sec, aria_ts_ns):
    print(f"\n[🧤] Processing glove file: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        fc = f['frame_count'][0]
        ts = np.array(f['ts'][:fc])
        pressure = np.array(f['pressure'][:fc])

    print(f"First timestamp {ts[0]}")
    print(f"Last timestamp {ts[-1]}")

    diffs = np.abs(ts - displayed_sec)
    idx = np.argmin(diffs)
    ts0 = ts[idx]
    shifted_ts_ns = ((ts[idx:] - ts0) * 1e9).astype(np.int64) + aria_ts_ns

    print(f"  [✓] Aligning from tactile ts={ts0:.6f}s → Aria ts={aria_ts_ns} ns")

    base, ext = os.path.splitext(h5_path)
    new_h5_path = base + "_aligned" + ext

    with h5py.File(new_h5_path, 'w') as f_new:
        f_new.create_dataset('ts', data=shifted_ts_ns, dtype='int64')
        f_new.create_dataset('pressure', data=pressure[idx:], dtype='float32')
        f_new.create_dataset('frame_count', data=np.array([len(shifted_ts_ns)], dtype='int32'))

    print(f"  [✔] Alignment complete, saved to {new_h5_path}")


if os.path.exists(left_h5_path):
    if os.path.exists(aligned_left_path):
        print(f"[!] Skipping {aligned_left_path} (tactile is already synced)")
    else:
        align_glove(left_h5_path, displayed_sec, aria_ts_ns)
else:
    print(f"[!] Skipping {left_h5_path} (file not found)")

if os.path.exists(right_h5_path):
    if os.path.exists(aligned_right_path):
        print(f"[!] Skipping {aligned_right_path} (tactile is already synced)")
    else:
        align_glove(right_h5_path, displayed_sec, aria_ts_ns)
else:
    print(f"[!] Skipping {right_h5_path} (file not found)")


# === INIT ===
provider = data_provider.create_vrs_data_provider(vrs_path)
stream_id = provider.get_stream_id_from_label("camera-rgb")
image_data = provider.get_image_data_by_index(stream_id, 0)
image_data_array = image_data[0].to_numpy_array()
frame_height, frame_width = image_data_array.shape[:2]
# Adjust for 90 degree rotation
frame_width, frame_height = frame_height, frame_width

device_calib = provider.get_device_calibration()
rgb_calib = device_calib.get_camera_calib(provider.get_label_from_stream_id(stream_id))
T_device_rgb = rgb_calib.get_transform_device_camera()


# === TACTILE SETUP ===
def load_glove_data(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return np.array(f['ts'][:f['frame_count'][0]]), np.array(f['pressure'][:f['frame_count'][0]])


left_ts, left_pressure = load_glove_data(aligned_left_path)
right_ts, right_pressure = load_glove_data(aligned_right_path)

# Set desired output FPS
fps = 25
frame_interval_ns = int(1e9 / fps)

# Create uniformly spaced timestamps between start and end
start_time = left_ts[0]
end_time = left_ts[-1]
uniform_ts = np.arange(start_time, end_time, frame_interval_ns)
timestamp_json_path = os.path.join(base_path, "frame_timestamps.json")
with open(timestamp_json_path, "w") as f:
    json.dump({i: int(ts) for i, ts in enumerate(uniform_ts)}, f, indent=2)

mapping = json.load(open(mapping_json_path))
tree = ET.parse(svg_file_path)
root = tree.getroot()
ns = {'svg': 'http://www.w3.org/2000/svg'}

# Parse Voronoi polygons from SVG: dict label -> list of (x,y) tuples
voronoi_polygons = {
    poly.attrib['id']: [tuple(map(float, p.split(','))) for p in poly.attrib['points'].strip().split()]
    for poly in root.findall('.//svg:polygon', ns)
}
polygon_labels = list(voronoi_polygons.keys())

# Precompute weight matrices for interpolation
def precompute_weights(mapping, is_left=True):
    weights = {}
    for label, neighbors in mapping.items():
        matrix = np.zeros((16, 16), dtype=np.float32)
        total_weight = 0.0
        for q in ['NE', 'NW', 'SW', 'SE']:
            source_id, dist = neighbors[q]
            if source_id != 'N/A':
                y, x = map(int, source_id.split('-'))
                if not is_left:
                    y, x = 15 - x, 15 - y
                weight = 1 / dist if dist > 0.001 else 1e6
                matrix[y, x] += weight
                total_weight += weight
        if total_weight > 0:
            matrix /= total_weight
        weights[label] = matrix
    return weights

left_weights = precompute_weights(mapping, is_left=True)
right_weights = precompute_weights(mapping, is_left=False)

def interpolate_fast(pressure16x16, precomputed_weights):
    return {label: np.sum(W * pressure16x16) for label, W in precomputed_weights.items()}




def hsv_to_rgb(h, s, v):
    """
    Converts HSV (Hue, Saturation, Value) to RGB (Red, Green, Blue) color space.
    Hue (h) is expected in degrees [0, 360).
    Saturation (s) and Value (v) are expected in [0, 1].
    Returns RGB tuple with values scaled to [0, 255].
    """
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    rp, gp, bp = 0, 0, 0 # Initialize to avoid potential UnboundLocalError

    if h < 60:
        rp, gp, bp = c, x, 0
    elif h < 120:
        rp, gp, bp = x, c, 0
    elif h < 180:
        rp, gp, bp = 0, c, x
    elif h < 240:
        rp, gp, bp = 0, x, c
    elif h < 300:
        rp, gp, bp = x, 0, c
    else: # h >= 300 and h < 360
        rp, gp, bp = c, 0, x

    # Scale the RGB components from [0, 1] to [0, 255]
    # And convert to integers
    r = int((rp + m) * 255)
    g = int((gp + m) * 255)
    b = int((bp + m) * 255)

    # Return as (B, G, R) if you are feeding this into a system that expects BGR (like OpenCV)
    # Otherwise, return (R, G, B) if standard RGB is expected.
    # Given your previous code returned (int(b), int(g), int(r)), we'll assume BGR.
    return (b, g, r)

def value_to_color(val, vmin=0, vmax=3000):
    """
    Maps a pressure value to a color using an HSV gradient.
    High pressure (low values) will be Red (hue 0).
    Low pressure (high values) will be Blue (hue 240).
    """
    # Clamp the value to the specified min/max range
    clamped = np.clip(val, vmin, vmax)

    # Normalize the clamped value to a 0-1 range
    norm = (clamped - vmin) / (vmax - vmin)

    # Calculate hue:
    # When norm is 0 (low val/high pressure), hue is 0 (Red).
    # When norm is 1 (high val/low pressure), hue is 240 (Blue).
    hue = norm * 240 # 0 = red, 240 = blue

    # Convert to RGB using full saturation (1.0) and value (brightness) (1.0)
    return hsv_to_rgb(hue, 1.0, 1.0)

# def hsv_to_rgb(h, s, v):
#     h = h % 360
#     c = v * s
#     x = c * (1 - abs((h / 60) % 2 - 1))
#     m = v - c

#     if h < 60:
#         rp, gp, bp = c, x, 0
#     elif h < 120:
#         rp, gp, bp = x, c, 0
#     elif h < 180:
#         rp, gp, bp = 0, c, x
#     elif h < 240:
#         rp, gp, bp = 0, x, c
#     elif h < 300:
#         rp, gp, bp = x, 0, c
#     else:
#         rp, gp, bp = c, 0, x

#     r, g, b = (rp + m) * 255, (gp + m) * 255, (bp + m) * 255 # Scale to 0-255
#     return (int(b), int(g), int(r)) # Then convert to int

# def value_to_color(val,vmin=0, vmax=3000):
#     clamped = np.clip(val, vmin, vmax)
#     norm = (clamped - vmin) / (vmax - vmin)
#     hue = (1 - norm) * 240  # 240 = blue, 0 = red
#     return hsv_to_rgb(hue, 1.0, 1.0)

# def value_to_color(val, vmin=0, vmax=3000):
#     clamped = np.clip(val, vmin, vmax)
#     norm = (clamped - vmin) / (vmax - vmin)
#     hue = (1 - norm) * 240  # 240 to 0 (blue to red)
#     # Convert HSV to BGR for OpenCV:
#     # OpenCV expects hue in [0,179], saturation and value in [0,255]
#     h = int(hue / 2)  # scale 0-240 -> 0-120 (approx)
#     s = 255
#     v = 255
#     hsv_pixel = np.uint8([[[h, s, v]]])
#     bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
#     # Return tuple of ints for cv2.fillPoly color
#     return (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

def get_closest_frame(target_ts, ts_array, pressure_array):
    idx = np.argmin(np.abs(ts_array - target_ts))
    return pressure_array[idx]

# === HAND DRAWING ===
HAND_CONNECTIONS = [
    (0, 7), (7,6),(6,5),(6,8),(5,17),
    (17,14),(14,11),(11,8),(8,9),(9,10),
    (10,1),(11,12),(12,13),(13,2),(14,15),
    (15,16),(16,3),(17,18),(18,19),(19,4)
]

def project_points_to_image(points_device, device_calib, camera_name="camera-rgb"):
    transform_device_camera = device_calib.get_transform_device_sensor(camera_name).to_matrix()
    transform_camera_device = np.linalg.inv(transform_device_camera)
    rgb_calib = device_calib.get_camera_calib(camera_name)
    points_device = np.asarray(points_device)
    if points_device.ndim == 1:
        points_device = points_device[None, :]
    R = transform_camera_device[0:3, 0:3]
    t = transform_camera_device[0:3, 3]
    points_camera = (R @ points_device.T).T + t
    projected_points = []
    for pt in points_camera:
        pixel = rgb_calib.project(pt)
        projected_points.append(pixel if pixel is not None else [np.nan, np.nan])
    return np.array(projected_points)

def draw_hand_skeleton(image, landmarks_2d, color):
    for x, y in landmarks_2d:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
    for start, end in HAND_CONNECTIONS:
        if start < len(landmarks_2d) and end < len(landmarks_2d):
            pt1 = landmarks_2d[start]
            pt2 = landmarks_2d[end]
            if not (np.any(np.isnan(pt1)) or np.any(np.isnan(pt2))):
                pt1 = tuple(np.round(pt1).astype(int))
                pt2 = tuple(np.round(pt2).astype(int))
                cv2.line(image, pt1, pt2, color, 2)

# === VIDEO WRITER ===
canvas_height = int(512 * 1.5)  # 50% taller tactile overlay
combined_height = canvas_height + frame_height
writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, combined_height))

# Determine overlay coordinate extents from SVG polygons for scaling
all_points = np.array([pt for poly in voronoi_polygons.values() for pt in poly])
min_x, min_y = all_points.min(axis=0)
max_x, max_y = all_points.max(axis=0)
overlay_width = int(max_x - min_x) + 20
overlay_height = int(max_y - min_y) + 20

# Precompute polygon points as numpy arrays and shift by min_x, min_y (+10 margin)
polygon_pts_shifted = {
    label: np.array([(x - min_x + 10, y - min_y + 10) for (x, y) in pts], dtype=np.int32)
    for label, pts in voronoi_polygons.items()
}

# Scale factor to resize overlay to final (frame_width, canvas_height)
scale_x = frame_width / overlay_width
scale_y = canvas_height / overlay_height

def create_tactile_overlay(left_interp, right_interp):
    # Create blank BGR image for overlay with white background (same size as SVG bounding box)
    overlay = np.ones((overlay_height, overlay_width, 3), dtype=np.uint8) * 255

    # Fill left hand polygons with colors from left_interp
    for label, pts in polygon_pts_shifted.items():
        val = left_interp.get(label, 0)
        color = value_to_color(val)
        cv2.fillPoly(overlay, [pts], color)

    # Reflect the left hand overlay horizontally
    overlay = cv2.flip(overlay, 1)  # flipCode=1 flips around vertical axis

    # Create overlay for right hand, white background as well
    overlay_right = np.ones((overlay_height, overlay_width, 3), dtype=np.uint8) * 255
    for label, pts in polygon_pts_shifted.items():
        val = right_interp.get(label, 0)
        color = value_to_color(val)
        cv2.fillPoly(overlay_right, [pts], color)

    # Combine left and right overlays side by side horizontally
    combined_overlay = np.zeros((overlay_height, overlay_width * 2, 3), dtype=np.uint8)
    combined_overlay[:, :overlay_width] = overlay
    combined_overlay[:, overlay_width:] = overlay_right

    # Preserve aspect ratio when resizing overlay to (frame_width, canvas_height)
    combined_width = overlay_width * 2
    combined_height_local = overlay_height
    overlay_aspect = combined_width / combined_height_local
    target_aspect = frame_width / canvas_height

    if overlay_aspect > target_aspect:
        scale = frame_width / combined_width
    else:
        scale = canvas_height / combined_height_local

    new_width = int(combined_width * scale)
    new_height = int(combined_height_local * scale)

    resized_overlay = cv2.resize(combined_overlay, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    final_overlay = np.ones((canvas_height, frame_width, 3), dtype=np.uint8) * 255

    x_offset = (frame_width - new_width) // 2
    y_offset = (canvas_height - new_height) // 2

    final_overlay[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_overlay

    return final_overlay

if os.path.exists(hand_tracking_path):
    hand_tracking_results = mps.hand_tracking.read_hand_tracking_results(hand_tracking_path)
else:
    hand_tracking_results = None

def process_single_frame(ts):
    # Get RGB frame closest to ts
    image_data = provider.get_image_data_by_time_ns(stream_id, ts, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
    if not image_data or not image_data[0]:
        return None

    frame = image_data[0].to_numpy_array()

    if hand_tracking_results is not None:
        # Get hand tracking result nearest to ts
        ht_result = get_nearest_hand_tracking_result(hand_tracking_results, ts)

        # Draw left hand skeleton
        if ht_result and ht_result.left_hand and ht_result.left_hand.landmark_positions_device:
            left_2d = project_points_to_image(
                np.array(ht_result.left_hand.landmark_positions_device),
                device_calib,
                "camera-rgb"
            )
            draw_hand_skeleton(frame, left_2d, (0, 255, 0))

        # Draw right hand skeleton
        if ht_result and ht_result.right_hand and ht_result.right_hand.landmark_positions_device:
            right_2d = project_points_to_image(
                np.array(ht_result.right_hand.landmark_positions_device),
                device_calib,
                "camera-rgb"
            )
            draw_hand_skeleton(frame, right_2d, (0, 0, 255))

    # Rotate frame 90 degrees clockwise
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Interpolate tactile pressure for left and right hands
    left_frame_pressure = get_closest_frame(ts, left_ts, left_pressure)
    right_frame_pressure = get_closest_frame(ts, right_ts, right_pressure)
    left_interp = interpolate_fast(left_frame_pressure, left_weights)
    right_interp = interpolate_fast(right_frame_pressure, right_weights)

    # Create tactile overlay image with Voronoi polygons colored
    overlay_resized = create_tactile_overlay(left_interp, right_interp)

    # Compose final frame: vertical concat of overlay on top, RGB frame below
    full_frame = np.zeros((combined_height, frame_width, 3), dtype=np.uint8)
    full_frame[:canvas_height] = overlay_resized
    full_frame[canvas_height:] = frame

    return full_frame

# === MAIN LOOP with parallelism ===
max_workers = 2 # Adjust this to your CPU cores for best performance
print(f"\n[🎥] Writing synchronized recording as .mp4 video")
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    with tqdm(total=len(uniform_ts)) as pbar:
        for frame in executor.map(process_single_frame, uniform_ts):
            if frame is not None:
                writer.write(frame)
            pbar.update()


writer.release()
print(f"Video saved to {output_path}")
