import os
import time
import cv2
import shutil
import concurrent.futures
from ultralytics import YOLO
import numpy as np
import pandas as pd
import subprocess
import sys
import glob


def extract_people(result):
    """Extracts person info (ID, confidence, bbox, keypoints) from a YOLO result into a dictionary."""
    boxes = getattr(result, 'boxes', [])
    keypoints = getattr(result, 'keypoints', [])
    people = {}
    for i, box in enumerate(boxes):
        obj_id = int(box.id) if box.id is not None else 9999
        obj = {
            'conf': float(box.conf),
            'bbox': tuple(map(int, box.xyxy[0])),
            'keypoints': keypoints.xy[i].cpu().numpy().astype(np.uint8) if keypoints and i < len(keypoints.data) else None
        }
        people[obj_id] = obj
    return people

def save_object(object_id, object, obj_img, experiment_folder, frame_id):
    '''Save an object to its subfolder and log its bounding box coordinates.'''
    obj_folder = os.path.join(experiment_folder, f'person_{object_id}')
    os.makedirs(obj_folder, exist_ok=True)
    cv2.imwrite(os.path.join(obj_folder, f'{frame_id}.png'), obj_img)
    x1, y1, x2, y2 = object['bbox']
    obj_info = [frame_id, object_id, 'person', x1, y1, x2, y2] 
    if object['keypoints'] is not None:
        keypoints = object['keypoints']
        keypoints = keypoints.reshape(-1, 2)
        obj_info.extend(keypoints.flatten().tolist())
    return obj_info

def generate_mask(img, model, conf=0.01, iou=0.01, imgsz=None, device=None):
    '''Use YOLO to segment a person from the background.'''
    results = model.predict(
        source = img,
        conf = conf,
        iou = iou,
        imgsz = imgsz,
        half = 'cuda' in device,
        device = device,
        batch = 1,
        max_det = 10,
        classes = [0], # only person
        retina_masks = True
    )
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if hasattr(results[0], 'masks') and results[0].masks is not None:
        # Use the first mask (since we track only one person)
        person_mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, person_mask)
    return mask

def estimate_camera_pose(img1, img2, K):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 5:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
    if E is None:
        return None, None
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten(), t.flatten()

def compute_camera_poses(background_folder, experiment_folder):
    background_frames = [bg for bg in os.listdir(background_folder) if bg.endswith(('.jpg', '.png'))]
    background_frames.sort()
    poses = pd.DataFrame(columns=['frame_id', 'rvec_x', 'rvec_y', 'rvec_z', 't_x', 't_y', 't_z'])
    for i in range(len(background_frames) - 1):
        img1 = cv2.imread(os.path.join(background_folder, background_frames[i]))
        img2 = cv2.imread(os.path.join(background_folder, background_frames[i + 1]))
        if img1 is None or img2 is None:
            continue
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        K = np.array([[1000, 0, img1.shape[1] // 2],
                      [0, 1000, img1.shape[0] // 2],
                      [0, 0, 1]])
        rvec, t = estimate_camera_pose(img1_gray, img2_gray, K)
        if rvec is not None and t is not None:
            poses = pd.concat([poses, pd.DataFrame([[i, *rvec, *t]], columns=poses.columns)], ignore_index=True)
    poses.to_csv(os.path.join(experiment_folder, 'camera_poses.csv'), index=False)
    print(f"Camera poses saved to {os.path.join(experiment_folder, 'camera_poses.csv')}")

def compute_camera_poses(background_folder, experiment_folder, frame_stride=1):
    """Run COLMAP to estimate camera poses from inpainted background frames.

    Args:
        background_folder (str): Path to folder with full background frames.
        experiment_folder (str): Path where COLMAP outputs will go.
        frame_stride (int): Use every N-th frame (default=1, i.e., use all).
    """
    sparse_dir = os.path.join(experiment_folder, 'sparse')
    database_path = os.path.join(experiment_folder, 'colmap.db')
    colmap_image_path = os.path.join(experiment_folder, 'colmap_images')
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(colmap_image_path, exist_ok=True)

    print(f"[COLMAP] Preparing input frames with stride {frame_stride}...")

    # Copy every N-th frame to a new folder for COLMAP
    all_frames = sorted(glob.glob(os.path.join(background_folder, '*')))
    selected_frames = all_frames[::frame_stride]
    for frame in selected_frames:
        shutil.copy(frame, os.path.join(colmap_image_path, os.path.basename(frame)))

    print("[COLMAP] Creating database and extracting features...")

    subprocess.run([
        'colmap', 'feature_extractor',
        '--database_path', database_path,
        '--image_path', colmap_image_path,
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'PINHOLE',
        '--SiftExtraction.use_gpu', '1'
    ], check=True)

    print("[COLMAP] Matching features...")

    subprocess.run([
        'colmap', 'exhaustive_matcher',
        '--database_path', database_path,
        '--SiftMatching.use_gpu', '1'
    ], check=True)

    print("[COLMAP] Running SfM...")

    subprocess.run([
        'colmap', 'mapper',
        '--database_path', database_path,
        '--image_path', colmap_image_path,
        '--output_path', sparse_dir
    ], check=True)

    print(f"[COLMAP] Finished. Results in {sparse_dir}")
def load_model(model_path):
    """Load a YOLO model from the given path."""
    if 'yolo' in model_path:
        return YOLO(model_path)
    else:
        raise ValueError('Model not supported.')

def setup_experiment(working_dir, vid):
    """Set up the experiment directories for a given video."""
    experiment_folder = f'{working_dir}/experiments/{os.path.basename(vid).split(".")[0]}'
    background_folder = os.path.join(experiment_folder, 'background')
    objects_folder = os.path.join(experiment_folder, 'objects')
    os.makedirs(objects_folder, exist_ok=True)
    os.makedirs(background_folder, exist_ok=True)
    return experiment_folder, background_folder, objects_folder

def process_video(video_path, objects_folder, background_folder, estimation_model, device):
    """Process the video for keypoint extraction."""
    frame_id = 0
    df_rows = []
    results = estimation_model.track(
        source = video_path,
        conf = 0.25,
        iou = 0.1,
        imgsz = 640,
        half = 'cuda' in device,
        device = device,
        batch = 10,
        max_det = 30,
        classes = [0], # person
        retina_masks = True,
        stream = True,
        persist = True,
    )

    protagonist_id = None  # Track the protagonist's object id

    for frame_id, result in enumerate(results):
        frame_img = result.orig_img
        people = extract_people(result)

        # On the first frame, pick the first detected person as the protagonist
        if protagonist_id is None and people:
            protagonist_id = next(iter(people.keys()))

        # Only process the protagonist in subsequent frames
        if protagonist_id in people:
            person = people[protagonist_id]
            person_img = frame_img[person['bbox'][1]:person['bbox'][3], person['bbox'][0]:person['bbox'][2]]
            df_rows.append(save_object(protagonist_id, person, person_img, objects_folder, frame_id))
            # Remove the protagonist from the background and inpaint
            mask = np.zeros(frame_img.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = person["bbox"]
            mask[y1:y2, x1:x2] = 255
            inpainted = cv2.inpaint(frame_img, mask, 3, cv2.INPAINT_TELEA)
            cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), inpainted)
        else:
            # protagonist not found in this frame, optionally handle this case
            cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), frame_img)

    return df_rows, frame_id

def segment_objects(objects_folder, segmentation_model, device, min_frames):
    """Segment objects in the video frames and create transparent PNGs."""
    for obj in os.listdir(objects_folder):
        obj_folder = os.path.join(objects_folder, obj)
        if os.path.isdir(obj_folder):
            # Delete people folders that are missing too many frames
            if obj.startswith("person_") and len(os.listdir(obj_folder)) < min_frames:
                shutil.rmtree(obj_folder)
            else:
                for img_file in os.listdir(obj_folder):
                    if not img_file.endswith('.png') or '_mask' in img_file or '_transparent' in img_file:
                        continue
                        
                    img_path = os.path.join(obj_folder, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Generate mask using the segmentation model
                        mask = generate_mask(img, segmentation_model, imgsz=img.shape[:2], device=device)
                        
                        # Save the mask for reference
                        mask_path = img_path.replace('.png', '_mask.png')
                        cv2.imwrite(mask_path, mask)
                        
                        # Create transparent PNG
                        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                        # Set alpha channel based on the mask (255 for person, 0 for background)
                        rgba[:, :, 3] = mask
                        
                        # Save transparent PNG
                        transparent_path = img_path.replace('.png', '_transparent.png')
                        cv2.imwrite(transparent_path, rgba)

def save_detection_data(df_rows, csv_file_path):
    """Save detection data to CSV."""
    df_columns = ['frame_id', 'object_id', 'class_id', 'x1', 'y1', 'x2', 'y2']
    if df_rows and isinstance(df_rows[0], list) and len(df_rows[0]) > 7:
        keypoints_count = (len(df_rows[0]) - 7) // 2
        for i in range(keypoints_count):
            df_columns.extend([f'keypoint_{i}_x', f'keypoint_{i}_y'])
    pd.DataFrame(df_rows, columns=df_columns).to_csv(csv_file_path, index=False)

def main():
    start = time.time()
    timing_data = []
    device = os.environ.get("DEVICE", "cuda")
    working_dir = os.environ.get("WORKING_DIR", "/PointStream")
    video_folder = os.environ.get("VIDEO_FOLDER", "/scenes")
    timing_csv_path = os.path.join(working_dir, "experiments/timing_data.csv")
    
    # Get video list
    video_file = os.environ.get("VIDEO_FILE")
    all_videos = [video_file] if video_file else [v for v in os.listdir(video_folder) if v.endswith(('.mp4','.mov','.avi'))]
    
    # Load models
    estimation_model = load_model(os.environ.get("EST_MODEL"))
    segmentation_model = load_model(os.environ.get("SEG_MODEL"))
    
    for vid in all_videos:
        video_path = os.path.join(video_folder, vid)
        experiment_folder, background_folder, objects_folder = setup_experiment(working_dir, vid)
        csv_file_path = os.path.join(experiment_folder, 'bounding_boxes.csv')
        
        # Process video (keypoint extraction)
        estimation_start = time.time()
        df_rows, frame_count = process_video(video_path, objects_folder, background_folder, estimation_model, device)
        estimation_time = time.time() - estimation_start
        timing_data.append({
            "video": vid,
            "task": "keypoint_extraction",
            "time_taken": estimation_time,
            "fps": frame_count / estimation_time if estimation_time > 0 else 0
        })
        
        # Segment objects
        segmentation_start = time.time()
        min_frames = frame_count * 0.9
        segment_objects(objects_folder, segmentation_model, device, min_frames)
        segmentation_time = time.time() - segmentation_start
        timing_data.append({
            "video": vid,
            "task": "segmentation",
            "time_taken": segmentation_time,
            "fps": frame_count / segmentation_time if segmentation_time > 0 else 0
        })
        
        # Save bounding boxes and keypoints
        save_detection_data(df_rows, csv_file_path)
        
        # Compute camera poses
        if os.path.exists(background_folder) and len(os.listdir(background_folder)) > 0:
            compute_camera_poses(background_folder, experiment_folder, frame_stride=30)
        else:
            print(f"[!] No background frames found in {background_folder}, skipping SfM for this video.")
    
    # Save timing data and create archive
    pd.DataFrame(timing_data).to_csv(timing_csv_path, index=False)
    shutil.make_archive(experiment_folder, 'zip', experiment_folder)

if __name__ == "__main__":
    main()