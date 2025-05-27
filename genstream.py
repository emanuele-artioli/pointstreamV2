import os
import cv2
import shutil
import concurrent.futures
from ultralytics import YOLO, SAM
import numpy as np
import pandas as pd
import subprocess
import sys
import glob


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    # if model path contains 'yolo', load YOLO model
    if 'yolo' in model_path.lower():
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
    elif 'sam' in model_path.lower():
        try:
            model = SAM(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def setup_experiment(working_dir, video_name):
    experiment_folder = os.path.join(working_dir, "experiments", video_name)
    background_folder = os.path.join(experiment_folder, "background")
    subject_folder = os.path.join(experiment_folder, "subject")
    
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    if not os.path.exists(background_folder):
        os.makedirs(background_folder)
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)
    
    return experiment_folder, background_folder, subject_folder


def inpaint_background(background):
    """Inpaint the background using OpenCV's inpainting method."""
    # Get the mask based on where the background is transparent
    mask = background[:, :, 3]
    mask = (mask == 0).astype(np.uint8) * 255  # Convert to binary mask (0 for transparent, 255 for opaque)
    background = background[:, :, :3]  # Remove alpha channel for inpainting
    # Inpaint the background using the mask
    inpainted_background = cv2.inpaint(background, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return inpainted_background


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


def main():
    device = os.environ.get("DEVICE", "cuda")
    working_dir = os.environ.get("WORKING_DIR", "/PointStream")
    video_folder = os.environ.get("VIDEO_FOLDER", "/scenes")
    
    # Get video list
    video_file = os.environ.get("VIDEO_FILE")
    all_videos = [video_file] if video_file else [v for v in os.listdir(video_folder) if v.endswith(('.mp4','.mov','.avi'))]
    
    # Load models
    detection_model = load_model(os.environ.get("DET_MODEL"))
    estimation_model = load_model(os.environ.get("EST_MODEL"))
    segmentation_model = load_model(os.environ.get("SEG_MODEL"))

    for vid in all_videos:
        # Parse the video first to detect the bounding box's center of the figure skater (the largest person)
        video_path = os.path.join(video_folder, vid)
        experiment_folder, background_folder, subject_folder = setup_experiment(working_dir, vid)
        csv_file_path = os.path.join(experiment_folder, 'bounding_boxes.csv')
        # Create a DataFrame with columns for each keypoint coordinates
        keypoint_columns = [f'keypoint_{i}_{coord}' for i in range(17) for coord in ['x', 'y']]
        df_columns = ['frame_id'] + keypoint_columns
        pose_df = pd.DataFrame(columns=df_columns)

        results = detection_model.track(
            source=video_path,
            conf=0.25,
            iou=0.4,
            imgsz=1920,
            half='cuda' in device,
            device=device,
            batch=1,
            max_det=30,
            classes=[0],  # class 0 is 'person'
            retina_masks=True,
            stream=True
        )
        os.makedirs(experiment_folder, exist_ok=True)
        for frame_id, result in enumerate(results):
            frame_img = result.orig_img
            boxes = getattr(result, 'boxes', [])
            # calculate largest bounding box
            if boxes:
                largest_box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                x1, y1, x2, y2 = tuple(map(int, largest_box.xyxy[0]))
                # Expand the bounding box by a bit to ensure the subject is fully captured
                width = x2 - x1
                height = y2 - y1
                x1 = max(0, int(x1 - 0.1 * width))
                y1 = max(0, int(y1 - 0.1 * height))
                x2 = min(frame_img.shape[1], int(x2 + 0.1 * width))
                y2 = min(frame_img.shape[0], int(y2 + 0.1 * height))
                
                # Call segmentation model to separate the subject from the background
                seg_results = segmentation_model(frame_img, bboxes=[[x1, y1, x2, y2]])
                if seg_results:
                    # Create alpha channels
                    person_alpha = seg_results[0].masks.data.cpu().numpy().astype(np.int16).reshape(frame_img.shape[0], frame_img.shape[1])
                    background_alpha = (person_alpha - 1) * -1  # Invert mask for background
                    person_alpha = (person_alpha * 255).astype(np.uint8)  # Convert to 0-255 range
                    background_alpha = (background_alpha * 255).astype(np.uint8)  # Convert to 0-255 range

                    # Add alpha to RGB frame
                    person_rgba = cv2.merge((frame_img, person_alpha))
                    background_rgba = cv2.merge((frame_img, background_alpha))

                    # Set transparent pixels to black for RGB channels
                    person_rgba[person_rgba[:, :, 3] == 0, :3] = 0
                    background_rgba[background_rgba[:, :, 3] == 0, :3] = 0

                    # Inpaint the background
                    inpainted_background = inpaint_background(background_rgba)

                    # Save images
                    cv2.imwrite(os.path.join(subject_folder, f'{frame_id:04d}.png'), person_rgba)
                    cv2.imwrite(os.path.join(background_folder, f'{frame_id:04d}.png'), inpainted_background)

                # Call estimation model on the separated subject
                est_results = estimation_model(
                    source=os.path.join(subject_folder, f'{frame_id:04d}.png'),
                    conf=0.25,
                    iou=0.4,
                    imgsz=1920,
                    half='cuda' in device,
                    device=device,
                    batch=1,
                    max_det=1,
                    retina_masks=True,
                    stream=True
                )
                for i, result in enumerate(est_results):
                    if hasattr(result, 'keypoints'):
                        keypoints = result.keypoints.xy[i].cpu().numpy().astype(np.uint16)
                    # Save the pose estimation results
                    pose_row = [frame_id] + keypoints.flatten().tolist()
                    pose_df.loc[len(pose_df)] = pose_row
        # Save the bounding boxes to a CSV file
        pose_df.to_csv(csv_file_path, index=False)


        # Compute camera poses using COLMAP
        compute_camera_poses(background_folder, experiment_folder, frame_stride=10)


if __name__ == "__main__":
    main()