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

def main():
    start = time.time()
    timing_data = []
    device = os.environ.get("DEVICE", "cuda")
    working_dir = os.environ.get("WORKING_DIR", "/PointStream")
    video_folder = os.environ.get("VIDEO_FOLDER", "/scenes")
    timing_csv_path = os.path.join(working_dir, "experiments/timing_data.csv")
    video_file = os.environ.get("VIDEO_FILE")
    if not video_file:
        all_videos = [v for v in os.listdir(video_folder) if v.endswith(('.mp4','.mov','.avi'))]
    else:
        all_videos = [video_file]
    estimation_model = os.environ.get("EST_MODEL", None)
    if 'yolo' in estimation_model:
        estimation_model = YOLO(estimation_model)
    else:
        raise ValueError('Model not supported.')

    for vid in all_videos:
        experiment_folder = f'{working_dir}/experiments/{os.path.basename(vid).split(".")[0]}'
        video_file = os.path.join(video_folder, vid)
        background_folder = os.path.join(experiment_folder, 'background')
        objects_folder = os.path.join(experiment_folder, 'objects')
        csv_file_path = os.path.join(experiment_folder, 'bounding_boxes.csv')
        os.makedirs(objects_folder, exist_ok=True)
        os.makedirs(background_folder, exist_ok=True)

        frame_id = 0
        df_rows = []
        estimation_start = time.time()
        results = estimation_model.track(
            source = video_file,
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

        skater_id = None  # Track the skater's object id

        for frame_id, result in enumerate(results):
            frame_img = result.orig_img
            people = extract_people(result)

            # On the first frame, pick the first detected person as the skater
            if skater_id is None and people:
                skater_id = next(iter(people.keys()))

            # Only process the skater in subsequent frames
            if skater_id in people:
                person = people[skater_id]
                person_img = frame_img[person['bbox'][1]:person['bbox'][3], person['bbox'][0]:person['bbox'][2]]
                df_rows.append(save_object(skater_id, person, person_img, objects_folder, frame_id))
                # Remove the skater from the background and inpaint
                mask = np.zeros(frame_img.shape[:2], dtype=np.uint8)
                x1, y1, x2, y2 = person["bbox"]
                mask[y1:y2, x1:x2] = 255
                inpainted = cv2.inpaint(frame_img, mask, 3, cv2.INPAINT_TELEA)
                cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), inpainted)
            else:
                # Skater not found in this frame, optionally handle this case
                cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), frame_img)

        estimation_time = time.time() - estimation_start
        estimation_fps = frame_id / estimation_time if estimation_time > 0 else 0
        timing_data.append({"video": vid, "task": "keypoint_extraction", "time_taken": estimation_time, "fps": estimation_fps})

        print(f'Total time taken for keypoint extraction: {time.time() - start} seconds')

        # Run a second pass of YOLO segmentation on the objects in the subfolders
        start = time.time()
        segmentation_model = os.environ.get("SEG_MODEL", None)
        if 'yolo' in segmentation_model:
            segmentation_model = YOLO(segmentation_model)
        else:
            raise ValueError('Model not supported.')

        min_frames = frame_id * 0.9
        segmentation_start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for obj in os.listdir(objects_folder):
                obj_folder = os.path.join(objects_folder, obj)
                if os.path.isdir(obj_folder):
                    # Delete people folders that are missing too many frames (should be every person besides skater)
                    if obj.startswith("person_") and len(os.listdir(obj_folder)) < min_frames:
                        shutil.rmtree(obj_folder)
                    else:
                        for img in os.listdir(obj_folder):
                            img_path = os.path.join(obj_folder, img)
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                mask = generate_mask(img, segmentation_model, imgsz=img.shape[:2], device=device)
                                cv2.imwrite(img_path.replace('.png', '_mask.png'), mask)
        segmentation_time = time.time() - segmentation_start
        segmentation_fps = frame_id / segmentation_time if segmentation_time > 0 else 0
        timing_data.append({"video": vid, "task": "segmentation", "time_taken": segmentation_time, "fps": segmentation_fps})
        print(f'Total time taken for segmentation: {time.time() - start} seconds')

        # Save all collected data to CSV at once
        df_columns = ['frame_id', 'object_id', 'class_id', 'x1', 'y1', 'x2', 'y2']
        if df_rows and isinstance(df_rows[0], list) and len(df_rows[0]) > 7:
            keypoints_count = (len(df_rows[0]) - 7) // 2
            for i in range(keypoints_count):
                df_columns.extend([f'keypoint_{i}_x', f'keypoint_{i}_y'])
        df = pd.DataFrame(df_rows, columns=df_columns)
        df.to_csv(csv_file_path, index=False)

        # Compute camera poses directly here
        if os.path.exists(background_folder) and len(os.listdir(background_folder)) > 0:
            compute_camera_poses(background_folder, experiment_folder)
        else:
            print(f"[!] No background frames found in {background_folder}, skipping SfM for this video.")

    timing_df = pd.DataFrame(timing_data)
    timing_df.to_csv(timing_csv_path, index=False)

    shutil.make_archive(experiment_folder, 'zip', experiment_folder)

if __name__ == "__main__":
    main()