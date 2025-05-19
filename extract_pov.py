import cv2
import numpy as np
import os
import pandas as pd
import sys
import shutil

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

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    
    # Convert rotation matrix to rotation vector (Rodrigues)
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten(), t.flatten()

def main():
    background_folder = sys.argv[1]
    experiment_folder = sys.argv[2]
    background_frames = [bg for bg in os.listdir(background_folder) if bg.endswith(('.jpg', '.png'))]
    background_frames.sort()
    poses = pd.DataFrame(columns=['frame_id', 'rvec_x', 'rvec_y', 'rvec_z', 't_x', 't_y', 't_z'])

    # For each pair of background frames
    for i in range(len(background_frames) - 1):
        img1 = cv2.imread(os.path.join(background_folder, background_frames[i]))
        img2 = cv2.imread(os.path.join(background_folder, background_frames[i + 1]))
        if img1 is None or img2 is None:
            continue
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Camera intrinsic matrix
        K = np.array([[1000, 0, img1.shape[1] // 2],
                    [0, 1000, img1.shape[0] // 2],
                    [0, 0, 1]])

        # Estimate camera pose
        rvec, t = estimate_camera_pose(img1, img2, K)
        
        # Save camera pose to dataframe
        poses = pd.concat([poses, pd.DataFrame([[i, *rvec, *t]], columns=['frame_id', 'rvec_x', 'rvec_y', 'rvec_z', 't_x', 't_y', 't_z'])], ignore_index=True)
    # Save poses to CSV
    poses.to_csv(os.path.join(experiment_folder, 'camera_poses.csv'), index=False)
    print(f"Camera poses saved to {os.path.join(experiment_folder, 'camera_poses.csv')}")

    shutil.make_archive(experiment_folder, 'zip', experiment_folder)
    # shutil.rmtree(experiment_folder)
    
if __name__ == "__main__":
    main()