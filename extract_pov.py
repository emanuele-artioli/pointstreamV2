import os
import pycolmap
import shutil
import json

def run_sfm_pipeline(image_dir, output_dir, camera_model='SIMPLE_RADIAL', focal_length_px=1200):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Automatically extract image size
    import cv2
    sample_img = cv2.imread(os.path.join(image_dir, sorted(os.listdir(image_dir))[0]))
    h, w = sample_img.shape[:2]

    # 1. Feature extraction and matching
    pycolmap.extract_features(image_path=image_dir, database_path=f"{output_dir}/database.db")
    pycolmap.match_features(database_path=f"{output_dir}/database.db")

    # 2. SfM reconstruction
    cameras = [{
        "model": camera_model,
        "width": w,
        "height": h,
        "params": [focal_length_px, w/2, h/2]  # fx, cx, cy for SIMPLE_RADIAL
    }]

    maps = pycolmap.incremental_mapping(
        database_path=f"{output_dir}/database.db",
        image_path=image_dir,
        output_path=output_dir,
        camera_mode="auto",
        single_camera=True,
        camera_params=cameras,
    )

    sfm_map = maps[0]
    poses = {}

    for image_name, image in sfm_map.images.items():
        qvec = image.qvec.tolist()  # [qw, qx, qy, qz]
        tvec = image.tvec.tolist()  # [tx, ty, tz]
        poses[image_name] = {
            "quaternion": qvec,
            "position": tvec
        }

    with open(os.path.join(output_dir, "camera_poses.json"), "w") as f:
        json.dump(poses, f, indent=2)

    print(f"[✓] Done. Poses written to {output_dir}/camera_poses.json")

# Example usage
run_sfm_pipeline(
    image_dir="frames",
    output_dir="sfm_output",
    focal_length_px=1200  # Adjust based on your camera
)
