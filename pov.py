import cv2
import numpy as np

def video_frame_iterator(video_path):
    """Takes a video path and returns an iterator over its frames."""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def canny_filter(image, low_threshold=50, high_threshold=150):
    """Applies Canny filter to an image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

def image_difference(img1, img2):
  """Calculates the normalized sum of absolute differences between two images."""
  if img1.shape != img2.shape:
      raise ValueError("Images must have the same shape")
  diff = np.sum(np.abs(img1.astype(np.int16) - img2.astype(np.int16)))
  norm_diff = diff / (img1.size)
  return norm_diff

def generate_new_povs(current_pov, difference_score, population_size=20, sigma=0.1):
    """Generates new PoVs based on a genetic optimization loop."""
    adjusted_sigma = sigma * (1 + difference_score * 5)
    new_povs = []
    for _ in range(population_size):
        noise = np.random.normal(0, adjusted_sigma, size=6)
        new_pov = current_pov + noise
        new_povs.append(new_pov)
    return new_povs

def mock_unity_capture(pov, image_size=(480, 640)):
    """Simulates Unity camera capture from a PoV."""
    img = np.zeros(image_size, dtype=np.uint8)
    center_x = int((pov[0] % 1) * image_size[1])  # x position
    center_y = int((pov[1] % 1) * image_size[0])  # y position
    radius = int(10 + abs(pov[2] % 10))  # z position influences radius
    cv2.circle(img, (center_x, center_y), radius, 255, -1)
    noise = np.random.randint(0, 30, image_size, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img

def main(video_path, max_iterations=100, difference_threshold=0.1):
    """Main function implementing the iterative camera pose estimation process."""
    frame_iter = video_frame_iterator(video_path)
    
    # Initialize PoV (6 DoF) randomly
    current_pov = np.random.rand(6)  # Initialize with random values
    
    for frame_idx, frame in enumerate(frame_iter):
        print(f"Processing frame {frame_idx}")
        # Apply Canny filter to video frame
        filtered_frame = canny_filter(frame)
        
        # Initialize search loop
        best_pov = current_pov
        best_diff = float('inf')
        iteration = 0
        
        while iteration < max_iterations and best_diff > difference_threshold:
            # Get mock Unity capture for current PoV
            capture = mock_unity_capture(best_pov, image_size=filtered_frame.shape[:2])
            # Apply Canny filter to capture
            filtered_capture = canny_filter(capture)
            # Calculate difference
            diff = image_difference(filtered_frame, filtered_capture)
            print(f"Iteration {iteration}: PoV: {current_pov},  Difference = {diff}")
            
            # If the difference is better, update best PoV
            if diff < best_diff:
                best_diff = diff
                current_pov = best_pov
            
            # Generate new PoVs based on genetic optimization
            new_povs = generate_new_povs(current_pov, best_diff)
            
            # Evaluate new PoVs and select best
            for pov in new_povs:
                capture = mock_unity_capture(pov, image_size=filtered_frame.shape[:2])
                filtered_capture = canny_filter(capture)
                diff = image_difference(filtered_frame, filtered_capture)
                if diff < best_diff:
                    best_diff = diff
                    best_pov = pov
            iteration += 1
        
        print(f"Best difference for frame {frame_idx}: {best_diff}")
        # Use best PoV for next frame initialization
        current_pov = best_pov

    print("Processing complete")

if __name__ == "__main__":
    # Example usage (replace with your video path)
    video_path = "./Datasets/YuzuruHanyu2018/1.mp4"
    main(video_path)
