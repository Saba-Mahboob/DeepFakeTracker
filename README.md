# Deepfake Detection and Motion Analysis

This project leverages Python, OpenCV, and machine learning models to detect deepfakes, track facial features, and analyze motion in video frames.

## Features

- **Face Detection**: Extracts and preprocesses face regions from video frames.
- **Motion Detection**: Identifies camera shake and facial movements between consecutive frames.
- **Segmentation**: Masks and isolates the "person" class in a frame using a pre-trained segmentation model.
- **Camera Shake Detection**: Detects significant camera movement using optical flow.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/deepfake-motion-analysis.git
cd deepfake-motion-analysis
pip install opencv-python numpy tensorflow onnx
```

## Usage

Run the following functions on video frames:

- **Face Detection**:  
  ```python
  face_detection_and_preprocess(frame, face_mesh_model)
  ```
- **Motion Detection**:  
  ```python
  motion_detection_fun(frame_1, frame_2, face_mesh_model)
  ```
- **Segmentation**:  
  ```python
  segment(frame)
  ```
- **Camera Shake Detection**:  
  ```python
  shake(frame, corners, old_gray, cam_moved, cam_status)
  ```

## Functions Overview

### `make_rect_shape_square(rect)`
Transforms a rectangular bounding box into a square while maintaining the center.

### `face_detection_and_preprocess(frame, face_mesh_images_total)`
Detects faces and extracts landmarks using the face mesh model, returning the preprocessed face and bounding box.

### `motion_detection_fun(frame_1, frame_2, face_mesh_images)`
Analyzes two frames to detect motion using facial feature movements.

### `segment(frame)`
Performs image segmentation to detect and mask people in the frame using a pre-trained segmentation model.

### `shake(frame, corners, old_gray, cam_moved, cam_status)`
Detects camera shake between consecutive frames using optical flow and feature point tracking.

### `preprocess(image)`
Prepares an image for machine learning models by resizing, normalizing, and transforming it.

### `number_to_point(array_list, list_of_points_face)`
Converts indices into corresponding points from a list of facial landmarks.

### `calculate_norm(list_of_points_face1, list_of_points_face2)`
Computes the Euclidean distance between corresponding points from two sets of facial landmarks.

### `init_new_features(gray_frame)`
Initializes feature points for tracking using Shi-Tomasi corner detection.

### `calculateDistance(x1, y1, x2, y2)`
Computes the Euclidean distance between two points.



