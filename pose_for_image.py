from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

if __name__ == '__main__':
    # STEP 1: Import the necessary modules.
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    
    print(vars(base_options))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        min_pose_detection_confidence = 0.1,
        min_pose_presence_confidence=0.1,
        min_tracking_confidence=0.1)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    # image = mp.Image.create_from_file("/home/alexander/ml/data/office/840766925000.jpg")    
    IMAGE_FILE = 'image.jpg'
    orig_image = mp.Image.create_from_file(IMAGE_FILE)
    ho, wo = orig_image.numpy_view().shape[0:2]
    IMAGE_FILE = '/home/alexander/ml/data/office/864765877000.jpg'
    image = mp.Image.create_from_file(IMAGE_FILE)
    image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (wo, ho))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # image = mp.Image.create_from_file("/home/alexander/ml/data/office/829767402000.jpg")    
    # image = mp.Image.create_from_file("image.jpg")

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)
    print(detection_result)

    # STEP 5: Process the detection result. In this case, visualize it.
    img = np.copy(image.numpy_view())
    print(img.shape)
    if len(img.shape) == 2:
       h, w = img.shape
       img = np.tile(img.reshape(h, w, 1), (1, 1, 3))
    annotated_image = draw_landmarks_on_image(img, detection_result)
    # annotated_image = (annotated_image > 0).astype(np.uint8) * 255
    cv2.imshow('a', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    if detection_result.segmentation_masks is not None and len(detection_result.segmentation_masks) > 0:
        cv2.imshow('b', detection_result.segmentation_masks[0].numpy_view())
    cv2.waitKey(0)