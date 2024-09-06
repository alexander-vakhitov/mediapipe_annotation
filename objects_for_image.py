
# Visualization utilities
#@markdown We implemented some functions to visualize the object detection results. <br/> Run the following cell to activate the functions.
import cv2
import numpy as np
import os
import json
import random
import shutil
import argparse

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 1
RECT_COLOR = (255, 0, 0)  # red
TEXT_COLOR = (0, 0, 255)  # 

def visualize_results(detection_result, image, removed={}):
    for i, detection in enumerate(detection_result):
        category = detection.categories[0]
        category_name = category.category_name
        if category_name != 'person':
            continue
        if i in removed:
           continue
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, RECT_COLOR, 3)

        # Draw label and score
        
        probability = round(category.score, 2)
        result_text = str(i) + ' ' + category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                        MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)


def check_annotation(
    image,
    detection_result
):
  """
  """

  detection_result_flt = []
  for detection in detection_result.detections:
      category = detection.categories[0]
      category_name = category.category_name
      if category_name != 'person':
          continue
      detection_result_flt.append(detection)
  detection_result = detection_result_flt
  
  is_finalized = False
  is_valid = True

  removed  = {}

  while not is_finalized:
    current_image = np.copy(image)
    visualize_results(detection_result, current_image, removed)    
            
    cv2.imshow('A', current_image)
    key = cv2.waitKey(0)
    if key == ord('f'):
       is_finalized = True
    elif key == ord('e'):
        is_finalized = True
        is_valid = False
    else:
       r = int(chr(key))
       if r in removed:
           del removed[r]           
       else:
           removed[r] = 1

    final_bbs = []
    for i, detection in enumerate(detection_result):
        category = detection.categories[0]
        category_name = category.category_name
        if category_name != 'person':
            continue
        if i in removed:
           continue
        bb = detection.bounding_box
        final_bbs.append([bb.origin_x, bb.origin_y, bb.width, bb.height])

  return current_image, final_bbs, is_valid


def count_people(detection_result):
    cnt = 0
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        if category_name != 'person':
            continue
        cnt += 1
    return cnt


def process_image_folder(folder_path, visualization_folder_path, output_folder_path, detector, wo, ho, max_people, max_empty):
    all_bbs_by_filename = {}
    people_cnt = 0
    empty_cnt = 0
    file_lst = os.listdir(folder_path)
    random.shuffle(file_lst)
    for fname in file_lst:
        if (people_cnt >= max_people) and (empty_cnt >= max_empty):
            break
        initial_image = cv2.imread(f'{folder_path}/{fname}')
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        hi, wi = initial_image.shape[0:2]

        image = cv2.resize(initial_image, (wo, ho))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # STEP 4: Detect objects in the input image.
        detection_result = detector.detect(image)
        current_people = count_people(detection_result)
        if people_cnt >= max_people and current_people > 0:
            continue
        if empty_cnt >= max_empty and current_people == 0:
            continue

        # STEP 5: Process the detection result. In this case, visualize it.
        image_copy = np.copy(image.numpy_view())
        # annotated_image = visualize(image_copy, detection_result)
        annotated_image, bbs, is_valid = check_annotation(image_copy, detection_result)

        if not is_valid:
            continue
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)        
        cv2.imwrite(f'{visualization_folder_path}/{fname}', rgb_annotated_image)
        cv2.imwrite(f'{output_folder_path}/{fname}', initial_image)

        # resize back the bbs
        bbs_resized = []
        coeff_resize_width = wi / float(wo)
        coeff_resize_height = hi / float(ho)
        for bb in bbs:
            bbs_resized.append([coeff_resize_width * bb[0], coeff_resize_height * bb[1], coeff_resize_width * bb[2], coeff_resize_height * bb[3]])
        all_bbs_by_filename[fname] = bbs_resized
        if len(bbs) > 0:
            people_cnt += 1
        else:
            empty_cnt += 1        
        print(f'Current counters: {people_cnt}/{max_people} {empty_cnt}/{max_empty}')
    
    with open(f'{visualization_folder_path}/annotation.json', 'w', encoding='utf-8') as f:
        json.dump(all_bbs_by_filename, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluates person detection")
    parser.add_argument("dataset", help="path to the dataset")
    parser.add_argument("visuals", help="visualization of the detections")
    parser.add_argument("output", help="path to the output folder, to save images and the annotation in json")
    args = parser.parse_args()    
  
    folder_path = args.dataset
    visualization_folder_path = args.visuals
    output_folder_path = args.output

    # folder_path = '/mnt/data_pipeline/ml/data/office/'
    # visualization_folder_path = '/mnt/data_pipeline/ml/results/mediapipe_detector/office_visualize/'
    # output_folder_path = '/mnt/data_pipeline/ml/results/mediapipe_detector/office/'

    os.makedirs(visualization_folder_path, exist_ok=True)
    os.makedirs(output_folder_path, exist_ok=True)

    # Create an ObjectDetector object.
    base_options = python.BaseOptions(model_asset_path='efficientdet_lite2.tflite')
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                        score_threshold=0.2)
    detector = vision.ObjectDetector.create_from_options(options)

    # Load the input image to get the image resolution.
    IMAGE_FILE = 'image.jpg'
    orig_image = mp.Image.create_from_file(IMAGE_FILE)
    ho, wo = orig_image.numpy_view().shape[0:2]

    # Process the folder
    process_image_folder(folder_path, visualization_folder_path, output_folder_path, detector, wo, ho, max_people=50, max_empty=50)