import numpy as np
import cv2
from ultralytics import YOLO
from imutils import resize
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision.transforms as transforms
from PIL import Image
import time
from tqdm import tqdm
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def update_tracker(bboxes, image):
    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:

        obj = [
            int((x1+x2)/2), int((y1+y2)/2),
            x2-x1, y2-y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, clss, image)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
        current_ids.append(track_id)

    return bboxes2draw

def get_bounding_box_center_frame(frame, model, names, object_class):
    
    bbox_coordinates = []
    frame_copy = frame.copy()

    # Perform object detection on the input frame using the specified model
    results = model(frame)
    bboxes = []

    # Iterate over the results of object detection
    for result in results:

        # Iterate over each bounding box detected in the result
        for r in result.boxes.data.tolist():
            # Extract the coordinates, score, and class ID from the bounding box
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_name = names.get(class_id)
            if class_name in object_class and score > 0.25:
                bboxes.append((x1, y1, x2, y2, class_id, score))
                
    # Return the list of center coordinates
    bboxes_tracker = update_tracker(bboxes, frame_copy)

    # Iterate over the results of object detection
    for r in bboxes_tracker:
        # Extract the coordinates, score, and class ID from the bounding box
        x1, y1, x2, y2, class_id, track_id = r
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        track_id = int(track_id)

        # Get the class name based on the class ID
        class_name = names.get(class_id)
        bbox_coordinates.append([x1, y1, x2, y2, class_name, track_id])

        distance_str = f"ID: {track_id}, {class_name}"
        text_size, _ = cv2.getTextSize(distance_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        center_x = (x1 + x2) // 2
        text_x = center_x - text_size[0] // 2
        text_y = y1 - 10  # Place the text slightly above the bounding box

        # Draw bounding box on the frame
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_copy, distance_str, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return bbox_coordinates, bboxes_tracker, frame_copy

def load_image_pairs(data_path, drive, camera_2='image_02', camera_3='image_03'):
    """
    Load left and right image pairs from KITTI dataset based on provided drive name.
    """
    left_image_folder = os.path.join(data_path, drive, camera_2, 'data')
    left_image_files = sorted(os.listdir(left_image_folder))

    return left_image_folder, left_image_files


def display_image_pair(index, left_image_folder, left_image_files):
    """
    Display a pair of left and right images given their index in the dataset.
    """
    # Load the left image
    left_image_path = os.path.join(left_image_folder, left_image_files[index])
    left_image = cv2.cvtColor(cv2.imread(left_image_path), cv2.COLOR_BGR2RGB)

    # Return the left and right images
    return left_image, left_image_path

def process_detection_results(model, left_image, object_class=['car', 'bicycle', 'person'], index=None):
    names = model.names
    bbox_coordinates, bboxes_tracker, frame_copy = get_bounding_box_center_frame(left_image, model, names, object_class)
    return bbox_coordinates, bboxes_tracker, frame_copy

def process_pipeline(data_path, drive, output_folder, model, infer_transform, object_class=['car', 'bicycle', 'person']):

    # Load image pairs
    left_image_folder, left_image_files = load_image_pairs(data_path, drive)

    for index, image_file in enumerate(tqdm(left_image_files)):
        # Load image pairs
        left_image, left_image_path = display_image_pair(index, left_image_folder, left_image_files)

        # Perform object detection
        bbox_coordinates, bboxes_tracker, frame_copy = process_detection_results(model, left_image, object_class, index)
        
        # Construct the output file path
        output_file = os.path.join(output_folder, f"processed_{image_file}")

        # Save the processed frame with detected distances
        cv2.imwrite(output_file, frame_copy[:, :, ::-1])
        print(f"Processed image saved: {output_file}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Usage example
    data_path = 'data/2011_09_26'
    drive = '2011_09_26_drive_0005_sync'
    output_folder = 'results'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    model = YOLO('yolov8l.pt')  # Replace with your YOLO model

    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(**normal_mean_var)])    


    process_pipeline(data_path, drive, output_folder, model, infer_transform)
