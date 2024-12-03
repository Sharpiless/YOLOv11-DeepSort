import numpy as np
import cv2

def calculate_distance(bbox_coordinates, frame, depth_map, disparity_map, show_output=True, bboxes_tracker=None):
    frame_copy = frame.copy()

    # Normalize the disparity map to [0, 255]
    disparity_map_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply a colorful colormap to the disparity map
    colormap = cv2.COLORMAP_JET
    disparity_map_colored = cv2.applyColorMap(disparity_map_normalized, colormap)

    # Normalize the depth map to [0, 255]
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Apply a colorful colormap to the depth map
    colormap = cv2.COLORMAP_BONE
    depth_map_colored = cv2.applyColorMap(depth_map_normalized, colormap)

    for bbox_coor in bbox_coordinates:
        x1, y1, x2, y2, class_name, track_id = bbox_coor
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        distance = depth_map[center_y][center_x]
        print("Calculated distance:", distance)

        # Convert distance to string
        distance_str = f"{distance:.2f} m, {class_name}, id-{track_id})"

        # Draw bounding box on the frame
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw bounding box on the frame
        cv2.rectangle(disparity_map_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw bounding box on the frame
        cv2.rectangle(depth_map_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate the text size
        text_size, _ = cv2.getTextSize(distance_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Calculate the position for placing the text
        text_x = center_x - text_size[0] // 2
        text_y = y1 - 10  # Place the text slightly above the bounding box

        # Calculate the rectangle coordinates
        rect_x1 = text_x - 5
        rect_y1 = text_y - text_size[1] - 5
        rect_x2 = text_x + text_size[0] + 5
        rect_y2 = text_y + 5

        # Draw white rectangle behind the text
        cv2.rectangle(frame_copy, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), cv2.FILLED)

        # Put text at the center of the bounding box
        cv2.putText(frame_copy, distance_str, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw white rectangle behind the text
        cv2.rectangle(disparity_map_colored, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), cv2.FILLED)

        # Put text at the center of the bounding box
        cv2.putText(disparity_map_colored, distance_str, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw white rectangle behind the text
        cv2.rectangle(depth_map_colored, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), cv2.FILLED)

        # Put text at the center of the bounding box
        cv2.putText(depth_map_colored, distance_str, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return disparity_map_colored, frame_copy, depth_map_colored

def parse_projection_matrix(line):
    """
    Parse a projection matrix from a line in the calibration file.
    """
    return np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)

def get_baseline_and_focal_length(calib_cam_to_cam_path):
    """
    Extract baseline and focal length from the camera calibration file.
    """
    p_left = p_right = None
    t_left = t_right = None
    
    with open(calib_cam_to_cam_path, 'r') as file:
        for line in file:
            if line.startswith('P_rect_02:'):
                p_left = parse_projection_matrix(line)
            elif line.startswith('P_rect_03:'):
                p_right = parse_projection_matrix(line)
            elif line.startswith('T_02:'):
                t_left = np.array([float(x) for x in line.split()[1:]])
            elif line.startswith('T_03:'):
                t_right = np.array([float(x) for x in line.split()[1:]])

    if p_left is None or p_right is None:
        raise ValueError("Missing projection matrices in the calibration file.")

    if t_left is None or t_right is None:
        raise ValueError("Missing translation vectors in the calibration file.")

    # Extract focal length from the left camera projection matrix
    focal_length = p_left[0, 0]  # f_x

    # Calculate baseline as the difference in x translation between the left and right cameras
    baseline = abs(t_left[0] - t_right[0])

    return baseline, focal_length

if __name__ == '__main__':
    
    # 示例用法
    calib_cam_to_cam_path = '/mnt/data/calib_cam_to_cam.txt'
    baseline, focal_length = get_baseline_and_focal_length(calib_cam_to_cam_path)

    # 打印结果
    print(f"Baseline: {baseline} meters")
    print(f"Focal Length: {focal_length} pixels")
