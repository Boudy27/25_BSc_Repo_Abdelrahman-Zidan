import pyzed.sl as sl
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ------------------------ Setup Class IDs for Detection ------------------------
CLASS_INDICES = {
    'person': 0,
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7
}
ALLOWED_CLS_IDS = list(CLASS_INDICES.values())

# ------------------------ Initialize YOLOv8 Model ------------------------
model = YOLO("yolov8n.pt")  # or yolov8s.pt for slightly bigger model

# ------------------------ Initialize ZED Camera ------------------------
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.VGA
init_params.camera_fps = 60
init_params.depth_mode = sl.DEPTH_MODE.QUALITY
init_params.coordinate_units = sl.UNIT.METER  # Set depth units to meters
init_params.depth_minimum_distance = 0.3  # Minimum depth distance in meters
init_params.depth_maximum_distance = 20  # Maximum depth distance in meters

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera Error:", err)
    exit()
else:
    print("ZED camera connected!")

cv2.namedWindow("YOLOv8 + Depth", cv2.WINDOW_NORMAL)

# Declare Mat objects
mat_left = sl.Mat()
depth_map = sl.Mat()
point_cloud = sl.Mat()

# Runtime parameters for depth
runtime_parameters = sl.RuntimeParameters()
runtime_parameters.confidence_threshold = 100  # Depth confidence threshold
runtime_parameters.texture_confidence_threshold = 100

# Object ID counter for ZED
object_id_counter = 0
tracked_objects = {}  # Dictionary to store object positions

try:
    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Get the left image
            zed.retrieve_image(mat_left, sl.VIEW.LEFT)
            frame_left = cv2.cvtColor(mat_left.get_data(), cv2.COLOR_BGRA2BGR)
            
            # Retrieve depth map and point cloud
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # ------------------------ YOLOv8 Detection ------------------------
            results = model.predict(
                source=frame_left,
                conf=0.6,
                iou=0.45,
                imgsz=640,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                half=True if torch.cuda.is_available() else False,
                classes=ALLOWED_CLS_IDS,
                verbose=False
            )

            # Get detection results
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            # Create a blank frame for drawing (to avoid YOLO's built-in annotations)
            annotated_frame = frame_left.copy()
            
            # Process each detection
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                cls_id = int(classes[i])
                
                # Get center of the bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Get depth value at the center point
                err, point_cloud_value = point_cloud.get_value(center_x, center_y)
                
                if np.isfinite(point_cloud_value[2]):  # Check if depth is valid
                    distance = np.sqrt(point_cloud_value[0]**2 + 
                                   point_cloud_value[1]**2 + 
                                   point_cloud_value[2]**2)
                    
                    # Simple object tracking based on position
                    current_pos = (center_x, center_y, distance)
                    matched_id = None
                    
                    # Check if this object matches any previously tracked object
                    for obj_id, (prev_pos, prev_cls) in tracked_objects.items():
                        prev_x, prev_y, prev_dist = prev_pos
                        dist_moved = np.sqrt((center_x-prev_x)**2 + (center_y-prev_y)**2)
                        
                        # If close enough to previous position and same class, assume it's the same object
                        if dist_moved < 50 and prev_cls == cls_id:
                            matched_id = obj_id
                            break
                    
                    # If no match found, assign new ID
                    if matched_id is None:
                        matched_id = object_id_counter
                        object_id_counter += 1
                    
                    # Update tracked objects
                    tracked_objects[matched_id] = (current_pos, cls_id)
                    
                    # Get class name from ID
                    cls_name = [k for k, v in CLASS_INDICES.items() if v == cls_id][0]
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Display class name and distance with ZED-assigned ID
                    cv2.putText(annotated_frame, 
                               f"ID:{matched_id} {cls_name} {distance:.2f}m",
                               (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, color, 2)

            # ------------------------ Show Frame ------------------------
            cv2.imshow("YOLOv8 + Depth", annotated_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
finally:
    zed.close()
    cv2.destroyAllWindows()