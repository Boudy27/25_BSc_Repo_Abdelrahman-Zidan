import pyzed.sl as sl
import cv2
import torch
from ultralytics import YOLO

# ------------------------ Setup Class IDs for Detection ------------------------
CLASS_INDICES = {
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

err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print("Camera Error:", err)
    exit()
else:
    print("ZED camera connected!")

cv2.namedWindow("YOLOv8 + Tracking", cv2.WINDOW_NORMAL)

# Declare Mat object
mat_left = sl.Mat()

try:
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Get the left image
            zed.retrieve_image(mat_left, sl.VIEW.LEFT)
            frame_left = cv2.cvtColor(mat_left.get_data(), cv2.COLOR_BGRA2BGR)

            # ------------------------ YOLOv8 Detection + Tracking ------------------------
            results = model.track(
                source=frame_left,
                persist=True,
                conf=0.6,
                iou=0.45,
                imgsz=640,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                half=True,
                classes=ALLOWED_CLS_IDS,
                tracker="bytetrack.yaml",  # You can customize if you want
                verbose=False
            )

            # Get annotated frame directly
            annotated_frame = results[0].plot()

            # ------------------------ Show Frame ------------------------
            cv2.imshow("YOLOv8 + Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
finally:
    zed.close()
    cv2.destroyAllWindows()
