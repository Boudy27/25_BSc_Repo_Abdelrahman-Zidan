import os
import can
import subprocess
import cantools
import cv2
import threading
import numpy as np
from kalman import Kalman

# Setup CAN Interface
def setup_can_interface(interface='can0', bitrate=500000):
    try:
        subprocess.run(['sudo', 'ip', 'link', 'set', interface, 'down'], check=True)
        subprocess.run(['sudo', 'ip', 'link', 'set', 'up', interface, 'type', 'can', 'bitrate', str(bitrate)], check=True)
        print(f'{interface} has been set up with bitrate {bitrate}.')
    except subprocess.CalledProcessError as e:
        print(f'Failed to set up {interface}: {e}')

# Load DBC file
dbc_path = '/home/pc6/Downloads/ESR_DBC_FILE1.dbc'  # Update this path
db = cantools.database.load_file(dbc_path)

# Global list to store detected objects with IDs
detected_objects = []
object_id_counter = 0  # Counter for object IDs

# Camera FoV
CAMERA_FOV_DEGREES = 78  # Camera field of view
HALF_FOV = CAMERA_FOV_DEGREES / 2  # 39 degrees left and right

# Initialize a dictionary to store Kalman filters for each object ID
kalman_filters = {}

# Function to listen for CAN messages
def listen_can_messages(interface='can0'):
    global detected_objects, object_id_counter
    bus = can.interface.Bus(channel=interface, interface='socketcan')

    for msg in bus:
        if msg.arbitration_id not in db._frame_id_to_message:
            continue  # Ignore unknown messages

        decoded_message = db.decode_message(msg.arbitration_id, msg.data)

        if "CAN_TX_TRACK_ANGLE" in decoded_message and "CAN_TX_TRACK_RANGE" in decoded_message and "CAN_TX_TRACK_RANGE_RATE" in decoded_message:
            track_angle = decoded_message["CAN_TX_TRACK_ANGLE"]
            track_range = decoded_message["CAN_TX_TRACK_RANGE"]
            track_range_rate = decoded_message["CAN_TX_TRACK_RANGE_RATE"]

            # Assign an ID to the detected object
            object_id = object_id_counter
            object_id_counter += 1

            # Add the detected object with its ID to the list
            detected_objects.append({
                "object_id": object_id,
                "track_angle": track_angle,
                "track_range": track_range,
                "track_range_rate": track_range_rate
            })

            # Keep only the last 10 objects to avoid clutter
            if len(detected_objects) > 10:
                detected_objects.pop(0)

# Start CAN listener in a separate thread
can_thread = threading.Thread(target=listen_can_messages, daemon=True)
can_thread.start()

# OpenCV - Display Webcam and Overlay Data
cap = cv2.VideoCapture(1)  # Adjust if necessary

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break

    height, width, _ = frame.shape  # Get screen dimensions

    # Fix the y-coordinate to the middle of the frame
    fixed_y = height // 2  # Keeps all objects horizontally centered

    for detected_object in detected_objects:
        object_id = detected_object["object_id"]
        track_angle = detected_object["track_angle"]
        track_range = detected_object["track_range"]
        track_range_rate = detected_object["track_range_rate"]

        # Create or get the Kalman filter for this object ID
        if object_id not in kalman_filters:
            # Create a new Kalman filter for this object
            kalman_filter = Kalman()
            kalman_filter.F = np.array([[1, 0], [0, 1]])  # Example state transition matrix
            kalman_filter.H = np.array([[1, 0]])  # Observation matrix (just position in this case)
            kalman_filter.x = np.array([track_angle, track_range])  # Initial state
            kalman_filter.P = np.eye(2)  # Initial uncertainty (identity matrix)
            kalman_filters[object_id] = kalman_filter
        else:
            kalman_filter = kalman_filters[object_id]

        # Apply Kalman filter prediction step
        kalman_filter.predict(Q=None)  # You can pass Q or let it use default

        # Update with new detected data (if necessary)
        measurement = np.array([track_angle])  # Example measurement (just track_angle)
        kalman_filter.update(measurement, R=np.eye(1) * 0.1)  # Measurement noise (adjust as necessary)

        # Now you can get the filtered state
        filtered_state = kalman_filter.x
        #print(f"Filtered State for Object ID {object_id}: {filtered_state}")

        # Normalize Angle (-39° to 39° mapped to screen width)
        x = int((track_angle + HALF_FOV) / CAMERA_FOV_DEGREES * width)

        # Ensure x is within the frame's width
        x = max(0, min(width - 1, x))
        
        if track_range != 0:
            # Draw the detected object as a red circle at fixed y
            cv2.circle(frame, (x, fixed_y), 10, (0, 0, 255), -1)

            # Add the range label above the circle
            range_label = f"{track_range:.1f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 255, 0)  # Green for the label
            thickness = 1
            text_size = cv2.getTextSize(range_label, font, font_scale, thickness)[0]
            text_x = x - text_size[0] // 2  # Center the text above the circle
            text_y = fixed_y - 15  # Place the label above the circle

            # Add the range label above the circle
            cv2.putText(frame, range_label, (text_x, text_y), font, font_scale, color, thickness)

            # Add the range rate label below the circle
            range_rate_label = f"{track_range_rate:.1f}m/s"
            text_size = cv2.getTextSize(range_rate_label, font, font_scale, thickness)[0]
            text_x = x - text_size[0] // 2  # Center the text below the circle
            text_y = fixed_y + 15  # Place the label below the circle

            # Add the range rate label below the circle
            cv2.putText(frame, range_rate_label, (text_x, text_y), font, font_scale, color, thickness)

    # Show the frame with radar projections
    cv2.imshow("Radar Projection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
