import os
import can
import subprocess
import cantools
import cv2
import threading

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

# Global list to store detected points
detected_objects = []

# Camera FoV
CAMERA_FOV_DEGREES = 78  # Camera field of view
HALF_FOV = CAMERA_FOV_DEGREES / 2  # 39 degrees left and right

# Function to listen for CAN messages
def listen_can_messages(interface='can0'):
    global detected_objects
    bus = can.interface.Bus(channel=interface, interface='socketcan')

    for msg in bus:
        if msg.arbitration_id not in db._frame_id_to_message:
            continue  # Ignore unknown messages

        decoded_message = db.decode_message(msg.arbitration_id, msg.data)

        if "CAN_TX_TRACK_ANGLE" in decoded_message and "CAN_TX_TRACK_RANGE" in decoded_message and "CAN_TX_TRACK_RANGE_RATE" in decoded_message:
            track_angle = decoded_message["CAN_TX_TRACK_ANGLE"]
            track_range = decoded_message["CAN_TX_TRACK_RANGE"]
            track_range_rate = decoded_message["CAN_TX_TRACK_RANGE_RATE"]

            # Add the detected object to the list
            detected_objects.append((track_angle, track_range, track_range_rate))

            # Keep only the last 10 objects to avoid clutter
            if len(detected_objects) > 10:
                detected_objects.pop(0)

# Start CAN listener in a separate thread
can_thread = threading.Thread(target=listen_can_messages, daemon=True)
can_thread.start()

# OpenCV - Display Webcam and Overlay Data
cap = cv2.VideoCapture(0)  # Adjust if necessary

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from camera")
        break

    height, width, _ = frame.shape  # Get screen dimensions

    # Fix the y-coordinate to the middle of the frame
    fixed_y = height // 2  # Keeps all objects horizontally centered

    for track_angle, track_range, track_range_rate in detected_objects:  # Unpack the tuple
        # Normalize Angle (-39° to 39° mapped to screen width)
        x = int((track_angle + HALF_FOV) / CAMERA_FOV_DEGREES * width)

        # Check if track_range is zero and handle it
        if track_range != 0:

            # Ensure x and fixed_y are within the frame's dimensions
            x = max(0, min(width - 1, x))  # Ensure x is within the image width
            fixed_y = max(0, min(height - 1, fixed_y))  # Ensure y is within the image height

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
            text_y = fixed_y - 15  # Place the label above the circle (adjust as needed)

            # Add the range label above the circle
            cv2.putText(frame, range_label, (text_x, text_y), font, font_scale, color, thickness)

            # Add the range rate label below the circle
            range_rate_label = f"{track_range_rate:.1f}m/s"
            text_size = cv2.getTextSize(range_rate_label, font, font_scale, thickness)[0]
            text_x = x - text_size[0] // 2  # Center the text below the circle
            text_y = fixed_y + 15  # Place the label below the circle (adjust as needed)

            # Add the range rate label below the circle
            cv2.putText(frame, range_rate_label, (text_x, text_y), font, font_scale, color, thickness)

    cv2.imshow("Radar Projection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
