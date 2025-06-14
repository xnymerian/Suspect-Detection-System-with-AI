# Weapon Detection System

This project is an artificial intelligence system that analyzes real-time video feeds from security cameras to detect weapons and provide necessary alerts.

## Features

- Real-time weapon detection
- Webcam and video file support
- Automatic recording system
- Visual alerts
- Detailed logging system
- High-performance YOLO model

## Installation

1. Install required packages:
pip install -r requirements.txt

2. Download YOLO models:
- `best.pt` (Custom trained model for weapon detection)
- `yolov8n.pt` (For general object detection)

## Usage

1. Start the application:
python -m streamlit run app.py

2. From the browser interface:
   - Upload videos
   - Use webcam
   - Test sample videos

3. Detected weapons are marked with red boxes and automatically recorded.

## Security System Integration

This system can be integrated with existing security systems:

### 1. Door Locking System Integration

def lock_doors():
    # Door locking API
    pass

def on_weapon_detected():
    lock_doors()
    send_alert()


### 2. Alarm System Integration

def trigger_alarm():
    # Alarm system API
    pass

def send_alert():
    trigger_alarm()
    notify_security()


### 3. Security Personnel Notification

def notify_security():
    # SMS/Email notification
    # Security personnel mobile app notification
    pass

## Integration Options

1. **API Integration**
   - Communication with other systems via REST API
   - Real-time notifications with WebSocket support
   - IoT device connection via MQTT protocol

2. **Hardware Integration**
   - Direct connection with IP cameras
   - RS485/RS232 communication with security systems
   - Visitor management system integration

3. **Software Integration**
   - Security Management Systems (SMS)
   - Event Management Systems
   - Record Management Systems

## Security Measures

1. **Data Security**
   - Encrypted video streaming
   - Secure API communication
   - Data backup system

2. **System Security**
   - User authorization
   - Access logs
   - Firewall configuration



