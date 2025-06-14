import os
import cv2
import streamlit as st
from datetime import datetime
from tempfile import NamedTemporaryFile
from ultralytics import YOLO
import time

# --- Streamlit Page Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Weapon Detection System")

# --- Initialize YOLO models ---
@st.cache_resource
def load_models():
    try:
        weapon_model = YOLO("best.pt")
        effect_model = YOLO("yolov8n.pt")
        print("YOLO Models loaded successfully.")
        return weapon_model, effect_model
    except Exception as e:
        st.error(f"Error loading YOLO models: {e}")
        print(f"Error loading YOLO models: {e}")
        return None, None

weapon_model, effect_model = load_models()
if weapon_model is None or effect_model is None:
    st.error("CRITICAL ERROR: YOLO models could not be loaded. The application cannot continue. Check model paths.")
    st.stop()

# --- COCO Labels, Output Folder, Sample Videos ---
coco_labels = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
output_folder = "detected_clips"
os.makedirs(output_folder, exist_ok=True)
SAMPLE_VIDEOS = {
    "CriminalThreatens (Example)": "sample/CriminalThreatens.mp4",
    "Person with a Gun (Example)": "sample/Shutter.mp4",
    "Person with Knife (Example)": "sample/WithGuns.mp4",
    "GroupPortrait Of Three People (Example)": "sample/GroupPortrait.mp4",
    "Sample Photos (Example)" : "sample/Samplephotos.mp4"
}

# --- Custom CSS ---
st.markdown("""
<style>
    .log-container .stTextArea textarea { color: #FFA500 !important; font-family: 'Consolas', 'Monaco', monospace !important; }
    .latest-log-info .stAlert { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# --- Session State Variables ---
if "log" not in st.session_state: st.session_state.log = ""
if "stop_camera" not in st.session_state: st.session_state.stop_camera = False
if "is_processing" not in st.session_state: st.session_state.is_processing = False
if "current_video_source" not in st.session_state: st.session_state.current_video_source = None
if "current_video_path" not in st.session_state: st.session_state.current_video_path = None

# --- MODIFIED Helper Function: write_log ---
def write_log(message, placeholder_to_update=None, is_error=False,
              log_level_for_ui="INFO",
              _db_video_source=None, _db_clip_path=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ui_prefix = "⚡"
    actual_ui_log_level = str(log_level_for_ui).upper()

    if is_error or "error" in message.lower():
        ui_prefix = "🔴"; actual_ui_log_level = "ERROR"
    elif "detected" in message.lower():
        ui_prefix = "⚠️"; actual_ui_log_level = "DETECTION"
    elif "recording started" in message.lower(): ui_prefix = "🔴"
    elif "recording stopped" in message.lower() or "finalized" in message.lower(): ui_prefix = "💾"
    elif "initiated" in message.lower() or "selected" in message.lower() or "uploaded" in message.lower():
        ui_prefix = "🚦"; actual_ui_log_level = "SYSTEM_EVENT"
    elif "stop" in message.lower() and "user" in message.lower():
         ui_prefix = "⏹️"; actual_ui_log_level = "USER_ACTION"
    elif "opened successfully" in message.lower(): ui_prefix = "✅"
    elif "loop finished" in message.lower(): ui_prefix = "🏁"

    new_session_log_entry = f"[{timestamp}] {ui_prefix} {message}"
    st.session_state.log = new_session_log_entry + "\n" + st.session_state.log
    max_log_lines = 100
    log_lines = st.session_state.log.splitlines()
    if len(log_lines) > max_log_lines: st.session_state.log = "\n".join(log_lines[:max_log_lines])

    if placeholder_to_update:
        if actual_ui_log_level == "ERROR": placeholder_to_update.error(f"🔴 Latest: {message}")
        else: placeholder_to_update.info(f"{ui_prefix} Latest: {message}")

    print(f"LOG: [{timestamp}] {actual_ui_log_level} - {message}")
    if _db_video_source: print(f"  Video Source: {_db_video_source}")
    if _db_clip_path: print(f"  Clip Path: {_db_clip_path}")

with st.sidebar:
    st.title("Controls & Log")
    st.markdown("Upload video, use webcam, or select sample.")
    st.markdown("---")
    uploaded_file_sidebar = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="uploader",
                                           disabled=st.session_state.is_processing)
    if st.button("📸 Start Webcam", key="start_cam", disabled=st.session_state.is_processing, use_container_width=True):
        if not st.session_state.is_processing:
            st.session_state.is_processing = True; st.session_state.stop_camera = False
            st.session_state.current_video_source = "webcam"; st.session_state.current_video_path = None
            write_log("Webcam feed initiated...", log_level_for_ui="SYSTEM_EVENT")
            st.rerun()
    st.markdown("---")
    st.subheader("Sample Videos")
    sample_options = ["None"] + list(SAMPLE_VIDEOS.keys())
    selected_sample = st.selectbox("Select sample:", options=sample_options, index=0, key="sample_select",
                                   disabled=st.session_state.is_processing)
    if selected_sample != "None":
        sample_p = SAMPLE_VIDEOS[selected_sample]
        if not os.path.exists(sample_p): st.warning(f"Sample missing: {sample_p}")
        elif st.button(f"Process: {selected_sample}", key=f"proc_samp_{selected_sample.replace(' ','_')}",
                       disabled=st.session_state.is_processing, use_container_width=True):
            if not st.session_state.is_processing:
                st.session_state.is_processing = True; st.session_state.stop_camera = False
                st.session_state.current_video_source = "sample"; st.session_state.current_video_path = sample_p
                write_log(f"Sample '{selected_sample}' selected.", log_level_for_ui="SYSTEM_EVENT")
                st.rerun()
    st.markdown("---")
    st.subheader(" Session Log")
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    st.text_area("LogView", value=st.session_state.log, height=250, disabled=True, key="session_log_area", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

st.title("Weapon & Object Detection System ")
st.markdown("Real-time weapon and object detection. Detected weapon clips are saved.")
st.markdown("---")
frame_placeholder = st.empty()
st.markdown('<div class="latest-log-info">', unsafe_allow_html=True)
status_placeholder = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

def process_video_feed(video_input, status_ph):
    cap = None
    ui_source_name = "Webcam"

    if isinstance(video_input, str):
        if not os.path.exists(video_input):
            msg = f"Video file not found: {video_input}"
            st.error(msg); write_log(msg, status_ph, is_error=True, _db_video_source=video_input)
            st.session_state.is_processing = False; st.session_state.stop_camera = True; return
        cap = cv2.VideoCapture(video_input)
        ui_source_name = os.path.basename(video_input)
        write_log(f"Processing video: {ui_source_name}...", status_ph, _db_video_source=ui_source_name)
    else:
        cap = cv2.VideoCapture(video_input)
        write_log("Attempting to start webcam...", status_ph, _db_video_source="webcam")

    if not cap or not cap.isOpened():
        msg = "Error opening video source."
        st.error(msg); write_log(msg, status_ph, is_error=True, _db_video_source=ui_source_name if isinstance(video_input, str) else "webcam")
        st.session_state.is_processing = False; st.session_state.stop_camera = True
        if cap: cap.release(); return

    status_ph.info(f"⏳ Processing {ui_source_name}...")
    write_log("Video source opened.", None, _db_video_source=ui_source_name if isinstance(video_input, str) else "webcam")

    recording = False; out = None; current_clip_path = ""
    fps = cap.get(cv2.CAP_PROP_FPS)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or fps > 120: fps = 20.0

    try:
        while not st.session_state.stop_camera:
            ret, frame = cap.read()
            if not ret:
                write_log("End of video or stream error.", status_ph, _db_video_source=ui_source_name if isinstance(video_input, str) else "webcam")
                break

            weapon_res = weapon_model(frame, verbose=False, half=True)
            effect_res = effect_model(frame, verbose=False, half=True)
            annot_frame = frame.copy()
            weapon_detected_this_frame = False
            detected_weapon_name = "Weapon"
            now_time = datetime.now()

            for r in weapon_res:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls.item()); conf = float(box.conf.item())
                        if cls == 0 and conf > 0.5:
                            weapon_detected_this_frame = True
                            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(annot_frame, (x1,y1),(x2,y2), (0,0,255),2)
                            cv2.putText(annot_frame, f"{detected_weapon_name} ({conf:.2f})", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
            for r in effect_res:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls = int(box.cls.item()); conf = float(box.conf.item())
                        if conf > 0.5:
                            obj_name = coco_labels[cls] if cls < len(coco_labels) else f"Obj-{cls}"
                            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                            cv2.rectangle(annot_frame, (x1,y1),(x2,y2), (255,0,0),2)
                            cv2.putText(annot_frame, f"{obj_name} ({conf:.2f})", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2)

            if weapon_detected_this_frame:
                if not recording:
                    ts_date = now_time.strftime("%d-%m-%y"); ts_time = now_time.strftime("%H-%M-%S")
                    vid_fname = f"{detected_weapon_name}_{ts_date}_{ts_time}.mp4"
                    current_clip_path = os.path.join(output_folder, vid_fname)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(current_clip_path, fourcc, fps, (fw,fh))
                    if not out.isOpened():
                        msg = f"Error opening VideoWriter: {current_clip_path}"
                        st.error(msg); write_log(msg, status_ph, is_error=True, _db_video_source=ui_source_name if isinstance(video_input, str) else "webcam")
                        st.session_state.stop_camera=True; break
                    recording = True
                    write_log(f"REC Start: {vid_fname}", status_ph, log_level_for_ui="RECORDING_EVENT", _db_clip_path=current_clip_path)
                if recording and out: out.write(frame)
                log_ts = now_time.strftime("%I:%M:%S %p")
                write_log(f"{detected_weapon_name} detected @{log_ts}", status_ph, log_level_for_ui="DETECTION", _db_clip_path=current_clip_path if recording else None)
            else:
                if recording:
                    recording=False
                    if out: out.release(); out=None
                    write_log(f"REC Stop: {os.path.basename(current_clip_path)}", status_ph, log_level_for_ui="RECORDING_EVENT", _db_clip_path=current_clip_path)
                    current_clip_path = ""
            
            rgb_frame = cv2.cvtColor(annot_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    except Exception as e:
        st.error(f"Error in video processing: {str(e)}")
        write_log(f"Error in video processing: {str(e)}", status_ph, is_error=True)
    finally:
        if cap: cap.release()
        if out: out.release()
        st.session_state.is_processing = False
        st.session_state.stop_camera = True
        write_log("Video processing stopped.", status_ph)

if uploaded_file_sidebar is not None:
    with NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file_sidebar.getvalue())
        tmp_file_path = tmp_file.name
    st.session_state.current_video_path = tmp_file_path
    st.session_state.current_video_source = "file"
    st.session_state.is_processing = True
    st.session_state.stop_camera = False
    write_log(f"Processing uploaded file: {uploaded_file_sidebar.name}", log_level_for_ui="SYSTEM_EVENT")
    process_video_feed(tmp_file_path, status_placeholder)
    os.unlink(tmp_file_path)

elif st.session_state.is_processing:
    if st.session_state.current_video_source == "webcam":
        process_video_feed(0, status_placeholder)
    elif st.session_state.current_video_source == "sample" and st.session_state.current_video_path:
        process_video_feed(st.session_state.current_video_path, status_placeholder)
    elif st.session_state.current_video_source == "file" and st.session_state.current_video_path:
        process_video_feed(st.session_state.current_video_path, status_placeholder)