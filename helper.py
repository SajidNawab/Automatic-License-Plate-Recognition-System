from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from pytube import YouTube
from model_util import LPPredictor
import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = 'Yes'
    display_ocr = 'Yes'
    is_display_tracker = True if display_tracker == 'Yes' else False
    is_display_ocr = True if display_ocr == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        # return is_display_tracker, tracker_type
    if is_display_ocr:
        ocr_val = st.radio("OCR", ("EasyOCR", "NO-OCR"), index=1)
        ocr_val = True if ocr_val == "EasyOCR" else False
        # return is_display_ocr, ocr_type
    return is_display_tracker, None, ocr_val


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None, is_display_ocr=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))
    # Display object tracking, if specified
    if is_display_tracking:
        # res = model.track(image, conf=conf, persist=True, tracker=tracker)
        annotated_frame, box_list, text_region_list, processed_text_region_list = model.getDetections(image)
    else:
        # Predict the objects in the image using the YOLOv8 model
        # res = model.predict(image, conf=conf)
        raise Exception("Not Implemented")
    
    if is_display_ocr:
        if processed_text_region_list:
            processed_text_region = processed_text_region_list[-1]
            if processed_text_region.shape[0]*processed_text_region.shape[1] >= 15000:
                text_region = text_region_list[-1]
                ocr_result = model.getTranscription(processed_text_region)
                if ocr_result:
                    recognized_text = ocr_result[0]
                else:
                    recognized_text = "NO-LP"
                
            else:
                processed_text_region = []
                text_region = []
                recognized_text = "BLUR"
        else:
            processed_text_region = []
            text_region = []
            recognized_text = "NO-LP"
    else:
        processed_text_region = []
        text_region = []
        recognized_text = "OCR NOT ENABLED"
    

    # # Plot the detected objects on the video frame
    res_plotted = annotated_frame
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    
    return text_region, processed_text_region, recognized_text
    


def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_tab = False
    # is_display_tracker, tracker = display_tracker_options()
    is_display_tracker, tracker, is_display_ocr = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    text_region, processed_text_region, recognized_text = _display_detected_frames(conf,
                                                                            model,
                                                                            st_frame,
                                                                            image,
                                                                            is_display_tracker,
                                                                            tracker,
                                                                            is_display_ocr
                                                                            )
                    
                    if not is_tab:
                        tab1, tab2 = create_tab()
                        with tab1:
                            st_frame_raw_lp = st.empty()
                        with tab2:
                            st_frame_processed_lp = st.empty()
                        is_tab = True
                    else:
                        if len(text_region) > 0:
                            with tab1:
                                st_frame_raw_lp.image(text_region,
                                            caption=recognized_text,
                                            channels="BGR"
                                            )
                            with tab2:
                                st_frame_processed_lp.image(
                                    processed_text_region,
                                    caption=recognized_text
                                )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url")
    is_tab = False
    # is_display_tracker, tracker = display_tracker_options()
    is_display_tracker, tracker, is_display_ocr = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    text_region, processed_text_region, recognized_text = _display_detected_frames(conf,
                                                                            model,
                                                                            st_frame,
                                                                            image,
                                                                            is_display_tracker,
                                                                            tracker,
                                                                            is_display_ocr
                                                                            )
                    
                    if not is_tab:
                        tab1, tab2 = create_tab()
                        with tab1:
                            st_frame_raw_lp = st.empty()
                        with tab2:
                            st_frame_processed_lp = st.empty()
                        is_tab = True
                    else:
                        if len(text_region) > 0:
                            with tab1:
                                st_frame_raw_lp.image(text_region,
                                            caption=recognized_text,
                                            channels="BGR"
                                            )
                            with tab2:
                                st_frame_processed_lp.image(
                                    processed_text_region,
                                    caption=recognized_text
                                )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_tab = False
    # is_display_tracker, tracker = display_tracker_options()
    is_display_tracker, tracker, is_display_ocr = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    text_region, processed_text_region, recognized_text = _display_detected_frames(conf,
                                                                            model,
                                                                            st_frame,
                                                                            image,
                                                                            is_display_tracker,
                                                                            tracker,
                                                                            is_display_ocr
                                                                            )
                    
                    if not is_tab:
                        tab1, tab2 = create_tab()
                        with tab1:
                            st_frame_raw_lp = st.empty()
                        with tab2:
                            st_frame_processed_lp = st.empty()
                        is_tab = True
                    else:
                        if len(text_region) > 0:
                            with tab1:
                                st_frame_raw_lp.image(text_region,
                                            caption=recognized_text,
                                            channels="BGR"
                                            )
                            with tab2:
                                st_frame_processed_lp.image(
                                    processed_text_region,
                                    caption=recognized_text
                                )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def create_tab():
    tab_titles = ['Raw LP', 'Processed LP']
    tab1, tab2 = st.tabs(tab_titles)
    return tab1, tab2
def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker, is_display_ocr = display_tracker_options()
    # tab_titles = ['Raw LP', 'Processed LP']
    # tab1, tab2 = st.tabs(tab_titles)
    is_tab = False
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            # st_frame_raw_lp = st.empty()
            # st_frame_processed_lp = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    text_region, processed_text_region, recognized_text = _display_detected_frames(conf,
                                                                            model,
                                                                            st_frame,
                                                                            image,
                                                                            is_display_tracker,
                                                                            tracker,
                                                                            is_display_ocr
                                                                            )
                    if not is_tab:
                        tab1, tab2 = create_tab()
                        with tab1:
                            st_frame_raw_lp = st.empty()
                        with tab2:
                            st_frame_processed_lp = st.empty()
                        is_tab = True
                    else:
                        if len(text_region) > 0:
                            with tab1:
                                st_frame_raw_lp.image(text_region,
                                            caption=recognized_text,
                                            channels="BGR"
                                            )
                            with tab2:
                                st_frame_processed_lp.image(
                                    processed_text_region,
                                    caption=recognized_text
                                )
                                
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
