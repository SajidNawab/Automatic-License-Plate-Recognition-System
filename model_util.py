from collections import defaultdict
import cv2
# import numpy as np
from ultralytics import YOLO
# import random
# import time
import easyocr


class LPPredictor:
    def __init__(self, yolo_weights='weights/best_n.pt',):
        self.detector = YOLO(yolo_weights)
        self.reader = easyocr.Reader(lang_list=['en'], gpu=True)
        self.easyocr_whitelist = '0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ'
        self.track_history = defaultdict(lambda: [])
        
    def getDetections(self, frame):
        text_region_list = []
        processed_text_region_list = []
        box_list = []
        results = self.detector.track(frame, persist=True, conf=0.4)
        annotated_frame = frame
        
        if results[0].boxes.id is not None:
            # flag = True
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot(labels=True, probs=False)

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                # Extract the text region and apply preprocessing
                text_region = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                text_region = cv2.resize(text_region, (200, 100))
                text_region = cv2.GaussianBlur(text_region, (5, 5), 0)
                processed_text_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
                processed_text_region = cv2.adaptiveThreshold(processed_text_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)

                box_list.append(box)
                text_region_list.append(text_region)
                processed_text_region_list.append(processed_text_region)
            
        return annotated_frame, box_list, text_region_list, processed_text_region_list
    
    def getTranscription(self, processed_text_region):
        self.ocr_results = self.reader.readtext(processed_text_region, allowlist=self.easyocr_whitelist, detail=0, min_size=50, batch_size=16)
        return self.ocr_results
        

