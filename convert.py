import os
import sys

import cv2 as cv
import requests
from ultralytics import YOLO


def download_model(model):
    """Downloads a pretreined YOLOv8 model from Ultralytics github
    """
    model_path = f'./models/{model}'

    if os.path.exists(model_path):
        print(f'Model {model} already exists at {model_path}')
        return model_path
    
    model_url = f'https://github.com/ultralytics/assets/releases/download/v0.0.0/{model}'
    with requests.get(model_url, stream=True) as response:
        with open(model_path, 'wb') as fd:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                fd.write(chunk)
    
    print(f'Model {model} saved to {model_path}')
    return model_path

def interpolate(trackings, start, end):
    """Tis function fill in the gaps between two XY points in tackings.
    """
    total_steps = end - start
    end_pos = trackings[end]
    start_pos = trackings[start]
    for i in range(start, end + 1):
        if start_pos is None:
            trackings[i] = end_pos
        elif end_pos is None:
            trackings[i] = start_pos
        else:
            step = i - start + 1
            sx, sy = start_pos
            ex, ey = end_pos
            step_x, step_y = (ex - sx) / total_steps, (ey - sy) / total_steps
            trackings[i] = (int(sx + step_x * step), int(sy + step_y * step))
    return trackings

def run(source, confidence=0.25):
    model_path = download_model('yolov8n.pt')
    
    model = YOLO(model_path)
    model.to('cuda')
    model.fuse()
    
    frame_id = 0
    last_ball_id = None
    ball_tracking = []
    classes = ['person', 'sports ball']
    predict_classes = [code for code, label in model.names.items() if label in classes]

    video_cap = cv.VideoCapture(source)

    """ First pass: tracking the sports ball object from the image 
    and feeding into the ball_tracking index."""
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        results = model(frame, stream=True, conf=confidence, classes=predict_classes)
        ball_position = None
        for res in results:
            data = res.boxes.data
            if not data.shape:
                continue

            for x1, y1, x2, y2, conf, cls in data:
                label = model.names.get(int(cls))
                if label == 'sports ball':
                    cx = int((x2 + x1) // 2)
                    cy = int((y2 + y1) // 2)
                    ball_position = (cx, cy)

        ball_tracking.append(ball_position)
        
        if ball_position:
            current_ball_id = len(ball_tracking) - 1
            if last_ball_id is None:
                interpolate(ball_tracking, start=0, end=current_ball_id)
            else:
                interpolate(ball_tracking, start=last_ball_id, end=current_ball_id)
            last_ball_id = current_ball_id
        
        frame_id += 1

    if ball_tracking[-1] is None:
        interpolate(ball_tracking, start=last_ball_id, end=len(ball_tracking) - 1)
    
    video_cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    cap_width  = int(video_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    cap_height = int(video_cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 

    ball_pos = None
    frame_id = 0
    output = source.split('/')[-1]

    formated_path = f'./results/{output}'
    print(f'Saving to {formated_path}')

    format_result = cv.VideoWriter(formated_path, cv.VideoWriter_fourcc(*'MP4V'), 30, (405, cap_height))
        
    """Second pass: Using the previous pass trackings to crop the video"""
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        
        if frame_id >= len(ball_tracking):
            break

        ball_pos = ball_tracking[frame_id] or ball_pos
        cx, cy = ball_pos
        frame_id += 1

        left_x = max(0, cx - 203)
        if left_x + 405 > frame.shape[1]:
            right_x = frame.shape[1]
            left_x = right_x - 405
        else:
            right_x = left_x + 405

        format_result.write(frame[:, left_x: right_x, :])
        
    video_cap.release()
    format_result.release()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f'{sys.argv[0]} filepath.mp4')
        sys.exit(1)

    source = sys.argv[1]
    if len(sys.argv) == 3:
        conf = float(sys.argv[2])
    else:
        conf = 0.25

    run(source, confidence=conf)