import cv2
from skimage.metrics import structural_similarity as ssim
import time

VIDEO_PATH = "videos/sample1.mp4"
SIMILARITY_THRESHOLD = 0.85
MIDPOINT = SIMILARITY_THRESHOLD + (1 - SIMILARITY_THRESHOLD)/2

INITIAL_INTERVAL = 0.1
MAX_INTERVAL = 2

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

prev_gray = None
last_kept_time = 0
current_interval = INITIAL_INTERVAL

frame_count = 0
kept_frames = 0
skipped_frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break
    frame_count +=1
    now = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 180))

    if prev_gray is None:
        print(f"Frame {frame_count}: first frame (no SSIM)")
        kept_frames += 1
        last_kept_time = now
        prev_gray = gray
        continue

    score, _ = ssim(prev_gray, gray, full=True)

    if score >= MIDPOINT:
        current_interval = min(current_interval * 2, MAX_INTERVAL)
        skipped_frames +=1
        decision = f"SKIP (interval↑ {current_interval:.2f}s)"
        
    elif score < SIMILARITY_THRESHOLD:
        prev_gray = gray
        last_kept_time= now
        current_interval = INITIAL_INTERVAL
        kept_frames +=1
        decision = 'KEPT (motion)'

    else : 
        if now - last_kept_time >= current_interval:
            prev_gray = gray
            last_kept_time= now
            current_interval = INITIAL_INTERVAL
            kept_frames +=1
            decision = 'KEPT (interval hit)'
        else:
            skipped_frames +=1
            decision = 'SKIPPED'

    print(f"Frame {frame_count}: SSIM={score:.4f} → {decision}")
    
cap.release()
print(f"Number of frames : {frame_count}")
print(f"Number of frames kept: {kept_frames}")
print(f"Number of frames skipped: {skipped_frames}")