import streamlit as st
import cv2
import tempfile
import time
from skimage.metrics import structural_similarity as ssim

def process_video(video_path):
    # Constants
    SIMILARITY_THRESHOLD = 0.85
    MIDPOINT = SIMILARITY_THRESHOLD + (1 - SIMILARITY_THRESHOLD) / 2
    INITIAL_INTERVAL = 0.1
    MAX_INTERVAL = 2.0

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None

    # UI Elements
    progress_bar = st.progress(0.0)
    col1, col2, col3 = st.columns(3)
    metric_total = col1.empty()
    metric_kept = col2.empty()
    metric_skipped = col3.empty()
    log_area = st.empty()

    total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # State variables
    prev_gray = None
    last_kept_time = 0.0
    current_interval = INITIAL_INTERVAL

    frame_count = 0
    kept_frames = 0
    skipped_frames = 0
    logs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()

        # Preprocess frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))

        score = 0.0
        decision = ""

        if prev_gray is None:
            prev_gray = gray
            last_kept_time = now
            kept_frames += 1
            decision = "KEPT (first frame)"
        else:
            score, _ = ssim(prev_gray, gray, full=True)

            if score >= MIDPOINT:
                current_interval = min(current_interval * 2, MAX_INTERVAL)
                skipped_frames += 1
                decision = f"SKIP (interval ↑ {current_interval:.2f}s)"

            elif score < SIMILARITY_THRESHOLD:
                prev_gray = gray
                last_kept_time = now
                current_interval = INITIAL_INTERVAL
                kept_frames += 1
                decision = "KEPT (motion)"

            else:
                if now - last_kept_time >= current_interval:
                    prev_gray = gray
                    last_kept_time = now
                    current_interval = INITIAL_INTERVAL
                    kept_frames += 1
                    decision = "KEPT (interval hit)"
                else:
                    skipped_frames += 1
                    decision = "SKIPPED"

        # UI updates (throttled)
        if frame_count == 1 or frame_count % 5 == 0:
            logs.append(f"Frame {frame_count}: SSIM={score:.4f} → {decision}")
            logs = logs[-5:]

            log_area.text("\n".join(logs))
            metric_total.metric("Total Frames", frame_count)
            metric_kept.metric("Kept Frames", kept_frames)
            metric_skipped.metric("Skipped Frames", skipped_frames)

            if total_frames_estimate > 0:
                progress_bar.progress(
                    min(frame_count / total_frames_estimate, 1.0)
                )

    cap.release()
    progress_bar.progress(1.0)

    return {
        "total_frames": frame_count,
        "kept_frames": kept_frames,
        "skipped_frames": skipped_frames,
        "logs": logs
    }


def display_results(results):
    st.success(
        f"Processing Complete! "
        f"Kept: {results['kept_frames']} | "
        f"Skipped: {results['skipped_frames']}"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Frames", results["total_frames"])
    c2.metric("Kept Frames", results["kept_frames"])
    c3.metric("Skipped Frames", results["skipped_frames"])


def main():
    st.set_page_config(page_title="Video Frame Processor", layout="wide")
    st.title("📹 Video Frame Processor (Similarity Deduplication)")

    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi"]
    )

    if "results" not in st.session_state:
        st.session_state.results = None
    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    if uploaded_file:
        if uploaded_file.name != st.session_state.last_file:
            st.session_state.results = None
            st.session_state.last_file = uploaded_file.name

        st.divider()

        if st.session_state.results:
            display_results(st.session_state.results)

            if st.button("Process Again"):
                st.session_state.results = None
                st.rerun()

        else:
            if st.button("Start Processing"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name

                with st.spinner("Processing video..."):
                    results = process_video(video_path)

                if results:
                    st.session_state.results = results
                    st.divider()
                    st.write("### Final Results")
                    display_results(results)


if __name__ == "__main__":
    main()
