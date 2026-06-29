"""Gradio web demo for basketball injury-risk analysis.

Upload a clip -> pick a reference frame -> click the player to analyse -> run the
(unchanged) ``process_video_analyze`` pipeline -> view the annotated video, risk
summary, landing strobes and per-frame CSVs.

Run with:  ``uv run python demo_web/app.py``  (after ``uv sync --extra web``).
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import gradio as gr

import bridge

WORK_ROOT = Path(__file__).parent / "work"
DETECTORS = ["yolo", "rfdetr"]
POSES = ["rtm", "mediapipe"]


def on_video(video_path: str | None):
    """When a clip is uploaded: size the frame slider and auto-load frame 0."""
    if not video_path:
        return (gr.update(maximum=0, value=0), None, None, [], None,
                "Upload a video to begin.")
    n = bridge.frame_count(video_path)
    rgb = bridge.read_frame_rgb(video_path, 0)
    annotated, boxes = bridge.detect_on_frame(video_path, 0, bridge.DEFAULT_CONF)
    status = (f"Detected {len(boxes)} player(s) on frame 0. "
              "Click one to select, or scrub to another frame.")
    return (gr.update(maximum=max(1, n - 1), value=0), rgb, annotated, boxes,
            None, status)


def load_frame(video_path: str | None, frame_idx: int, conf: float):
    """Detect on the chosen reference frame and reset the selection."""
    if not video_path:
        raise gr.Error("Upload a video first.")
    rgb = bridge.read_frame_rgb(video_path, int(frame_idx))
    annotated, boxes = bridge.detect_on_frame(video_path, int(frame_idx), conf)
    status = (f"Detected {len(boxes)} player(s) on frame {int(frame_idx)}. "
              "Click one to select.")
    return rgb, annotated, boxes, None, status


def on_select(raw_rgb, boxes, selected, evt: gr.SelectData):
    """Map a click on the frame to a detection box and highlight it."""
    if raw_rgb is None or not boxes:
        return gr.update(), selected, "Load a frame first."
    x, y = evt.index  # (x, y) pixel coordinates within the image
    idx = bridge.pick_box(boxes, int(x), int(y))
    annotated = bridge.render_boxes(raw_rgb, boxes, idx)
    return annotated, idx, f"Selected player #{idx}. Click **Run analysis**."


def run(
    video_path: str | None,
    frame_idx: int,
    conf: float,
    three_d: bool,
    detector_name: str,
    pose_name: str,
    boxes,
    selected,
    progress=gr.Progress(track_tqdm=True),
):
    """Run the analyze pipeline on the selected player and gather artifacts."""
    if not video_path:
        raise gr.Error("Upload a video first.")
    if not boxes or selected is None:
        raise gr.Error("Click a player on the frame to select them first.")

    progress(0.0, desc="Starting analysis (this can take a while on CPU)...")
    work_dir = WORK_ROOT / uuid4().hex
    try:
        res = bridge.run_analysis(
            video_path=video_path,
            target_box=boxes[selected],
            frame_idx=int(frame_idx),
            confidence=conf,
            three_d=three_d,
            detector_name=detector_name,
            pose_name=pose_name,
            work_dir=work_dir,
        )
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc

    gallery = [str(p) for p in res.strobes]
    if res.trajectory_map:
        gallery.append(str(res.trajectory_map))
    downloads = [str(p) for p in (res.jumps_csv, res.falls_csv,
                                  res.joint_angles_csv) if p]
    video_out = str(res.video) if res.video else None
    return video_out, res.summary, gallery, downloads


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Basketball Injury-Risk Analysis") as demo:
        gr.Markdown(
            "# 🏀 Basketball Injury-Risk Analysis\n"
            "Upload a clip, pick a reference frame, **click the player** to "
            "analyse, then run the LESS-inspired landing analysis."
        )

        # Per-session state.
        raw_rgb = gr.State(None)     # current frame, no overlay (for re-draw)
        boxes_state = gr.State([])   # detection boxes on the current frame
        selected_state = gr.State(None)  # index of the chosen box

        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="1. Upload video", sources=["upload"])
                with gr.Row():
                    sample_dd = gr.Dropdown(
                        choices=bridge.list_sample_videos(), value=None,
                        label="…or pick a test video from data/", scale=3)
                    use_sample_btn = gr.Button("Use", scale=1)
                frame_slider = gr.Slider(
                    0, 1, value=0, step=1, label="2. Reference frame")
                conf_slider = gr.Slider(
                    0.05, 0.9, value=bridge.DEFAULT_CONF, step=0.05,
                    label="Detection confidence")
                load_btn = gr.Button("Load frame", variant="secondary")
                with gr.Accordion("Advanced options", open=False):
                    three_d_chk = gr.Checkbox(
                        value=False, label="3D cuboids + live move arrows (--3d)")
                    detector_dd = gr.Dropdown(
                        DETECTORS, value="yolo", label="Detector backbone")
                    pose_dd = gr.Dropdown(
                        POSES, value="rtm", label="Pose backbone")
                with gr.Row():
                    run_btn = gr.Button("Run analysis", variant="primary")
                    cancel_btn = gr.Button("Cancel", variant="stop")

            with gr.Column(scale=2):
                frame_img = gr.Image(
                    label="3. Click the player to analyse",
                    interactive=False, type="numpy")
                status = gr.Markdown("Upload a video to begin.")

        gr.Markdown("## Results")
        with gr.Tabs():
            with gr.Tab("Annotated video"):
                video_out = gr.Video(label="Annotated result")
            with gr.Tab("Risk summary"):
                summary_out = gr.Markdown(
                    "_Run an analysis to see the risk summary._")
            with gr.Tab("Landing strobes / path"):
                gallery_out = gr.Gallery(label="Strobes & trajectory", columns=2)
            with gr.Tab("Data (CSV)"):
                files_out = gr.File(
                    label="jumps / falls / joint angles", file_count="multiple")

        # --- wiring ---
        video_outputs = [frame_slider, raw_rgb, frame_img, boxes_state,
                         selected_state, status]
        video_in.change(on_video, [video_in], video_outputs)
        # Load a bundled test clip into the uploader, then run the same detect step.
        use_sample_btn.click(
            lambda p: p, [sample_dd], [video_in],
        ).then(on_video, [video_in], video_outputs)
        load_btn.click(
            load_frame, [video_in, frame_slider, conf_slider],
            [raw_rgb, frame_img, boxes_state, selected_state, status],
        )
        frame_img.select(
            on_select, [raw_rgb, boxes_state, selected_state],
            [frame_img, selected_state, status],
        )
        run_event = run_btn.click(
            run,
            [video_in, frame_slider, conf_slider, three_d_chk, detector_dd,
             pose_dd, boxes_state, selected_state],
            [video_out, summary_out, gallery_out, files_out],
            concurrency_limit=1,  # CPU-heavy: one analysis at a time
        )
        cancel_btn.click(None, None, None, cancels=[run_event])

    return demo


if __name__ == "__main__":
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    build_app().queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )
