# Web demo — basketball injury-risk analysis

A [Gradio](https://www.gradio.app/) front-end for the analyze pipeline. Upload a
clip, click the player you want, and get the annotated video + risk summary +
landing strobes + per-frame CSVs in the browser. It wraps the **same**
`process_video_analyze` used by the CLI — no analysis logic is duplicated here,
and the command-line tool is unaffected.

## Setup

From the project root (needs the project installed + system **ffmpeg** for the
browser-friendly H.264 re-encode of the output video):

```bash
uv sync --extra web        # adds gradio to the existing env
# ffmpeg: macOS `brew install ffmpeg`  ·  Debian/Ubuntu `apt install ffmpeg`
```

## Run

```bash
uv run python demo_web/app.py
```

Open the printed local URL (default <http://127.0.0.1:7860>).

## How to use

1. **Upload** a video, **or** pick a bundled clip from the `data/` dropdown and
   click **Use**. Frame 0 is detected automatically.
2. **Pick a reference frame** with the slider (start just before the jump), then
   **Load frame**. Candidate players are drawn with numbered boxes.
3. **Click** the player to analyse — the chosen box turns green.
4. *(optional)* Open **Advanced options** for `--3d`, or to switch the detector /
   pose backbone (`rfdetr` / `mediapipe` need `uv sync --extra benchmark`).
5. **Run analysis**. Processing is CPU-bound and can take a few minutes; a
   progress bar tracks it, and **Cancel** stops the current run. The annotated
   video keeps the skeleton/box overlays but omits the movement trail/arrows for
   a cleaner view. Results appear in the tabs below, with a colour-coded risk
   summary.

## Notes

- **Concurrency:** analysis runs one at a time (`concurrency_limit=1`) since it
  saturates the CPU.
- **Artifacts** are written to `demo_web/work/<run-id>/` (gitignored).
- **No ffmpeg?** The result still renders but uses the `mp4v` codec, which some
  browsers won't play — install ffmpeg for in-browser playback.

## Files

| File | Role |
|------|------|
| `app.py` | Gradio UI + event wiring |
| `bridge.py` | glue to `sport_monitoring` (frame detect, click→box, run, re-encode) |
| `work/` | per-session uploads + outputs (gitignored) |
