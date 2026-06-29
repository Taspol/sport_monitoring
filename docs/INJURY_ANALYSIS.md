# Injury-Risk Analysis — References & Implementation

How the `--analyze` mode of `sport_monitoring` detects and estimates ACL
injury-risk for a single basketball player, and the published evidence each rule
is built on.

> **Scope / disclaimer.** This is a **2D single-camera screening proxy** built on
> RTMPose body-17 keypoints. It is designed for *relative flagging and trend
> spotting*, **not** clinical-grade measurement or a calibrated ACL-injury
> probability. True 3D kinetics (ground-reaction force, joint moments) and
> foot/ankle kinematics are not measured. Use it to surface candidates for
> review, not to diagnose.

---

## 1. Why these references — non-contact ACL injury in basketball

Basketball ACL injuries are overwhelmingly **non-contact**, occurring during
deceleration / jump-landing in a position of slight knee flexion + knee valgus.
The three prospective/validation studies below define which *observable*
landing-mechanics raise that risk, and they are the basis for every threshold in
the code.

| Ref | Study | Cohort | What it gives us |
|-----|-------|--------|------------------|
| **Hewett et al., 2005** — *Am J Sports Med* 33(4):492–501 ([link](https://journals.sagepub.com/doi/10.1177/0363546504269591)) | Prospective 3D kinematics/kinetics of a drop-jump | 205 female **basketball**/soccer/volleyball athletes | **Dynamic knee valgus** (knee abduction angle/moment at landing) prospectively predicts non-contact ACL injury (78 % sensitivity, 73 % specificity). → valgus is the principal *modifiable* factor. |
| **Leppänen et al., 2017** — *Am J Sports Med* 45(2):386–393 ([link](https://pubmed.ncbi.nlm.nih.gov/27637264/)) | Prospective vertical drop-jump, 1–3 yr follow-up | 171 female **basketball/floorball** players (15 ACL injuries) | **Stiff landings** raise ACL risk: lower **peak knee flexion** (HR 0.55 per +10°) and higher vGRF (HR 1.26 per +100 N). → low knee flexion at landing is the sport-specific red flag. |
| **Padua et al., 2009** — *Am J Sports Med* 37(10):1996–2002 (JUMP-ACL) ([link](https://pubmed.ncbi.nlm.nih.gov/19726623/)) | Validation of a clinical 2D screen vs. 3D | — | The **Landing Error Scoring System (LESS)**: a 17-item jump-landing checklist; total **LESS ≥ 5/17 = high risk** (≈86 % sensitivity, relative risk ~10.7). → the structure and cut-points for our per-landing score. |

Supporting context (not a threshold source): ACL injury occurs at slight knee
flexion (< ~25–30°) with valgus near initial contact, and the knee abduction
angle increases sharply in the first ~40 ms after contact.

---

## 2. Pipeline overview

```
video frame
   │
   ▼  YOLO26-m + ByteTrack            (detector.py)      → person boxes + tracklet ids
   ▼  FusedTracker / FusedSelected    (tracker.py)       → stable id for the analysed player
   ▼  RTMPose-m (body7, 17 COCO kpts) (rtm_pose.py)      → keypoints (x, y, score)
   ▼  joint_angles()                  (biomechanics.py)  → per-frame 2D angles
   ▼  risk_flags()                    (biomechanics.py)  → per-frame valgus / trunk flags
   ▼  detect_landings() + _score()    (landing.py)       → landings + 9-item LESS-subset
   ▼  assess_player_risk()            (risk.py)          → overall LOW/MOD/HIGH grade
   ▼  outputs: CSVs, risk_summary.txt, pose_trajectory.png (with risk card)
```

The skeleton model is **COCO-17** (RTMPose body7). It has **no foot keypoints**,
so true ankle dorsiflexion is unavailable — the LESS ankle/foot items are
therefore omitted (see §6, "what we drop").

---

## 3. What we measure — 2D joint angles (`biomechanics.py`)

All angles are computed from COCO-17 keypoints; a keypoint is only used if its
confidence ≥ `JOINT_MIN_SCORE` (0.4). Indices: shoulders 5/6, hips 11/12,
knees 13/14, ankles 15/16.

| Angle | Definition | Risk meaning |
|-------|-----------|--------------|
| **Knee flexion** (L/R) | `180° − interior angle(hip, knee, ankle)`; 0 = straight leg | Low at landing = **stiff landing** (Leppänen) |
| **Knee valgus** (L/R) | Signed deviation of the knee from the hip→ankle line, normalised by leg length; **+ = medial collapse** toward the opposite hip | High = **dynamic knee valgus** (Hewett) |
| **Hip flexion** (L/R) | `180° − interior angle(shoulder, hip, knee)` | Low = stiff landing / poor shock absorption |
| **Trunk lean** | Angle of the hip→shoulder vector from vertical, `atan2(\|dx\|, \|dy\|)` (mixes forward + lateral in 2D) | High = excessive lean → raised knee load |

These per-frame angles are written for the analysed player to
`output/<clip>_joint_angles.csv`.

---

## 4. Per-frame risk flags (`biomechanics.py: risk_flags`)

Only angles that are risky **whenever** they occur are flagged every frame.
Knee/hip *flexion* is **not** flagged per frame (a straight leg is normal while
running) — flexion risk is judged only at landings (§5–6).

| Flag | Rule | Config | Basis |
|------|------|--------|-------|
| `knee_valgus_l/r` | valgus `>` 12° | `RISK_KNEE_VALGUS_MAX = 12` | Hewett 2005 |
| `trunk_lean` | lean `>` 30° from vertical | `RISK_TRUNK_LEAN_MAX = 30` | trunk-control literature |

(`RISK_KNEE_FLEX_MIN`, `RISK_HIP_FLEX_MIN` exist for reporting but are applied at
landings, not per frame.)

---

## 5. Landing detection (`landing.py: detect_landings`)

LESS is scored *at a landing*, so we first find jumps as **up-and-down spikes in
the player's body trajectory** (mid-hip `y`, falling back to mid-shoulder). For
each peak we look a fixed number of frames **before and after** it and require the
body to have risen from the ground on the way up *and* dropped back down on the way
down. It is purely shape-based with **no dependency on a global baseline** (a
skewed/noisy signal can't hide a jump). The spike-height threshold is
**scale-invariant**, a fraction of leg length (hip→ankle px).

Algorithm:
1. Interpolate gaps, smooth the body-trajectory and foot-height signals with a
   moving average (`LAND_SMOOTH = 3`).
2. Find every **local maximum** of the trajectory (a candidate apex — highest point
   on screen).
3. For each apex, take a **ballistic window** of samples within `LAND_MAX_AIR_FRAMES`
   (12) **real frames** before and after it (real frame time, not list index, so it
   is robust to dropped-frame tracking gaps). A jump is fast, so this tight window
   rejects slow positional drift — a player running across the court moves their
   mid-hip in the image over many frames, which a wide window would misread as a
   spike. Both sides must contain at least `LAND_MIN_SIDE_FRAMES` (3) samples —
   otherwise the peak sits at a clip edge where the up-down spike can't be seen.
4. Take the lowest the body got on the **left** (ground before takeoff) and on the
   **right** (ground after landing). `rise_up` = apex − left-ground, `rise_down` =
   apex − right-ground.
5. The trajectory must have spiked **up and back down**: **both** `rise_up` and
   `rise_down` ≥ `LAND_MIN_JUMP_RATIO` (0.30) × leg-length. A wobble (too small) or a
   one-sided rise (standing up / a clip cut mid-jump, where one side never drops) is
   rejected.
6. **Airborne check (the decisive jump test)**: within the window the **feet** must
   actually leave the ground — grounded foot level − highest foot level ≥
   `LAND_MIN_AIR_RATIO` (0.20) × leg-length. This rejects non-jump actions
   (squat-and-stand, reaching up, leaning, a perspective bob) where the body centre
   wiggles but the feet stay planted. (Skipped only when the feet are never visible.)
7. **IC (initial contact)** = the first frame after the apex where the **feet**
   return to their grounded level (largest `y` = lowest on screen) — real touchdown
   for LESS scoring.
8. **Absorption phase** = IC → peak knee flexion, searched over the next
   `LAND_ABSORB_FRAMES` (6) frames.

Each landing records `ic_frame`, `apex_frame`, `maxflex_frame`, then is scored.

---

## 6. Per-landing score — automated **LESS subset** (`landing.py: _score`)

We implement the **9 LESS items derivable from 2D body-17 keypoints**. Each
failed item = 1 point (higher = worse), evaluated at **IC** and across the
**absorption phase (IC → peak flexion)**.

| # | Item | Rule (failed if…) | Config | LESS basis |
|---|------|-------------------|--------|------------|
| 1 | `knee_flex_ic` | knee flexion at IC `<` 30° | `LESS_KNEE_FLEX_IC_MIN = 30` | stiff landing (Leppänen) |
| 2 | `hip_flex_ic` | hip flexion at IC `<` 30° | `LESS_HIP_FLEX_IC_MIN = 30` | stiff landing |
| 3 | `trunk_flex_ic` | trunk too upright at IC (`<` 10°) | `LESS_TRUNK_FLEX_IC_MIN = 10` | LESS trunk-flexion item |
| 4 | `lateral_trunk_ic` | trunk lean at IC `>` 30° | `LESS_TRUNK_LEAN_MAX = 30` | LESS lateral-trunk item |
| 5 | `knee_valgus_ic` | knee valgus at IC `>` 12° | `LESS_KNEE_VALGUS_IC_MAX = 12` | Hewett / LESS valgus item |
| 6 | `stance_width` | ankle-sep / shoulder-width `<` 0.8 or `>` 2.2 | `LESS_STANCE_NARROW/WIDE` | LESS stance-width item |
| 7 | `asymmetric` | \|L−R ankle height\| / leg-length `>` 0.15 (one foot lands first) | `LESS_SYMM_MAX = 0.15` | LESS asymmetric-landing item |
| 8 | `knee_flex_disp` | IC→peak knee-flexion increase `<` 45° (doesn't bend to absorb) | `LESS_KNEE_FLEX_DISP_MIN = 45` | LESS knee-flexion-displacement item |
| 9 | `knee_valgus_disp` | valgus increase during absorption `>` 8° (collapses inward) | `LESS_KNEE_VALGUS_DISP_MAX = 8` | LESS / Hewett |

**Per-landing category** (scaled from Padua's validated 17-item cut of ≥5/17):

| Subset score (of 9) | Category | Config |
|---------------------|----------|--------|
| ≤ 1 | good | — |
| 2–3 | moderate | `LESS_SUBSET_MOD_SCORE = 2` |
| ≥ 4 | high-risk landing | `LESS_SUBSET_HIGH_SCORE = 4` |

**What we drop** vs. the full 17-item LESS: foot-position/dorsiflexion items
(no foot keypoints), toe-in/toe-out, and overall-impression — all require 3D or
foot tracking. So this is explicitly a **subset proxy**, not the clinical LESS.

A landing **banner** is drawn on the exported video at each detected landing, and
all landings are written to `output/<clip>_landings.csv`.

---

## 7. Overall per-player risk grade (`risk.py: assess_player_risk`)

The per-frame flags and per-landing scores are aggregated into one
**LOW / MODERATE / HIGH** rating, weighted by the evidence hierarchy
(valgus > landing-LESS > stiff landing > trunk).

**Exposure metrics collected for the analysed player:**
- `valgus_frame_rate` — % of tracked frames flagged with knee valgus.
- `trunk_frame_rate` — % of frames with excessive trunk lean.
- landing metrics — mean/max LESS-subset, count of high-risk landings, count of
  **stiff** landings (items 1/2/8) and **valgus** landings (items 5/9).

**Composite exposure (0–100), for display** — weights mirror evidence priority:

```
composite = 100 × ( 0.40·valgus_exposure        # Hewett — primary
                  + 0.25·mean_LESS/6             # Padua
                  + 0.20·stiff_landing_rate      # Leppänen
                  + 0.15·trunk_exposure )
```

**Evidence-based hard rules** (any HIGH trigger ⇒ HIGH; the rating can only be
raised, never lowered, by the composite fallback):

| Trigger | Rating | Config | Basis |
|---------|--------|--------|-------|
| valgus in ≥ 25 % of frames | **HIGH** | `RISK_VALGUS_FRAME_HIGH = 0.25` | Hewett 2005 |
| any high-risk landing (LESS-subset ≥ 4/9) | **HIGH** | `LESS_SUBSET_HIGH_SCORE` | Padua 2009 |
| ≥ 50 % of landings stiff | **HIGH** | `RISK_STIFF_LANDING_RATE = 0.50` | Leppänen 2017 |
| ≥ 50 % of landings with valgus collapse | **HIGH** | `RISK_VALGUS_LANDING_RATE = 0.50` | Hewett |
| valgus in ≥ 10 % of frames | MODERATE | `RISK_VALGUS_FRAME_MOD = 0.10` | — |
| mean landing LESS-subset ≥ 2 | MODERATE | `LESS_SUBSET_MOD_SCORE = 2` | — |
| trunk lean in ≥ 15 % of frames | MODERATE | `RISK_TRUNK_FRAME_MOD = 0.15` | — |
| composite ≥ 60 / ≥ 30 | HIGH / MODERATE | `RISK_COMPOSITE_HIGH/MOD` | fallback |

Every trigger appends a **cited reason** to the report, e.g.
*"Dynamic knee valgus in 31 % of frames (Hewett 2005: valgus is the principal
ACL predictor)."*

---

## 8. Fall / on-ground detection (head-to-toe height) — `fall.py`

Separate from the chronic ACL screen, we flag **acute fall / on-ground events**
for the analysed player by measuring their **head-to-toe height** over time. The
intuition: a fall makes a person *shorter* — when they go down, the head, hips and
feet collapse toward one height, so the measured standing height drops far below
the player's own normal.

**Per-frame metric** (`fall_metrics`) — the head-to-toe height:

| Metric | How | Standing | Fallen |
|--------|-----|----------|--------|
| **Height** (primary) | gravity-axis span head→feet, distance-normalised: `(ankle_y − head_y) / body_length` | ~1.0+ (full height) | ~0 (head, hips, feet at one height) |
| **Upper-body drop** (temporal) | how far the head+hip centre fell over the previous `FALL_DROP_WINDOW` frames, ÷ body length | ~0 | large (rapid descent) |
| **Cloud orientation** (fallback) | PCA major-axis angle + bbox aspect, used only when no height is measurable | ~0–15°, tall | ~70–90°, wide |

Dividing by skeletal body length makes the height **distance-invariant** (a player
far from the camera is small in pixels but still ~1.0). `detect_falls` then takes
the player's **own normal** height = a high percentile (`FALL_BASELINE_PCTL = 80`)
of their height series, and flags a frame **on-ground** when
`height < FALL_HEIGHT_DROP_RATIO (0.55) × normal`. Because it compares to *their
own* normal, a deep squat or toe-touch (where the **head stays high**, so height
barely changes) does not false-trigger — only an actual collapse does.

**Event confirmation** (`detect_falls`): consecutive on-ground frames are grouped
(tolerating short `FALL_GAP_FRAMES` gaps from missed detections) and only kept if
the run lasts ≥ `FALL_MIN_FRAMES` (debounce — a 1–2 frame dive/stumble is ignored).
A fall is tagged **sudden** if the body centroid dropped ≥ `FALL_DROP_RATIO ×
body-length` within the preceding `FALL_DROP_WINDOW` frames (distinguishing a real
fall from someone who was already low, e.g. sitting).

**Config:** `FALL_BASELINE_PCTL`, `FALL_HEIGHT_DROP_RATIO`, `FALL_VERTICALITY_MAX`
(absolute fallback when no baseline forms), `FALL_AXIS_ANGLE_MIN`,
`FALL_TRUNK_ANGLE_MIN`, `FALL_ASPECT_MAX`, `FALL_MIN_ELONGATION`,
`FALL_MIN_KEYPOINTS`, `FALL_MIN_FRAMES`, `FALL_GAP_FRAMES`, `FALL_DROP_WINDOW`,
`FALL_DROP_RATIO`.

**Surfaced as:** a red border + `FALL / ON GROUND` banner on `_injury.mp4` for the
event span, a `Falls (on-ground): N` line in the risk card/summary, and a
`<clip>_falls.csv` (start/end/lowest frame, duration, confidence, sudden flag).
Falls are reported as **acute events alongside** the ACL grade — they do **not**
alter the LOW/MOD/HIGH ACL rating (that stays a pure landing-mechanics screen).

> Like the rest of this module it is a **2D single-camera heuristic** — a relative
> flag, not a certified fall detector.

---

## 9. Outputs (analyze mode)

| File | Contents |
|------|----------|
| `<clip>_joint_angles.csv` | per-frame 2D angles for the analysed player |
| `<clip>_jumps.csv` | each jump: apex/IC/max-flex frames + 9 LESS items + score |
| `<clip>_risk_summary.txt` | the graded risk report with cited reasons + disclaimer |
| `<clip>_pose_track.csv` | per-frame 17 COCO keypoints (x, y, score) |
| `<clip>_pose_trajectory.png` | full stroboscopic skeleton trail: viridis **old->new gradient** (normal posture) + injury-risk-flagged poses **red** + per-pose orange move-trend **arrows** + colour-coded **risk card** |
| `<clip>_pose_trajectory_risk.png` | **risk-only** view of the same trail: the viridis gradient (normal posture) is cut out, leaving **only the red risk poses**. Consecutive flagged frames (within `RISK_CLUSTER_GAP`) are collapsed to **one representative pose per risk moment** so distinct moments are visible instead of an over-plotted blob. Falls back to the full trail when the clip has **no** risk frames (so it is never blank). |
| `<clip>_trajectory_map.png` | the analysed player's movement path + direction **arrow** |
| `<clip>_falls.csv` | each fall: start/end/lowest frame, duration, confidence, sudden flag |
| `<clip>_injury.mp4` | annotated video: skeleton, joint-angle panel, landing + **fall** banners |
| `<clip>_clean.mp4` | clean video: only boxes + skeletons + **ID labels**, tracked player highlighted |

---

## 10. Known limitations

- **2D only.** No ground-reaction force or joint moments (the *kinetic*
  predictors in Hewett/Leppänen). Valgus is estimated as a 2D frontal-plane
  deviation, sensitive to camera angle.
- **No foot/ankle.** body-17 lacks foot keypoints → no dorsiflexion, and ankle
  injuries (the most common in basketball) are not modelled. A wholebody/feet
  RTMPose model would be needed.
- **Thresholds are screening cut-points, not calibrated probabilities.** Even the
  source studies note these measures are better for group-level association than
  individual screening (Leppänen's ROC analysis).
- **Single-camera, single-view** valgus/trunk readings degrade when the player
  is side-on, partially occluded, or far from the camera.

- **Fall detection** is posture-geometry only: a player fully occluded behind
  others, or whose torso keypoints drop below `FALL_MIN_KEYPOINTS`, can't be
  judged; a steep camera angle can also distort the cloud aspect.

---

## 11. Where to tune

All thresholds live in `config.py`:
- Per-frame flags: `RISK_KNEE_VALGUS_MAX`, `RISK_TRUNK_LEAN_MAX`, …
- Landing detection: `LAND_*`
- LESS-subset cut-offs: `LESS_*`
- Per-player aggregation: `RISK_VALGUS_FRAME_*`, `RISK_*_LANDING_RATE`,
  `RISK_COMPOSITE_*`, `LESS_SUBSET_*`
- Fall detection: `FALL_*`

Implementation: `biomechanics.py` (angles + flags), `landing.py` (detection +
LESS subset), `fall.py` (fall / on-ground detection), `risk.py` (aggregation),
`visualize.py` (overlays/card/banners), `pipeline.py: process_video_analyze`
(orchestration).

---

## 12. Evaluation

The risk grade is a funnel — `track -> pose -> detect landings -> LESS -> grade` —
so the two numbers that actually decide the output are **landing-event detection**
and **LESS agreement**. We evaluate those end-to-end against a small hand-labelled
golden set; upstream stages are validated implicitly by their effect on these.

**Two gates:**
- *Unit gate (fast, every change):* the synthetic `detect_landings` cases in the
  test suite (clean / raised-baseline / two-jumps / noisy / planted / drift / …).
- *Integration gate (golden clips):* `benchmark/labels.csv` holds hand-marked
  landing frames + a human LESS score per landing. `scripts/evaluate.py` reads the
  `output/<clip>_jumps.csv` that `--analyze` already wrote (no pipeline re-run, runs
  in <1 s) and reports:
  - **Landing detection** — precision / recall / F1 with an IC frame tolerance
    (`--tol`, default ±5), greedy nearest matching, plus mean |IC frame error|.
    This is the over/under-count check.
  - **LESS agreement** — MAE, signed bias, Pearson r and ICC(2,1) over matched
    landings.

```
uv run python scripts/evaluate.py            # all labelled clips
uv run python scripts/evaluate.py --tol 4    # stricter frame tolerance
```

### 12.1 Backbone comparison (detector × pose)

The pipeline factors into a **detector** (person boxes + ByteTrack ids) and a
top-down **pose** estimator (17 COCO keypoints/box), each behind a tiny interface
(`backbones.py`), so the four combinations can be A/B'd on the same golden clips:

| detector \ pose | `rtm` (RTMPose) | `mediapipe` (BlazePose→COCO-17) |
|---|---|---|
| `yolo` (YOLO + ByteTrack) | default | `--detector yolo --pose mediapipe` |
| `rfdetr` (RF-DETR + supervision ByteTrack) | `--detector rfdetr --pose rtm` | `--detector rfdetr --pose mediapipe` |

ByteTrack (association) and the player-selection step are held **constant** across
detectors, so a row isolates the detector/pose model, not the bookkeeping. BlazePose's
33 landmarks are remapped to COCO-17 (`pose_mediapipe._BLAZE_TO_COCO`) so every
downstream LESS/angle calc is identical regardless of backend. The heavy packages
(`mediapipe`, `rfdetr`, `supervision`) import lazily and live in the optional
`benchmark` extra — `uv sync --extra benchmark` — so the default path needs none.

Non-default variants write to `output/<detector>_<pose>/` (so they don't clobber the
canonical `yolo_rtm` outputs in `output/`), and each pass drops a `<clip>_timing.json`
for the FPS column. `--auto-select` skips the click window — it picks the jumper headlessly
as the track with the widest vertical foot-line excursion
(`selector.auto_select_player`) — so the four passes run unattended. Run all four,
then compare:

```
uv sync --extra benchmark
uv run sport-monitoring --analyze data/basket_S1T3_pre.mp4 --no-video --auto-select                         # yolo_rtm  -> output/
uv run sport-monitoring --analyze data/basket_S1T3_pre.mp4 --no-video --auto-select --pose mediapipe         # -> output/yolo_mediapipe/
uv run sport-monitoring --analyze data/basket_S1T3_pre.mp4 --no-video --auto-select --detector rfdetr         # -> output/rfdetr_rtm/
uv run sport-monitoring --analyze data/basket_S1T3_pre.mp4 --no-video --auto-select --detector rfdetr --pose mediapipe
uv run python scripts/evaluate.py --variants yolo_rtm,yolo_mediapipe,rfdetr_rtm,rfdetr_mediapipe
```

One row per variant: detection F1/P/R, mean |IC frame error|, LESS MAE/bias/r/ICC,
and analyse FPS — i.e. accuracy **and** speed on the same clips.

### 12.2 Results so far (`basket_S1T3_pre`, 2 landings)

| variant | F1 | P | R | IC err | LESS MAE | bias | r | ICC | FPS |
|---|---|---|---|---|---|---|---|---|---|
| **yolo_rtm** (default) | **1.00** | 1.00 | 1.00 | **0.0** | **0.00** | 0.00 | 1.00 | 1.00 | 1.1 |
| rfdetr_rtm | **1.00** | 1.00 | 1.00 | 0.5 | **0.00** | 0.00 | 1.00 | 1.00 | 0.9 |
| yolo_mediapipe | 1.00 | 1.00 | 1.00 | 3.0 | 1.00 | −1.00 | n/a* | n/a* | 1.1 |
| rfdetr_mediapipe | 0.00 | 0.00 | 0.00 | n/a | n/a | n/a | n/a | n/a | 1.1 |

\* `r`/`ICC` are degenerate at n=2 (reported as −1.00); read MAE/bias instead.

**Read-out (on this one clip — directional, not yet statistically significant):**
- **Pose is the deciding axis.** Both RTMPose variants nail detection *and* LESS
  (MAE 0.00, IC error ≤0.5 frame). The detector barely matters under RTMPose: RF-DETR
  ties YOLO on accuracy, ~15 % slower.
- **MediaPipe degrades the pose** on these small, far figures: `yolo_mediapipe` still
  finds both landings but LESS drifts ~1 point (under-scores) and IC timing slips 3
  frames; `rfdetr_mediapipe` loses the jump signal entirely (0 detections).
- **Speed** is ~1 FPS for all on CPU — pose dominates and MediaPipe-per-crop ≈ RTMPose,
  so MediaPipe buys no speed while costing accuracy.
- **Verdict:** keep **YOLO + RTMPose**; RF-DETR + RTMPose is a viable equal-accuracy
  swap. This is **1 clip / 2 landings** — label S1T1/S1T2/outdoor1_fall in
  `benchmark/labels.csv` and re-run before trusting the LESS correlation numbers.

### 12.3 Metric definitions (for the paper's evaluation section)

Notation: each clip yields a set of predicted initial-contact frames
$\{\hat t_i\}$ with LESS scores $\{\hat s_i\}$, and hand labels $\{t_j\}$ with expert
LESS scores $\{s_j\}$. A prediction matches a label when $|\hat t_i - t_j| \le \tau$
(tolerance $\tau$, default 5 frames); matching is greedy nearest-first and one-to-one
(every prediction and every label is used at most once). Matched pairs are true
positives (**TP**); unmatched predictions are false positives (**FP**, the system
invented a landing); unmatched labels are false negatives (**FN**, a real landing was
missed).

**Landing-detection metrics** — *does the system find the right events, no more, no
fewer?*

| metric | formula | meaning | range / target |
|---|---|---|---|
| Precision $P$ | $\dfrac{TP}{TP+FP}$ | of the landings flagged, the fraction that are real (purity; low $P$ = false alarms) | 0–1, ↑ |
| Recall $R$ | $\dfrac{TP}{TP+FN}$ | of the real landings, the fraction caught (coverage; low $R$ = misses) | 0–1, ↑ |
| $F_1$ | $\dfrac{2PR}{P+R}$ | harmonic mean of $P$ and $R$; single balanced summary | 0–1, ↑ |
| IC error | $\dfrac{1}{|TP|}\sum_{(i,j)\in TP}\lvert \hat t_i - t_j\rvert$ | mean **temporal** localization error of the contact instant, in frames (÷ video fps → seconds; e.g. 3 f @30 fps = 100 ms). Defined over matches only | frames, ↓ (0 = exact) |

**LESS-agreement metrics** — *given a correctly found landing, is its 0–9 risk score
the one an expert would give?* Computed over the $n$ matched landings that also carry
an expert score.

| metric | formula | meaning | range / target |
|---|---|---|---|
| MAE | $\dfrac1n\sum\lvert \hat s_i - s_j\rvert$ | average magnitude of disagreement, in LESS points | ≥0, ↓ (0 = identical) |
| Bias | $\dfrac1n\sum(\hat s_i - s_j)$ | mean *signed* error: **+** = system over-scores risk, **−** = under-scores. Separates a systematic offset from random scatter | ≈0 target |
| Pearson $r$ | $\dfrac{\sum(\hat s_i-\bar{\hat s})(s_j-\bar s)}{\sqrt{\sum(\hat s_i-\bar{\hat s})^2\sum(s_j-\bar s)^2}}$ | linear **co-variation**: do the two scores rank landings the same way? Insensitive to a constant offset; undefined if either side is constant | −1…1, ↑ |
| ICC(2,1) | two-way random-effects, single rater, absolute agreement | **agreement** treating system and expert as interchangeable raters; unlike $r$ it *penalizes* systematic bias and scale differences. Conventional bands: <0.5 poor, 0.5–0.75 moderate, 0.75–0.9 good, >0.9 excellent | −1…1, ↑ |

Report **both** $r$ and ICC: $r$ answers "do they move together?", ICC answers "do
they actually agree?" Two raters can correlate perfectly ($r=1$) yet disagree (one
always reads +2) — ICC catches that, $r$ does not.

**Throughput** — **FPS** $= \text{frames} / \text{wall-clock seconds}$ for the analyse
pass (single CPU thread; from `<clip>_timing.json`, covers detection + pose + scoring).
Reported as a first-class axis because CPU deployment is latency-bound and pose
estimation dominates cost.

**Power / how to phrase it.** These are *agreement* statistics on a small golden set,
so always state $n$ (matched landings) and the clip count. With very small $n$ (e.g.
$n=2$), Pearson $r$ and ICC are **degenerate** — two points are collinear, forcing
$r=\pm1$ — and the harness prints `n/a`; report MAE and bias instead and treat the
single-clip figures as a sanity/agreement check, not a validation. For any published
reliability claim, pool matched landings across multiple labelled clips (target
$n \ge 15$–20) before reporting $r$/ICC, and quote the $\tau$ used.

**Scope (honest):** this measures *agreement* with hand labels + a human LESS read
("are the right events detected and scored like an expert would"), **not** injury
prediction — that needs a prospective cohort with follow-up (out of scope; see §10).
```
```
