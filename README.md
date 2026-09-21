# ME193 HW2 - AprilTag Centering with PID

Webcam sees an AprilTag mounted on top of the LEGO car. A PID loop drives the
car forward/backward (both wheels together - no steering) until the tag is
horizontally centered in frame. Vertical position doesn't matter - the
"centered" zone is drawn as a band spanning the full frame height.

## Setup

This folder lives under a path containing a `:` (`MATLAB:Code For Tufts`),
and Python's `venv` refuses to create an environment inside a path with a
colon in it (colon is the `PATH` separator). So the virtual environment for
this project lives outside that path, in your home directory:

```bash
python3 -m venv ~/.venvs/me193hw2
source ~/.venvs/me193hw2/bin/activate
pip install -r requirements.txt
```

Activate that same venv any time before running scripts here.

## Run

```bash
source ~/.venvs/me193hw2/bin/activate
python3 main.py
```

Press `q` in the video window to quit. The car stops and disconnects
cleanly on exit (including Ctrl+C).

## Tuning

Constants at the top of `main.py`:

- `CARD_COLOR` / `CARD_SERIAL` - must match your Double Motor's printed
  Connection Card. Defaults to the same card used in the main ME193 project
  (`pose-race-193`) - update if this is a different physical hub.
- `KP`, `KI` - PI gains on the horizontal pixel error (no `KD` - see Design
  notes). Start with `KP` only (leave `KI` at 0), increase until the car
  responds promptly without overshooting badly, and only add `KI` if
  there's a persistent steady-state offset.
- `SMOOTHING_ALPHA` - EMA smoothing on the tag's detected position (0-1;
  lower = smoother but laggier). Cuts frame-to-frame pixel jitter before it
  reaches the controller.
- `MAX_ACCEL` - max speed change per second; ramps the commanded speed
  smoothly instead of snapping between values every frame.
- `MAX_SPEED` - caps PID output to a motor speed range. Starts at 25 (out of
  a possible 100) as a safety margin while tuning; raise once gains feel
  stable.
- `DEADBAND_PX` - how many pixels of error count as "close enough" (stops
  the motors instead of jittering around the setpoint).
- `INVERT_DIRECTION` - flip to `True` if the car drives away from center
  instead of toward it (means the camera/tag mounting is oriented opposite
  to what the script assumes).

## AprilTag family

Detection uses OpenCV's built-in `aruco` module (see
`apriltag_detector.py`), defaulting to the `36h11` family - the one printed
by the standard AprilTag generator. If your tag was printed from a
different family (16h5, 25h9, etc.), change `DEFAULT_DICTIONARY` in
`apriltag_detector.py`.

## Design notes

- Only the tag's horizontal (x) position is used for control - vertical (y)
  position doesn't matter, so the on-screen "centered" zone is drawn as a
  vertical band spanning the full frame height rather than a single point.
- No `KD` (derivative) term: with a noisy pixel-position signal, it mostly
  amplified frame-to-frame jitter into spurious speed spikes rather than
  damping real oscillation. Input smoothing (`SMOOTHING_ALPHA`) handles
  noise instead, and `MAX_ACCEL` (a slew-rate limiter) keeps every speed
  transition - including stopping in the deadband - a smooth ramp.
- Connecting to the Double Motor over BLE is synchronous/blocking (same as
  the main ME193 project), but each PID-loop's motor command is sent with
  `blocking=False` so the vision loop isn't paced by Bluetooth round-trip
  latency.

## iPhone (Continuity Camera) variant

`main_iphone.py` is the same control scheme, but the camera source is an
iPhone mounted sideways on the car (facing perpendicular to its driving
direction) instead of a stationary webcam, streaming to this Mac via
Continuity Camera, watching a stationary AprilTag placed near/on this
computer. As the car drives, the moving camera sweeps past the stationary
tag - the same kind of horizontal parallax shift as the original
stationary-camera/moving-tag setup, just with the roles swapped - so the
same horizontal-centering logic applies unchanged.

Setup specific to this variant:

- **Continuity Camera prerequisites**: iPhone on iOS 16+, signed into the
  same Apple ID as this Mac, Wi-Fi + Bluetooth on for both, Continuity
  Camera enabled (System Settings -> General -> AirDrop & Handoff). Verify
  it works in Photo Booth or FaceTime before running the script.
- **Find the camera index**: run `python3 list_cameras.py` - it previews
  each available camera index so you can identify which one is the iPhone,
  then set `CAMERA_INDEX` in `main_iphone.py` to that number. This isn't
  guaranteed stable across machines/reboots, so re-check if things stop
  working after a restart.
- **`INVERT_DIRECTION` needs re-tuning** - don't assume `main.py`'s tuned
  value carries over; the moving-camera/stationary-tag arrangement can flip
  the apparent parallax direction relative to the original setup.
- **Wi-Fi dropouts**: Continuity Camera streams over Wi-Fi and can drop
  frames. A failed frame read stops the motors and retries rather than
  driving blind; `MAX_READ_FAILURES` consecutive failures exits the script
  entirely.

## Celebration animation

Once the tag has stayed centered for `COMPLETE_HOLD_FRAMES` consecutive
frames (in both `main.py` and `main_iphone.py`), the run is considered
complete: the motors stop, and `celebration.run_celebration()` plays a
"shatter into confetti" animation using `CELEBRATION_PHOTO` (defaults to
`ChrisRogers.png`) before the script exits.

No generative image model is involved - `celebration.py` draws a party hat
as vector shapes (positioned via a simple background-threshold heuristic,
since OpenCV 5.x dropped the bundled Haar cascade face detector this
project would otherwise have used), pads the canvas with matching
background color if the source photo doesn't have enough headroom above
the hair, then slices the resulting image into a grid of tiles that fall
with simple gravity/velocity physics, mixed with small drawn confetti
pieces that tumble independently.

To use a different photo, swap `CELEBRATION_PHOTO` to another image path -
works best with a similar tightly-cropped headshot on a plain, light,
roughly-uniform background (the head-detection heuristic relies on that
contrast).
