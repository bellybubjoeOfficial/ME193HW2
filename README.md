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
- `KP`, `KI`, `KD` - PID gains on the horizontal pixel error. Start with `KP`
  only (leave `KI`/`KD` at 0), increase until the car responds promptly
  without overshooting badly, then add a little `KD` to damp oscillation,
  and only add `KI` if there's a persistent steady-state offset.
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
- Connecting to the Double Motor over BLE is synchronous/blocking (same as
  the main ME193 project), but each PID-loop's motor command is sent with
  `blocking=False` so the vision loop isn't paced by Bluetooth round-trip
  latency.
