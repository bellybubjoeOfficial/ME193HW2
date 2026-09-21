# Entry point: iPhone (Continuity Camera) -> AprilTag detection -> PID -> Double Motor.
#
# Same control scheme as main.py, but the camera is now an iPhone mounted
# sideways on the car (facing perpendicular to its driving direction),
# streaming to this Mac over Continuity Camera, watching a stationary
# AprilTag placed near/on this computer. As the car drives forward/backward,
# the moving camera sweeps past the stationary tag - the same kind of
# horizontal parallax shift as the original setup's stationary-camera/
# moving-tag arrangement, just with the roles swapped. So the tag's
# horizontal (x) pixel position is still the control signal; vertical (y)
# position still doesn't matter.
#
# Continuity Camera adds two things main.py didn't need to handle:
#   - The right camera device index isn't fixed - run list_cameras.py to
#     find it, then set CAMERA_INDEX below.
#   - It streams over Wi-Fi, so occasional dropped frames are expected;
#     brief read failures stop the motors and retry rather than crashing.

import time
from collections import deque

import cv2
import legoeducation as le
import numpy as np

import celebration
from apriltag_detector import AprilTagDetector
from pid import PID

# --- Camera device index for the iPhone via Continuity Camera - run
# list_cameras.py to find the right value for your machine. ---
CAMERA_INDEX = 2

# If the camera read fails this many times in a row (e.g. a Wi-Fi hiccup),
# stop trying and exit rather than looping forever with no video.
MAX_READ_FAILURES = 30

# --- LEGO connection card - update to match your Double Motor's card ---
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = '5164'

# --- PID gains - start conservative and tune from here ---
KP = 0.15
KI = 0.09
KD = 0.04

# Conservative output cap while tuning, independent of KP - this is the
# actual ceiling on how fast the car can ever go, regardless of how far off
# center the tag is.
MAX_SPEED = 25.0

# Pixel error inside this band counts as "centered" - stops the motors
# instead of jittering around the setpoint.
DEADBAND_PX = 15.0

# EMA smoothing on the tag's detected horizontal position (0-1: higher =
# less smoothing, more responsive; lower = smoother, laggier). Removes
# frame-to-frame pixel jitter - and Continuity Camera's added latency makes
# this arguably even more useful here than in main.py.
SMOOTHING_ALPHA = 0.3

# Max speed change allowed per second - ramps the commanded speed smoothly
# instead of snapping to a new value every frame. This also smooths the
# transition into/out of the deadband stop.
MAX_ACCEL = 80.0

# The camera moving past a stationary tag can shift the apparent parallax
# direction relative to the original stationary-camera setup - re-test this
# rather than assuming main.py's tuned value carries over. Flip to True if
# the car drives away from center instead of toward it.
INVERT_DIRECTION = True

# If no tag is seen for this many consecutive frames, stop the motors.
MAX_MISSED_FRAMES = 10

# Once the tag has stayed centered for this many consecutive frames, the
# run is considered complete: stop, play the celebration animation, and exit.
CELEBRATION_PHOTO = 'ChrisRogers.png'
COMPLETE_HOLD_FRAMES = 45

# Temporarily disabled while live-tuning gains via the sliders below - the
# run shouldn't stop/exit just because the tag centered mid-tuning. Flip
# back to True to restore the objective-complete stop-and-exit behavior.
ENABLE_COMPLETION_STOP = False

# Temporarily disabled (see ENABLE_COMPLETION_STOP above, which already
# skips this when off) - the photo popup interrupts each tuning run. Flip
# back to True to restore it.
ENABLE_CELEBRATION = False

WINDOW_NAME = 'AprilTag Centering (iPhone)'

# How many past frames the live PID-term graph shows at once.
GRAPH_HISTORY_FRAMES = 150

# Pixel size of the live PID-term graph, drawn in the frame's top-right corner.
GRAPH_SIZE = (240, 110)

# BGR colors for the P/I/D contribution lines on the graph.
GRAPH_COLORS = {'P': (0, 165, 255), 'I': (0, 220, 0), 'D': (255, 0, 255)}

# Live PID-gain sliders, attached to the main window: each trackbar position
# is the gain value * SLIDER_SCALE (OpenCV trackbars are integer-only), so
# e.g. a KP trackbar position of 40 with SLIDER_SCALE=100 means KP=0.40. The
# *_SLIDER_MAX values are just generous tuning headroom, not hard limits.
SLIDER_SCALE = 100.0
KP_SLIDER_MAX = 300
KI_SLIDER_MAX = 100
KD_SLIDER_MAX = 100


def _status_label(detection, missed_frames):
	"""Human-readable detection state for the HUD: tag found, coasting
	through a brief dropout (see main_iphone.py's missed-detection handling),
	or given up and stopped."""
	if detection is not None:
		return 'TAG DETECTED'
	if missed_frames < MAX_MISSED_FRAMES:
		return f'GRACE ({missed_frames}/{MAX_MISSED_FRAMES})'
	return 'NO TAG - STOPPED'


def _draw_hud(frame, speed, error, status_label, kp, ki, kd):
	"""Top-left text block: net speed, pixel error, detection state, and the
	live PID gains as currently set by the sliders on this window."""
	error_text = f'{error:+.0f}px' if error is not None else '--'
	lines = [
		f'Speed: {speed:+.1f} / {MAX_SPEED:.0f}',
		f'Error: {error_text}',
		f'Status: {status_label}',
		f'KP={kp:.2f}  KI={ki:.2f}  KD={kd:.2f}',
	]
	for i, line in enumerate(lines):
		y = 25 + i * 24
		cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _draw_pid_graph(frame, p_history, i_history, d_history, origin):
	"""Scrolling line graph of the last GRAPH_HISTORY_FRAMES P/I/D
	contributions to the commanded speed (see PID.last_p/last_i/last_d),
	each scaled to +-MAX_SPEED, drawn into a fixed box at `origin`."""
	x0, y0 = origin
	w, h = GRAPH_SIZE

	cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), -1)
	cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (200, 200, 200), 1)

	legend_x = x0 + 4
	for label, color in GRAPH_COLORS.items():
		cv2.putText(frame, label, (legend_x, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
		legend_x += 18

	mid_y = y0 + h // 2
	cv2.line(frame, (x0, mid_y), (x0 + w, mid_y), (100, 100, 100), 1)

	def _polyline(history):
		if len(history) < 2:
			return None
		points = []
		for i, s in enumerate(history):
			px = x0 + int(i / (GRAPH_HISTORY_FRAMES - 1) * w)
			norm = max(-1.0, min(1.0, s / MAX_SPEED)) if MAX_SPEED else 0.0
			py = mid_y - int(norm * (h / 2 - 4))
			points.append((px, py))
		return np.array(points, dtype=np.int32)

	for label, history in (('P', p_history), ('I', i_history), ('D', d_history)):
		points = _polyline(history)
		if points is not None:
			cv2.polylines(frame, [points], False, GRAPH_COLORS[label], 2)


def main():
	doublemotor = le.DoubleMotor()
	doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)

	if not doublemotor.connected:
		print('Error connecting to Double Motor.')
		exit(1)

	detector = AprilTagDetector()
	pid = PID(KP, KI, KD, output_limits=(-MAX_SPEED, MAX_SPEED))

	cv2.namedWindow(WINDOW_NAME)
	cv2.createTrackbar('KP x100', WINDOW_NAME, int(KP * SLIDER_SCALE), KP_SLIDER_MAX, lambda _: None)
	cv2.createTrackbar('KI x100', WINDOW_NAME, int(KI * SLIDER_SCALE), KI_SLIDER_MAX, lambda _: None)
	cv2.createTrackbar('KD x100', WINDOW_NAME, int(KD * SLIDER_SCALE), KD_SLIDER_MAX, lambda _: None)

	cap = cv2.VideoCapture(CAMERA_INDEX)
	last_time = time.time()
	missed_frames = 0
	smoothed_cx = None
	current_speed = 0.0
	target_speed = 0.0
	p_term = 0.0
	i_term = 0.0
	d_term = 0.0
	consecutive_read_failures = 0
	centered_frames = 0
	p_history = deque(maxlen=GRAPH_HISTORY_FRAMES)
	i_history = deque(maxlen=GRAPH_HISTORY_FRAMES)
	d_history = deque(maxlen=GRAPH_HISTORY_FRAMES)

	try:
		print("Centering AprilTag horizontally in frame. Press 'q' to quit.")
		while cap.isOpened():
			ok, frame = cap.read()

			if not ok:
				consecutive_read_failures += 1
				print(f'Camera read failed ({consecutive_read_failures}/{MAX_READ_FAILURES}) - stopping motors.')
				doublemotor.movement_stop()
				current_speed = 0.0
				if consecutive_read_failures >= MAX_READ_FAILURES:
					print('Too many consecutive camera read failures - exiting.')
					break
				continue

			consecutive_read_failures = 0

			pid.kp = cv2.getTrackbarPos('KP x100', WINDOW_NAME) / SLIDER_SCALE
			pid.ki = cv2.getTrackbarPos('KI x100', WINDOW_NAME) / SLIDER_SCALE
			pid.kd = cv2.getTrackbarPos('KD x100', WINDOW_NAME) / SLIDER_SCALE

			h, w = frame.shape[:2]
			target_x = w / 2.0

			now = time.time()
			dt = now - last_time
			last_time = now

			detection = detector.detect(frame)

			if detection is None:
				missed_frames += 1
				error = None
				centered_frames = 0
				if missed_frames >= MAX_MISSED_FRAMES:
					target_speed = 0.0
					p_term = i_term = d_term = 0.0
					pid.reset()
					smoothed_cx = None
				# else: keep coasting at the last commanded target_speed (and
				# last P/I/D contributions) - a brief dropout (e.g. motion
				# blur or a Wi-Fi hiccup) shouldn't yank the car to a stop;
				# only a sustained loss should.
			else:
				missed_frames = 0
				smoothed_cx = detection.cx if smoothed_cx is None else (
					SMOOTHING_ALPHA * detection.cx + (1 - SMOOTHING_ALPHA) * smoothed_cx
				)
				error = smoothed_cx - target_x

				if abs(error) < DEADBAND_PX:
					target_speed = 0.0
					p_term = i_term = d_term = 0.0
					pid.reset()
					centered_frames += 1
				else:
					output = pid.update(error, dt)
					target_speed = -output if INVERT_DIRECTION else output
					sign = -1.0 if INVERT_DIRECTION else 1.0
					p_term, i_term, d_term = sign * pid.last_p, sign * pid.last_i, sign * pid.last_d
					centered_frames = 0

				cv2.polylines(frame, [detection.corners.astype(int)], True, (0, 0, 255), 5)
				cv2.circle(frame, (int(detection.cx), int(detection.cy)), 5, (0, 0, 255), -1)

			if ENABLE_COMPLETION_STOP and centered_frames >= COMPLETE_HOLD_FRAMES:
				print('Objective complete! Tag centered.')
				doublemotor.movement_stop()
				if ENABLE_CELEBRATION:
					cv2.destroyWindow(WINDOW_NAME)
					celebration.run_celebration(CELEBRATION_PHOTO)
				break

			# Slew-rate limit: move current_speed toward target_speed by at
			# most MAX_ACCEL * dt this frame, so speed always ramps smoothly
			# (including down to a stop) instead of snapping between values.
			max_step = MAX_ACCEL * dt
			speed_diff = max(-max_step, min(max_step, target_speed - current_speed))
			current_speed += speed_diff
			speed = current_speed

			doublemotor.movement_move_tank(speed, speed, blocking=False)

			p_history.append(p_term)
			i_history.append(i_term)
			d_history.append(d_term)

			# Centered zone: a vertical band spanning the full frame height -
			# any y position within it counts as "centered" since only
			# horizontal position is controlled.
			band_x1 = int(target_x - DEADBAND_PX)
			band_x2 = int(target_x + DEADBAND_PX)
			cv2.rectangle(frame, (band_x1, 0), (band_x2, h), (255, 0, 0), 2)

			status = (
				f"error={error:.0f}px  speed={speed:.0f}" if error is not None
				else f"no tag ({missed_frames} frames)"
			)
			cv2.putText(frame, status, (10, h - 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

			_draw_hud(frame, speed, error, _status_label(detection, missed_frames), pid.kp, pid.ki, pid.kd)
			_draw_pid_graph(frame, p_history, i_history, d_history, origin=(w - GRAPH_SIZE[0] - 10, 10))
			cv2.imshow(WINDOW_NAME, frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		doublemotor.movement_stop()
		cap.release()
		cv2.destroyAllWindows()
		doublemotor.disconnect()


if __name__ == '__main__':
	main()
