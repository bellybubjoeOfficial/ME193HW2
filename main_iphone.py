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

import cv2
import legoeducation as le

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
KP = 0.35
KI = 0.0

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


def main():
	doublemotor = le.DoubleMotor()
	doublemotor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)

	if not doublemotor.connected:
		print('Error connecting to Double Motor.')
		exit(1)

	detector = AprilTagDetector()
	pid = PID(KP, KI, output_limits=(-MAX_SPEED, MAX_SPEED))

	cap = cv2.VideoCapture(CAMERA_INDEX)
	last_time = time.time()
	missed_frames = 0
	smoothed_cx = None
	current_speed = 0.0
	consecutive_read_failures = 0

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

			h, w = frame.shape[:2]
			target_x = w / 2.0

			now = time.time()
			dt = now - last_time
			last_time = now

			detection = detector.detect(frame)

			if detection is None:
				missed_frames += 1
				target_speed = 0.0
				error = None
				if missed_frames >= MAX_MISSED_FRAMES:
					pid.reset()
					smoothed_cx = None
			else:
				missed_frames = 0
				smoothed_cx = detection.cx if smoothed_cx is None else (
					SMOOTHING_ALPHA * detection.cx + (1 - SMOOTHING_ALPHA) * smoothed_cx
				)
				error = smoothed_cx - target_x

				if abs(error) < DEADBAND_PX:
					target_speed = 0.0
					pid.reset()
				else:
					output = pid.update(error, dt)
					target_speed = -output if INVERT_DIRECTION else output

				cv2.polylines(frame, [detection.corners.astype(int)], True, (0, 0, 255), 5)
				cv2.circle(frame, (int(detection.cx), int(detection.cy)), 5, (0, 0, 255), -1)

			# Slew-rate limit: move current_speed toward target_speed by at
			# most MAX_ACCEL * dt this frame, so speed always ramps smoothly
			# (including down to a stop) instead of snapping between values.
			max_step = MAX_ACCEL * dt
			speed_diff = max(-max_step, min(max_step, target_speed - current_speed))
			current_speed += speed_diff
			speed = current_speed

			doublemotor.movement_move_tank(speed, speed, blocking=False)

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
			cv2.imshow('AprilTag Centering (iPhone)', frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		doublemotor.movement_stop()
		cap.release()
		cv2.destroyAllWindows()
		doublemotor.disconnect()


if __name__ == '__main__':
	main()
