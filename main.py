# Entry point: webcam -> AprilTag detection -> PID -> Double Motor.
#
# An AprilTag sits on top of the car. The car only drives forward/backward
# (both wheels at the same speed - no steering), and a PID loop drives it
# until the tag's horizontal (x) pixel position is centered in the webcam
# frame - vertical (y) position doesn't matter, so the "centered" zone is
# drawn as a vertical band spanning the full frame height.

import time

import cv2
import legoeducation as le

from apriltag_detector import AprilTagDetector
from pid import PID

# --- LEGO connection card - update to match your Double Motor's card ---
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = '5164'

# --- PID gains - start conservative and tune from here ---
KP = 0.15
KI = 0.0

# Conservative output cap while tuning, independent of KP - this is the
# actual ceiling on how fast the car can ever go, regardless of how far off
# center the tag is.
MAX_SPEED = 25.0

# Pixel error inside this band counts as "centered" - stops the motors
# instead of jittering around the setpoint.
DEADBAND_PX = 15.0

# EMA smoothing on the tag's detected horizontal position (0-1: higher =
# less smoothing, more responsive; lower = smoother, laggier). Removes the
# frame-to-frame pixel jitter that would otherwise show up directly as
# jerky speed changes.
SMOOTHING_ALPHA = 0.3

# Max speed change allowed per second - ramps the commanded speed smoothly
# instead of snapping to a new value every frame. This also smooths the
# transition into/out of the deadband stop.
MAX_ACCEL = 80.0

# Flip to True if the car drives away from center instead of toward it -
# means the physical mounting has the tag/camera oriented opposite to what
# this script assumes.
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

	cap = cv2.VideoCapture(0)
	last_time = time.time()
	missed_frames = 0
	smoothed_cx = None
	current_speed = 0.0

	try:
		print("Centering AprilTag horizontally in frame. Press 'q' to quit.")
		while cap.isOpened():
			ok, frame = cap.read()
			if not ok:
				break

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
			cv2.imshow('AprilTag Centering', frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	finally:
		doublemotor.movement_stop()
		cap.release()
		cv2.destroyAllWindows()
		doublemotor.disconnect()


if __name__ == '__main__':
	main()
