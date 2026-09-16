"""Enumerate camera devices so you can find which index is the iPhone
(Continuity Camera) vs. the Mac's built-in camera.

Run this, note which preview window shows the iPhone's feed, then set
CAMERA_INDEX in main_iphone.py to that number - indices aren't guaranteed
stable across machines/reboots, so this is a one-time (or per-session)
lookup, not something to hardcode confidently without checking.

Usage: python3 list_cameras.py
Press any key to advance to the next camera, or 'q' to stop early.
"""

import cv2

MAX_INDEX_TO_TRY = 5
WARMUP_FRAMES = 5  # first frame(s) from a camera can be blank while it warms up


def main():
	for index in range(MAX_INDEX_TO_TRY):
		cap = cv2.VideoCapture(index)
		if not cap.isOpened():
			cap.release()
			continue

		frame = None
		for _ in range(WARMUP_FRAMES):
			ok, frame = cap.read()
			if not ok:
				frame = None
				break

		cap.release()
		if frame is None:
			continue

		print(f"Index {index}: opened - press any key for the next camera, 'q' to quit.")
		cv2.imshow(f'Camera index {index}', frame)
		key = cv2.waitKey(0) & 0xFF
		cv2.destroyAllWindows()
		if key == ord('q'):
			break


if __name__ == '__main__':
	main()
