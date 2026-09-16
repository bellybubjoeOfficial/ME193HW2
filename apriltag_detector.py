"""AprilTag detection via OpenCV's built-in aruco module (no extra deps).

Uses the AprilTag 36h11 dictionary by default - the family printed by most
standard AprilTag generators (e.g. the AprilTag repo's tag36h11 set). If your
tag was printed from a different family, change DEFAULT_DICTIONARY below.
"""

from dataclasses import dataclass

import cv2
import cv2.aruco as aruco

DEFAULT_DICTIONARY = aruco.DICT_APRILTAG_36h11


@dataclass
class TagDetection:
	tag_id: int
	cx: float
	cy: float
	corners: "cv2.typing.MatLike"  # shape (4, 2), pixel coordinates


class AprilTagDetector:
	def __init__(self, dictionary=DEFAULT_DICTIONARY):
		self._dictionary = aruco.getPredefinedDictionary(dictionary)
		self._params = aruco.DetectorParameters()
		self._detector = aruco.ArucoDetector(self._dictionary, self._params)

	def detect(self, frame_bgr):
		"""Returns the first detected tag as a TagDetection, or None if none found."""
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		corners, ids, _ = self._detector.detectMarkers(gray)

		if ids is None or len(ids) == 0:
			return None

		tag_corners = corners[0][0]  # (4, 2): the 4 tag corners for the first detection
		cx = float(tag_corners[:, 0].mean())
		cy = float(tag_corners[:, 1].mean())
		# ids is (N,) in OpenCV 5.x but (N, 1) in older 4.x releases - ravel
		# handles either shape.
		tag_id = int(ids.ravel()[0])
		return TagDetection(tag_id=tag_id, cx=cx, cy=cy, corners=tag_corners)
