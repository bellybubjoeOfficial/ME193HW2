"""Objective-complete celebration: party hat + photo-shatter/confetti fall.

No generative image model involved - the hat is drawn as vector shapes
positioned via a simple background-threshold heuristic (this photo has a
plain light background, so separating subject from background is
reliable), and "confetti" is the real photo sliced into tiles that fall
with simple physics, mixed with small drawn confetti pieces that tumble.
"""

import random

import cv2
import numpy as np

HAT_COLOR_A = (60, 40, 220)     # BGR - red-ish
HAT_COLOR_B = (200, 130, 20)    # BGR - blue-ish
POMPOM_COLOR = (60, 220, 240)   # BGR - yellow-ish

CONFETTI_COLORS = [
	(60, 40, 220), (200, 130, 20), (60, 220, 240),
	(80, 200, 80), (180, 80, 200), (240, 240, 240),
]

GRID_ROWS = 10
GRID_COLS = 8
GRAVITY = 0.6
NUM_CONFETTI = 60


def _find_head_bbox(gray):
	"""Locate the subject's head via background thresholding.

	Assumes a plain, light, roughly-uniform background (true for a typical
	studio headshot) - anything darker than the threshold is "subject."
	Only looks at the top band of the image to avoid shoulders/shadow
	skewing the horizontal center.
	"""
	h, w = gray.shape
	_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

	band = mask[: int(h * 0.3), :]
	ys, xs = np.where(band > 0)
	if len(xs) == 0:
		return w // 2, 0, w // 3  # fallback: centered, top of image

	center_x = int((xs.min() + xs.max()) / 2)
	top_y = int(ys.min())
	head_width = int(xs.max() - xs.min())
	return center_x, top_y, head_width


def draw_party_hat(image_bgr):
	"""Returns image_bgr (padded with extra background-colored headroom, if
	needed) with a drawn birthday hat on top of the head.

	A tightly-cropped headshot may not have enough background above the
	hair to fit a proportionally-sized hat, so this pads the canvas upward
	with the photo's own background color rather than shrinking the hat to
	an unrealistically small size.
	"""
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	center_x, top_y, head_width = _find_head_bbox(gray)

	hat_width = max(int(head_width * 0.9), 40)
	hat_height = int(hat_width * 0.55)
	overlap = int(hat_height * 0.12)  # sink the hat base slightly into the hairline

	base_y = top_y + overlap
	tip_y = base_y - hat_height

	if tip_y < 0:
		pad_top = -tip_y + 10
		bg_color = [int(c) for c in image_bgr[0, 0]]
		h, w = image_bgr.shape[:2]
		padded = np.full((h + pad_top, w, 3), bg_color, dtype=np.uint8)
		padded[pad_top:, :] = image_bgr
		out = padded
		base_y += pad_top
		tip_y += pad_top
	else:
		out = image_bgr.copy()

	half_w = hat_width // 2

	bands = 5
	for i in range(bands):
		t0 = i / bands
		t1 = (i + 1) / bands
		# Linear taper from the tip (width 0) down to the base (full width).
		y0 = int(tip_y + (base_y - tip_y) * t0)
		y1 = int(tip_y + (base_y - tip_y) * t1)
		w0 = int(half_w * t0)
		w1 = int(half_w * t1)
		pts = np.array([
			[center_x - w0, y0], [center_x + w0, y0],
			[center_x + w1, y1], [center_x - w1, y1],
		])
		color = HAT_COLOR_A if i % 2 == 0 else HAT_COLOR_B
		cv2.fillPoly(out, [pts], color)

	cv2.circle(out, (center_x, tip_y), max(hat_width // 10, 6), POMPOM_COLOR, -1)
	return out


class _FallingTile:
	def __init__(self, patch, x, y):
		self.patch = patch
		self.x = float(x)
		self.y = float(y)
		self.vx = random.uniform(-2.0, 2.0)
		self.vy = random.uniform(-4.0, -1.0)  # small initial "pop" upward

	def step(self):
		self.vy += GRAVITY
		self.x += self.vx
		self.y += self.vy

	def offscreen(self, canvas_h):
		return self.y > canvas_h


class _Confetti:
	def __init__(self, x, canvas_h):
		self.x = float(x)
		self.y = float(random.uniform(-canvas_h * 0.3, 0))
		self.vx = random.uniform(-1.5, 1.5)
		self.vy = random.uniform(1.0, 3.0)
		self.angle = random.uniform(0, 360)
		self.angular_vel = random.uniform(-12, 12)
		self.size = random.randint(4, 9)
		self.color = random.choice(CONFETTI_COLORS)

	def step(self):
		self.vy += GRAVITY * 0.3
		self.x += self.vx
		self.y += self.vy
		self.angle += self.angular_vel

	def draw(self, canvas):
		rad = np.radians(self.angle)
		cos_a, sin_a = np.cos(rad), np.sin(rad)
		hw, hh = self.size, self.size * 0.5
		corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
		rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
		pts = (corners @ rot.T) + [self.x, self.y]
		cv2.fillPoly(canvas, [pts.astype(int)], self.color)


def _paste_patch(canvas, patch, x, y):
	"""Copies patch onto canvas at (x, y), clipped to canvas bounds."""
	ch, cw = canvas.shape[:2]
	ph, pw = patch.shape[:2]
	x0, y0 = int(x), int(y)
	x1, y1 = x0 + pw, y0 + ph

	cx0, cy0 = max(x0, 0), max(y0, 0)
	cx1, cy1 = min(x1, cw), min(y1, ch)
	if cx0 >= cx1 or cy0 >= cy1:
		return

	px0, py0 = cx0 - x0, cy0 - y0
	px1, py1 = px0 + (cx1 - cx0), py0 + (cy1 - cy0)
	canvas[cy0:cy1, cx0:cx1] = patch[py0:py1, px0:px1]


def _make_tiles(photo, offset_x=0, offset_y=0):
	h, w = photo.shape[:2]
	tile_h, tile_w = h // GRID_ROWS, w // GRID_COLS
	tiles = []
	for row in range(GRID_ROWS):
		for col in range(GRID_COLS):
			y, x = row * tile_h, col * tile_w
			patch = photo[y:y + tile_h, x:x + tile_w]
			tiles.append(_FallingTile(patch, offset_x + x, offset_y + y))
	return tiles


# Fullscreen mode stretches whatever's shown to fill the real display, so the
# canvas is built at a normal screen-like (16:9) aspect ratio rather than a
# tall narrow strip - otherwise it'd look badly distorted stretched wide.
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

# The photo (with hat) is scaled to at most this fraction of the canvas
# height, leaving room above/below for the hat and the fall.
PHOTO_HEIGHT_FRACTION = 0.55


def run_celebration(photo_path, window_name='Objective Complete!', duration_frames=150):
	"""Runs the shatter-into-confetti animation fullscreen.

	Blocks until the animation finishes (all tiles off-screen or
	duration_frames elapses) or the user presses 'q'.
	"""
	photo = cv2.imread(photo_path)
	if photo is None:
		print(f'Could not load {photo_path} for celebration animation.')
		return

	photo = draw_party_hat(photo)
	ph, pw = photo.shape[:2]

	max_h = int(CANVAS_HEIGHT * PHOTO_HEIGHT_FRACTION)
	if ph > max_h:
		scale = max_h / ph
		photo = cv2.resize(photo, (int(pw * scale), max_h))
		ph, pw = photo.shape[:2]

	canvas_w, canvas_h = CANVAS_WIDTH, CANVAS_HEIGHT
	offset_x = (canvas_w - pw) // 2
	offset_y = int(canvas_h * 0.08)

	tiles = _make_tiles(photo, offset_x, offset_y)
	confetti = [_Confetti(random.uniform(0, canvas_w), canvas_h) for _ in range(NUM_CONFETTI)]

	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	for _ in range(duration_frames):
		canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

		for tile in tiles:
			tile.step()
			if not tile.offscreen(canvas_h):
				_paste_patch(canvas, tile.patch, tile.x, tile.y)

		for piece in confetti:
			piece.step()
			if piece.y > canvas_h:
				piece.y = random.uniform(-30, 0)
				piece.x = random.uniform(0, canvas_w)
			piece.draw(canvas)

		cv2.putText(canvas, 'Objective Complete!', (20, 40),
					cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
		cv2.imshow(window_name, canvas)

		if cv2.waitKey(30) & 0xFF == ord('q'):
			break

		if all(tile.offscreen(canvas_h) for tile in tiles):
			break

	cv2.waitKey(1500)
	cv2.destroyWindow(window_name)
