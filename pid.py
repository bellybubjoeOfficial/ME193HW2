"""Minimal PI controller. Pure logic, no camera/hardware deps.

No derivative term: with a noisy pixel-position signal, a D term mostly
amplifies frame-to-frame jitter into spurious speed spikes rather than
damping real oscillation. Input smoothing (done upstream, before error is
computed) handles noise instead.
"""


class PID:
	def __init__(self, kp, ki, output_limits=(-100.0, 100.0)):
		self.kp = kp
		self.ki = ki
		self.output_min, self.output_max = output_limits
		self._integral = 0.0

	def reset(self):
		self._integral = 0.0

	def update(self, error, dt):
		"""Advance the controller by one time step and return the clamped output."""
		if dt <= 0:
			return 0.0

		self._integral += error * dt
		output = self.kp * error + self.ki * self._integral
		clamped = max(self.output_min, min(self.output_max, output))

		# Anti-windup: if the output saturated, undo this step's integral
		# contribution so the integral term doesn't keep growing unbounded
		# while the actuator is already maxed out.
		if clamped != output:
			self._integral -= error * dt

		return clamped
