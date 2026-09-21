"""Minimal PID controller. Pure logic, no camera/hardware deps.

The D term is computed on the (upstream-smoothed) error, so it stays usable
without amplifying raw frame-to-frame pixel jitter into spurious speed
spikes - keep KD low if it starts jittering.
"""


class PID:
	def __init__(self, kp, ki, kd=0.0, output_limits=(-100.0, 100.0)):
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self.output_min, self.output_max = output_limits
		self._integral = 0.0
		self._prev_error = None

		# Per-term contributions from the most recent update(), pre-clamp -
		# exposed for display/graphing (e.g. tuning HUDs), not used
		# internally. They sum to the unclamped output, which may exceed
		# output_limits when the total output saturates.
		self.last_p = 0.0
		self.last_i = 0.0
		self.last_d = 0.0

	def reset(self):
		self._integral = 0.0
		self._prev_error = None
		self.last_p = 0.0
		self.last_i = 0.0
		self.last_d = 0.0

	def update(self, error, dt):
		"""Advance the controller by one time step and return the clamped output."""
		if dt <= 0:
			return 0.0

		self._integral += error * dt
		derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
		self._prev_error = error

		self.last_p = self.kp * error
		self.last_i = self.ki * self._integral
		self.last_d = self.kd * derivative

		output = self.last_p + self.last_i + self.last_d
		clamped = max(self.output_min, min(self.output_max, output))

		# Anti-windup: if the output saturated, undo this step's integral
		# contribution so the integral term doesn't keep growing unbounded
		# while the actuator is already maxed out.
		if clamped != output:
			self._integral -= error * dt

		return clamped
