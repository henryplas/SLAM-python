# The particle filter, prediciton and correction.
# In addition to the filtered particles, the density is estimated. In this
# simple case, the mean position and heading is computed from all particles.
#
# slam_08_c_density_estimation.
# Claus Brenner, 04.01.2013
from lego_robot import *
from slam_e_library import get_cylinders_from_scan, assign_cylinders
from math import sin, cos, pi, atan2, sqrt
import random
from scipy.stats import norm as normal_dist
import numpy as np


class ParticleFilter:
	def __init__(self, initial_particles,
				 robot_width, scanner_displacement,
				 control_motion_factor, control_turn_factor,
				 measurement_distance_stddev, measurement_angle_stddev):
		# The particles.
		self.particles = initial_particles

		# Some constants.
		self.robot_width = robot_width
		self.scanner_displacement = scanner_displacement
		self.control_motion_factor = control_motion_factor
		self.control_turn_factor = control_turn_factor
		self.measurement_distance_stddev = measurement_distance_stddev
		self.measurement_angle_stddev = measurement_angle_stddev

	# State transition. This is exactly the same method as in the Kalman filter.
	@staticmethod
	def g(state, control, w):
		x, y, theta = state
		l, r = control
		if r != l:
			alpha = (r - l) / w
			rad = l/alpha
			g1 = x + (rad + w/2.)*(sin(theta+alpha) - sin(theta))
			g2 = y + (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
			g3 = (theta + alpha + pi) % (2*pi) - pi
		else:
			g1 = x + l * cos(theta)
			g2 = y + l * sin(theta)
			g3 = theta

		return (g1, g2, g3)

	def predict(self, control):
		"""The prediction step of the particle filter."""
		l, r = control
		alpha_1 = self.control_motion_factor
		alpha_2 = self.control_turn_factor

		left_var = (alpha_1 * l)**2 + (alpha_2 * (l-r))**2
		right_var = (alpha_1 * r)**2 + (alpha_2 * (l-r))**2
		left_std = sqrt(left_var)
		right_std = sqrt(right_var)

		new_part = []
		for particle in self.particles:
			l_prime = random.gauss(l, left_std)
			r_prime = random.gauss(r, right_std)
			p_u = [l_prime,r_prime]
			new_part.append(self.g(particle, p_u, self.robot_width))

		self.particles = new_part


	# Measurement. This is exactly the same method as in the Kalman filter.
	@staticmethod
	def h(state, landmark, scanner_displacement):
		"""Takes a (x, y, theta) state and a (x, y) landmark, and returns the
		   corresponding (range, bearing)."""

		dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
		dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
		r = sqrt(dx * dx + dy * dy)
		alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi
		return (r, alpha)

	def probability_of_measurement(self, measurement, predicted_measurement):
		"""Given a measurement and a predicted measurement, computes
		   probability."""

		d, alpha = measurement
		d_prime, alpha_prime = predicted_measurement

		alpha = atan2(sin(alpha), cos(alpha))
		alpha_prime = atan2(sin(alpha_prime), cos(alpha_prime))

		delta_alpha = alpha - alpha_prime
		delta_alpha = atan2(sin(delta_alpha), cos(delta_alpha))

		ret = normal_dist.pdf(d - d_prime, 0 , self.measurement_distance_stddev) * \
				normal_dist.pdf(delta_alpha, 0, self.measurement_angle_stddev)

		return ret

	def compute_weights(self, cylinders, landmarks):
		"""Computes one weight for each particle, returns list of weights."""
		weights = []
		for p in self.particles:
			# Get list of tuples:
			# [ ((range_0, bearing_0), (landmark_x, landmark_y)), ... ]
			assignment = assign_cylinders(cylinders, p,
				self.scanner_displacement, landmarks)

			prob = 1.
			for tupl in assignment:
				meas_prime = self.h(p, tupl[1], self.scanner_displacement)
				prob *= self.probability_of_measurement(tupl[0], meas_prime)

			weights.append(prob)
		weights = np.array(weights)
		total = sum(weights)
		return weights/total

	def resample(self, weights):
		"""Return a list of particles which have been resampled, proportional
		   to the given weights."""
		num_particles = len(self.particles)

		return [self.particles[np.random.choice(num_particles, p=weights)]
				for i in range(num_particles)]

	def correct(self, cylinders, landmarks):
		"""The correction step of the particle filter."""
		# First compute all weights.
		weights = self.compute_weights(cylinders, landmarks)
		# Then resample, based on the weight array.
		self.particles = self.resample(weights)

	def print_particles(self, file_desc):
		"""Prints particles to given file_desc output."""
		if not self.particles:
			return
		print("PA", file=file_desc, end=' ')
		for p in self.particles:
			print("%.0f %.0f %.3f" % p, file=file_desc, end=' ')
		print(file=file_desc)

	def get_mean(self):
		"""Compute mean position and heading from all particles."""

		# --->>> This is the new code you'll have to implement.
		# Return a tuple: (mean_x, mean_y, mean_heading).
		parts = np.array(self.particles)
		x_hat = sum(parts[:,0]) / len(parts)
		y_hat = sum(parts[:,1]) / len(parts)
		
		cos_list = np.cos(parts[:,2])
		sin_list = np.sin(parts[:,2])
		v_x = sum(cos_list)
		v_y = sum(sin_list)
		theta_hat = atan2(v_y, v_x)
		return (x_hat, y_hat, theta_hat)


if __name__ == '__main__':
	# Robot constants.
	scanner_displacement = 30.0
	ticks_to_mm = 0.349
	robot_width = 155.0

	# Cylinder extraction and matching constants.
	minimum_valid_distance = 20.0
	depth_jump = 100.0
	cylinder_offset = 90.0

	# Filter constants.
	control_motion_factor = 0.35  # Error in motor control.
	control_turn_factor = 0.6  # Additional error due to slip when turning.
	measurement_distance_stddev = 200.0  # Distance measurement error of cylinders.
	measurement_angle_stddev = 15.0 / 180.0 * pi  # Angle measurement error.

	# Generate initial particles. Each particle is (x, y, theta).
	number_of_particles = 200
	measured_state = (1850.0, 1897.0, 213.0 / 180.0 * pi)
	standard_deviations = (100.0, 100.0, 10.0 / 180.0 * pi)
	initial_particles = []
	for i in range(number_of_particles):
		initial_particles.append(tuple([
			random.gauss(measured_state[j], standard_deviations[j])
			for j in range(3)]))

	# Setup filter.
	pf = ParticleFilter(initial_particles,
						robot_width, scanner_displacement,
						control_motion_factor, control_turn_factor,
						measurement_distance_stddev,
						measurement_angle_stddev)

	# Read data.
	logfile = LegoLogfile()
	logfile.read("robot4_motors.txt")
	logfile.read("robot4_scan.txt")
	logfile.read("robot_arena_landmarks.txt")
	reference_cylinders = [l[1:3] for l in logfile.landmarks]

	# Loop over all motor tick records.
	# This is the particle filter loop, with prediction and correction.
	f = open("particle_filter_mean.txt", "w")
	for i in range(len(logfile.motor_ticks)):
		# Prediction.
		control = map(lambda x: x * ticks_to_mm, logfile.motor_ticks[i])
		pf.predict(control)

		# Correction.
		cylinders = get_cylinders_from_scan(logfile.scan_data[i], depth_jump,
			minimum_valid_distance, cylinder_offset)
		pf.correct(cylinders, reference_cylinders)

		# Output particles.
		pf.print_particles(f)
		
		# Output state estimated from all particles.
		mean = pf.get_mean()
		print("F %.0f %.0f %.3f" %\
			  (mean[0] + scanner_displacement * cos(mean[2]),
			   mean[1] + scanner_displacement * sin(mean[2]),
			   mean[2]), file=f)

	f.close()
