# For each cylinder in the scan, find its cartesian coordinates,
# in the world coordinate system.
# Find the closest pairs of cylinders from the scanner and cylinders
# from the reference, and the optimal transformation which aligns them.
# Then, use this transform to correct the pose.
# 04_d_apply_transform
# Claus Brenner, 14 NOV 2012
from lego_robot import *
from slam_b_library import filter_step
from slam_04_a_project_landmarks import\
	 compute_scanner_cylinders, write_cylinders
from math import sqrt, atan2
import numpy as np


# Given a list of cylinders (points) and reference_cylinders:
# For every cylinder, find the closest reference_cylinder and add
# the index pair (i, j), where i is the index of the cylinder, and
# j is the index of the reference_cylinder, to the result list.
# This is the function developed in slam_04_b_find_cylinder_pairs.
def find_cylinder_pairs(cylinders, reference_cylinders, max_radius):
	cylinder_pairs = []
	for i, cyl in enumerate(cylinders):
		distance = 0
		min_dist = 999999
		min_element = 0
		for j, ref_cyl in enumerate(reference_cylinders):
			distance = sqrt((cyl[0] - ref_cyl[0])**2 + (cyl[1] - ref_cyl[1])**2)
			if min_dist > distance:
				min_element = j
				min_dist = distance
		if min_dist < max_radius:
			cylinder_pairs.append((i, min_element))

	return cylinder_pairs

# Given a point list, return the center of mass.
def compute_center(point_list):
	# Safeguard against empty list.
	if not point_list:
		return (0.0, 0.0)
	# If not empty, sum up and divide.
	sx = sum([p[0] for p in point_list])
	sy = sum([p[1] for p in point_list])
	return (sx / len(point_list), sy / len(point_list))

# Given a left_list of points and a right_list of points, compute
# the parameters of a similarity transform: scale, rotation, translation.
# If fix_scale is True, use the fixed scale of 1.0.
# The returned value is a tuple of:
# (scale, cos(angle), sin(angle), x_translation, y_translation)
# i.e., the rotation angle is not given in radians, but rather in terms
# of the cosine and sine.
def estimate_transform(left_list, right_list, fix_scale = False):
	 # Compute left and right center.	
	lc = compute_center(left_list)
	rc = compute_center(right_list)
	lc = np.array(lc)
	rc = np.array(rc)
	left_list = np.array(left_list)
	right_list = np.array(right_list)

	# --->>> Insert here your code to compute lambda, c, s and tx, ty.
	if not left_list.any():
		return None
	l_prime = left_list - lc
	r_prime = right_list - rc

	cs, ss, rr, ll = 0, 0, 0, 0
	for i in range(len(left_list)):
		cs += r_prime[i][0] * l_prime[i][0] + r_prime[i][1] * l_prime[i][1]
		ss += -r_prime[i][0] * l_prime[i][1] + r_prime[i][1] * l_prime[i][0]
		rr += r_prime[i][0] * r_prime[i][0] + r_prime[i][1] * r_prime[i][1]
		ll += l_prime[i][0] * l_prime[i][0] + l_prime[i][1] * l_prime[i][1]

	if ll < 1e-4 and rr < 1e-4:
		return None

	if fix_scale:
		la = 1
	else:
		la = sqrt(rr/ll)

	c = cs / sqrt(cs**2 + ss**2)
	s = ss / sqrt(cs**2 + ss**2)

	R = np.array([[c, -s], [s, c]])

	t = rc - la * R @ lc
	tx = t[0]
	ty = t[1]
	return la, c, s, tx, ty

	# Given a similarity transformation:
	# trafo = (scale, cos(angle), sin(angle), x_translation, y_translation)
	# and a point p = (x, y), return the transformed point.
def apply_transform(trafo, p):
	la, c, s, tx, ty = trafo
	lac = la * c
	las = la * s
	x = lac * p[0] - las * p[1] + tx
	y = las * p[0] + lac * p[1] + ty
	return (x, y)

# Correct the pose = (x, y, heading) of the robot using the given
# similarity transform. Note this changes the position as well as
# the heading.
def correct_pose(pose, trafo):
	la, c, s, tx, ty = trafo
	corrected = apply_transform(trafo, pose)
	heading = atan2(s, c) + pose[2]
	return (corrected[0], corrected[1], heading)


if __name__ == '__main__':
	# The constants we used for the filter_step.
	scanner_displacement = 30.0
	ticks_to_mm = 0.349
	robot_width = 150.0

	# The constants we used for the cylinder detection in our scan.    
	minimum_valid_distance = 20.0
	depth_jump = 100.0
	cylinder_offset = 90.0

	# The maximum distance allowed for cylinder assignment.
	max_cylinder_distance = 400.0

	# The start pose we obtained miraculously.
	pose = (1850.0, 1897.0, 3.717551306747922)

	# Read the logfile which contains all scans.
	logfile = LegoLogfile()
	logfile.read("robot4_motors.txt")
	logfile.read("robot4_scan.txt")

	# Also read the reference cylinders (this is our map).
	logfile.read("robot_arena_landmarks.txt")
	reference_cylinders = [l[1:3] for l in logfile.landmarks]

	out_file = open("apply_transform.txt", "w")
	for i in range(len(logfile.scan_data)):
		# Compute the new pose.
		pose = filter_step(pose, logfile.motor_ticks[i],
						   ticks_to_mm, robot_width,
						   scanner_displacement)

		# Extract cylinders, also convert them to world coordinates.
		cartesian_cylinders = compute_scanner_cylinders(
			logfile.scan_data[i],
			depth_jump, minimum_valid_distance, cylinder_offset)
		world_cylinders = [LegoLogfile.scanner_to_world(pose, c)
						   for c in cartesian_cylinders]

		# For every cylinder, find the closest reference cylinder.
		cylinder_pairs = find_cylinder_pairs(
			world_cylinders, reference_cylinders, max_cylinder_distance)

		# Estimate a transformation using the cylinder pairs.
		trafo = estimate_transform(
			[world_cylinders[pair[0]] for pair in cylinder_pairs],
			[reference_cylinders[pair[1]] for pair in cylinder_pairs],
			fix_scale = True)

		# Transform the cylinders using the estimated transform.
		transformed_world_cylinders = []
		if trafo:
			transformed_world_cylinders =\
				[apply_transform(trafo, c) for c in
				 [world_cylinders[pair[0]] for pair in cylinder_pairs]]

		# Also apply the trafo to correct the position and heading.
		if trafo:
			pose = correct_pose(pose, trafo)

		# Write to file.
		# The pose.
		print("F %f %f %f" % pose, file=out_file)
		# The detected cylinders in the scanner's coordinate system.
		write_cylinders(out_file, "D C", cartesian_cylinders)
		# The detected cylinders, transformed using the estimated trafo.
		write_cylinders(out_file, "W C", transformed_world_cylinders)

	out_file.close()
