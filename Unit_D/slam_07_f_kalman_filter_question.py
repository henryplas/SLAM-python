# The full Kalman filter, consisting of prediction and correction step.
#
# slam_07_f_kalman_filter
# Claus Brenner, 12.12.2012
from lego_robot import *
from math import sin, cos, pi, atan2, sqrt
from numpy import *
from slam_d_library import get_observations, write_cylinders


class ExtendedKalmanFilter:
    def __init__(self, state, covariance,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev):
        # The state. This is the core data of the Kalman filter.
        self.state = state
        self.covariance = covariance

        # Some constants.
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev

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

        return array([g1, g2, g3])

    @staticmethod
    def dg_dstate(state, control, w):
        x, y, theta = state
        l, r = control
        m = array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])  

        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            m[0][2] = (rad + (w / 2)) * (cos(theta + alpha) - cos(theta))
            m[1][2] = (rad + (w / 2)) * (sin(theta + alpha) - sin(theta))

        else:
            m[0][2] = -l * sin(theta)
            m[1][2] = l * cos(theta)

        return m


    @staticmethod
    def dg_dcontrol(state, control, w):
        x, y, theta = state
        l, r = tuple(control)
        
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            theta_prime = theta + alpha

            A = ((w * r) / (r - l)**2)
            B = ((r + l) / (2 * (r - l)))
            C = -((w * l) / (r - l)**2)

            dg1_dl = A * (sin(theta_prime) - sin(theta)) - (B * cos(theta_prime))
            dg2_dl = A * (-cos(theta_prime) + cos(theta)) - (B * sin(theta_prime))
            dg3_dl = -(1 / w)
            dg1_dr = C * (sin(theta_prime) - sin(theta)) + B * cos(theta_prime)
            dg2_dr = C * (-cos(theta_prime) + cos(theta)) + B * sin(theta_prime)
            dg3_dr = 1 / w

            m = array([[dg1_dl, dg1_dr], 
                        [dg2_dl, dg2_dr], 
                        [dg3_dl, dg3_dr]])

        else:   
            dg1_dl = 0.5 * (cos(theta) + (l/w) * sin(theta))
            dg2_dl = 0.5 * (sin(theta) - (l/w) * cos(theta))
            dg3_dl = -(1 / w)
            dg1_dr = 0.5 * (-(l/w) * sin(theta) + cos(theta))
            dg2_dr = 0.5 * ((l/w) * cos(theta) + sin(theta))
            dg3_dr = 1 / w

            m = array([[dg1_dl, dg1_dr], 
                        [dg2_dl, dg2_dr], 
                        [dg3_dl, dg3_dr]])
            
        return m

    @staticmethod
    def get_error_ellipse(covariance):
        """Return the position covariance (which is the upper 2x2 submatrix)
           as a triple: (main_axis_angle, stddev_1, stddev_2), where
           main_axis_angle is the angle (pointing direction) of the main axis,
           along which the standard deviation is stddev_1, and stddev_2 is the
           standard deviation along the other (orthogonal) axis."""
        eigenvals, eigenvects = linalg.eig(covariance[0:2,0:2])
        angle = atan2(eigenvects[1,0], eigenvects[0,0])
        return (angle, sqrt(eigenvals[0]), sqrt(eigenvals[1]))        

    def predict(self, control):
        """The prediction step of the Kalman filter."""
        # covariance' = G * covariance * GT + R
        # where R = V * (covariance in control space) * VT.
        # Covariance in control space depends on move distance.
        l, r = control

        sigL2 = (self.control_motion_factor*l)**2 \
                + (self.control_turn_factor*(r-l))**2
        sigR2 = (self.control_motion_factor*r)**2 \
                + (self.control_turn_factor*(r-l))**2

        control_cov = diag([sigL2, sigR2])
        V = self.dg_dcontrol(self.state, control, self.robot_width)
        G = self.dg_dstate(self.state, control, self.robot_width)
        self.covariance = G @ self.covariance @ G.T + V @ control_cov @ V.T

        # state' = g(state, control)
        self.state = self.g(self.state, control, self.robot_width)


    @staticmethod
    def h(state, landmark, scanner_displacement):
        """Takes a (x, y, theta) state and a (x, y) landmark, and returns the
           measurement (range, bearing)."""
        dx = landmark[0] - (state[0] + scanner_displacement * cos(state[2]))
        dy = landmark[1] - (state[1] + scanner_displacement * sin(state[2]))
        r = sqrt(dx * dx + dy * dy)
        alpha = (atan2(dy, dx) - state[2] + pi) % (2*pi) - pi

        return array([r, alpha])

    @staticmethod
    def dh_dstate(state, landmark, scanner_displacement):

        # --->>> Insert your code here.
        # Note that:
        x, y, theta = state
        x_m, y_m = landmark
        d = scanner_displacement
        x_l = x + d * cos(theta)
        y_l = y + d * sin(theta)

        delta_x = x_m - x_l
        delta_y = y_m - y_l

        q = delta_x**2 + delta_y**2

        dr_dx = -delta_x / sqrt(q)
        dr_dy = -delta_y / sqrt(q)
        dr_dtheta = (d /sqrt(q)) * (delta_x * sin(theta) - delta_y * cos(theta))
        da_dx = delta_y / q
        da_dy = -delta_x / q
        da_dtheta = (-d / q) * (delta_x * cos(theta) + delta_y * sin(theta)) - 1


        return array([[dr_dx, dr_dy, dr_dtheta], [da_dx, da_dy, da_dtheta]]) # Replace this.


    def correct(self, measurement, landmark):

        Q = diag([self.measurement_distance_stddev**2, self.measurement_angle_stddev**2])
        Z = array(measurement)
        Z[1] = atan2(sin(Z[1]), cos(Z[1]))

        H = self.dh_dstate(self.state, landmark, self.scanner_displacement)
        P = self.covariance

        K = P @ H.T @ linalg.inv(H @ P @ H.T + Q)
        pred_z = self.h(self.state, landmark, self.scanner_displacement)
        innov = Z - pred_z
        innov[1] = atan2(sin(innov[1]), cos(innov[1]))

        self.state = self.state + K @ innov
        self.covariance = (eye(3) - K @ H) @ P

if __name__ == '__main__':
    # Robot constants.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 155.0

    # Cylinder extraction and matching constants.
    minimum_valid_distance = 20.0
    depth_jump = 100.0
    cylinder_offset = 90.0
    max_cylinder_distance = 300.0

    # Filter constants.
    control_motion_factor = 0.35  # Error in motor control.
    control_turn_factor = 0.6  # Additional error due to slip when turning.
    measurement_distance_stddev = 200.0  # Distance measurement error of cylinders.
    measurement_angle_stddev = 15.0 / 180.0 * pi  # Angle measurement error.

    # Measured start position.
    initial_state = array([1850.0, 1897.0, 213.0 / 180.0 * pi])
    # Covariance at start position.
    initial_covariance = diag([100.0**2, 100.0**2, (10.0 / 180.0 * pi) ** 2])
    # Setup filter.
    kf = ExtendedKalmanFilter(initial_state, initial_covariance,
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

    # Loop over all motor tick records and all measurements and generate
    # filtered positions and covariances.
    # This is the Kalman filter loop, with prediction and correction.
    states = []
    covariances = []
    matched_ref_cylinders = []
    for i in range(len(logfile.motor_ticks)):
        # Prediction.
        control = array(logfile.motor_ticks[i]) * ticks_to_mm
        kf.predict(control)

        # Correction.
        observations = get_observations(
            logfile.scan_data[i],
            depth_jump, minimum_valid_distance, cylinder_offset,
            kf.state, scanner_displacement,
            reference_cylinders, max_cylinder_distance)
        for j in range(len(observations)):
            kf.correct(*observations[j])

        # Log state, covariance, and matched cylinders for later output.
        states.append(kf.state)
        covariances.append(kf.covariance)
        matched_ref_cylinders.append([m[1] for m in observations])

    # Write all states, all state covariances, and matched cylinders to file.
    f = open("kalman_prediction_and_correction.txt", "w")
    for i in range(len(states)):
        # Output the center of the scanner, not the center of the robot.
        print("F %f %f %f" % \
            tuple(states[i] + [scanner_displacement * cos(states[i][2]),
                               scanner_displacement * sin(states[i][2]),
                               0.0]), file=f)
        # Convert covariance matrix to angle stddev1 stddev2 stddev-heading form
        e = ExtendedKalmanFilter.get_error_ellipse(covariances[i])
        print("E %f %f %f %f" % (e + (sqrt(covariances[i][2,2]),)), file=f)
        # Also, write matched cylinders.
        write_cylinders(f, "W C", matched_ref_cylinders[i])

    f.close()