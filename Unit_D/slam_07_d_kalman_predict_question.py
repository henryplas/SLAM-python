# The complete Kalman prediction step (without the correction step).
#
# slam_07_d_kalman_predict_solution
# Claus Brenner, 12.12.2012
from lego_robot import *
from math import sin, cos, pi, atan2
from numpy import *


class ExtendedKalmanFilter:
    def __init__(self, state, covariance,
                 robot_width,
                 control_motion_factor, control_turn_factor):
        # The state. This is the core data of the Kalman filter.
        self.state = state
        self.covariance = covariance

        # Some constants.
        self.robot_width = robot_width
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor

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
        left, right = control
        sigma2l = (self.control_motion_factor * left)**2 + (self.control_turn_factor * (left - right))**2
        sigma2r = (self.control_motion_factor * right)**2 + (self.control_turn_factor * (left - right))**2
        sigma_control = array([[sigma2l, 0], [0, sigma2r]])

        G = ExtendedKalmanFilter.dg_dstate(self.state, control,self.robot_width)
        V = ExtendedKalmanFilter.dg_dcontrol(self.state, control,self.robot_width)

        self.covariance = (G @ self.covariance) @ G.T + (V @ sigma_control) @ V.T

        self.state = ExtendedKalmanFilter.g(self.state, control, self.robot_width)



if __name__ == '__main__':
    # Robot constants.
    scanner_displacement = 30.0
    ticks_to_mm = 0.349
    robot_width = 155.0

    # Filter constants.
    control_motion_factor = 0.35  # Error in motor control.
    control_turn_factor = 0.6  # Additional error due to slip when turning.

    # Measured start position.
    initial_state = array([1850.0, 1897.0, 213.0 / 180.0 * pi])
    # Covariance at start position.
    initial_covariance = diag([100.0**2, 100.0**2, (10.0 / 180.0 * pi) ** 2])
    # Setup filter.
    kf = ExtendedKalmanFilter(initial_state, initial_covariance,
                              robot_width,
                              control_motion_factor, control_turn_factor)

    # Read data.
    logfile = LegoLogfile()
    logfile.read("robot4_motors.txt")

    # Loop over all motor tick records and generate filtered positions and
    # covariances.
    # This is the Kalman filter loop, without the correction step.
    states = []
    covariances = []
    for m in logfile.motor_ticks:
        # Prediction.
        control = array(m) * ticks_to_mm
        kf.predict(control)

        # Log state and covariance.
        states.append(kf.state)
        covariances.append(kf.covariance)

    # Write all states, all state covariances, and matched cylinders to file.
    f = open("kalman_prediction.txt", "w")
    for i in range(len(states)):
        # Output the center of the scanner, not the center of the robot.
        print( "F %f %f %f" % \
            tuple(states[i] + [scanner_displacement * cos(states[i][2]),
                               scanner_displacement * sin(states[i][2]),
                               0.0]), file=f)
        # Convert covariance matrix to angle stddev1 stddev2 stddev-heading form
        e = ExtendedKalmanFilter.get_error_ellipse(covariances[i])
        print("E %f %f %f %f" % (e + (sqrt(covariances[i][2,2]),)), file=f)

    f.close()
