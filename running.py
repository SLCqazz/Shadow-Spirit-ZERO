import time
import math
import numpy as np
from pca9685 import PCA9685
from spot_micro_kinematics.spot_micro_stick_figure import SpotMicroStickFigure
from math import pi, sin, cos, sqrt, atan2, acos

d2r = pi/180
r2d = 180/pi

class BezierGaitController:
    def __init__(self, robot):
        self.robot = robot
        self.stance_height = -0.14
        self.step_length = 0.08
        self.step_height = 0.04
        self.phase = 0
        self.freq = 1.0
        self.v_x = 0
        self.v_y = 0
        self.omega = 0
        self.moving = False
        self.initial_angles = self.get_initial_angles()

    def get_initial_angles(self):
        return [
            [0, -30 * d2r, 60 * d2r],  # Right back
            [0, -30 * d2r, 60 * d2r],  # Right front
            [0, 30 * d2r, -60 * d2r],  # Left front
            [0, 30 * d2r, -60 * d2r]   # Left back
        ]

    def bezier_curve(self, t, P0, P1, P2, P3):
        return (1-t)**3 * P0 + 3*(1-t)**2 * t * P1 + 3*(1-t) * t**2 * P2 + t**3 * P3

    def leg_trajectory(self, phase, x_offset, y_offset):
        if not self.moving:
            return x_offset, y_offset, self.stance_height

        if phase < 0.5:  # Stance phase
            x = x_offset + self.step_length * (0.5 - phase)
            y = y_offset
            z = self.stance_height
        else:  # Swing phase
            t = (phase - 0.5) * 2
            P0 = np.array([x_offset - self.step_length/2, y_offset, self.stance_height])
            P1 = np.array([x_offset - self.step_length/4, y_offset, self.stance_height])
            P2 = np.array([x_offset + self.step_length/4, y_offset, self.stance_height + self.step_height])
            P3 = np.array([x_offset + self.step_length/2, y_offset, self.stance_height])
            point = self.bezier_curve(t, P0, P1, P2, P3)
            x, y, z = point

        # Ensure the point is within a reachable sphere
        max_reach = self.robot.hip_length + self.robot.upper_leg_length + self.robot.lower_leg_length
        current_reach = sqrt(x**2 + y**2 + z**2)
        if current_reach > max_reach * 0.9:  # Use 90% of max reach to be safe
            scale = (max_reach * 0.9) / current_reach
            x *= scale
            y *= scale
            z *= scale

        return x, y, z

    def inverse_kinematics(self, x, y, z, leg_index):
        # Leg dimensions
        l1 = self.robot.hip_length
        l2 = self.robot.upper_leg_length
        l3 = self.robot.lower_leg_length

        # Adjust for hip offset
        if leg_index in [0, 1]:  # Right legs
            y -= l1
        else:  # Left legs
            y += l1

        # Calculate hip angle
        hip_angle = atan2(y, abs(x))  # Use abs(x) to keep hip rotation minimal

        # Calculate distance from hip to foot in x-y plane
        d = sqrt(x**2 + y**2)

        # Calculate distance from hip to foot in 3D space
        r = sqrt(d**2 + z**2)

        # Check if the point is reachable
        if r > l2 + l3:
            print(f"Warning: Position out of reach for leg {leg_index}")
            return self.initial_angles[leg_index]

        # Calculate angles for knee and ankle
        cos_ankle = (l2**2 + l3**2 - r**2) / (2 * l2 * l3)
        cos_ankle = max(min(cos_ankle, 1), -1)  # Clamp to [-1, 1]
        ankle_angle = acos(cos_ankle)

        a1 = atan2(z, d)
        a2 = acos((l2**2 + r**2 - l3**2) / (2 * l2 * r))
        knee_angle = a1 + a2

        # Adjust angles based on leg side and orientation
        if leg_index in [0, 1]:  # Right legs
            hip_angle = -hip_angle  # Negate for right side
            knee_angle = pi/2 - knee_angle
            ankle_angle = ankle_angle - pi
        else:  # Left legs
            knee_angle = knee_angle - pi/2
            ankle_angle = pi - ankle_angle

        # Clamp angles to reasonable ranges
        hip_angle = max(min(hip_angle, pi/4), -pi/4)  # Limit hip angle to ±45 degrees
        knee_angle = max(min(knee_angle, pi/2), -pi/2)
        ankle_angle = max(min(ankle_angle, pi), -pi)

        return hip_angle, knee_angle, ankle_angle

    def update(self, dt):
        if self.moving:
            self.phase += self.freq * dt
            if self.phase >= 1:
                self.phase -= 1

            # Calculate foot positions
            rf = self.leg_trajectory(self.phase, self.robot.body_length/2, self.robot.body_width/2)
            lf = self.leg_trajectory((self.phase + 0.5) % 1, self.robot.body_length/2, -self.robot.body_width/2)
            rb = self.leg_trajectory((self.phase + 0.5) % 1, -self.robot.body_length/2, self.robot.body_width/2)
            lb = self.leg_trajectory(self.phase, -self.robot.body_length/2, -self.robot.body_width/2)

            # Adjust for forward/backward motion
            motion_adjust = self.v_x * dt
            rf = (rf[0] + motion_adjust, rf[1], rf[2])
            lf = (lf[0] + motion_adjust, lf[1], lf[2])
            rb = (rb[0] + motion_adjust, rb[1], rb[2])
            lb = (lb[0] + motion_adjust, lb[1], lb[2])

            # Calculate leg angles using inverse kinematics
            rb_angles = self.inverse_kinematics(*rb, 0)
            rf_angles = self.inverse_kinematics(*rf, 1)
            lf_angles = self.inverse_kinematics(*lf, 2)
            lb_angles = self.inverse_kinematics(*lb, 3)

            # Set leg angles
            self.robot.set_leg_angles([rb_angles, rf_angles, lf_angles, lb_angles])

            # Update robot position and orientation
            self.robot.x += self.v_x * dt
            self.robot.y += self.v_y * dt
            self.robot.psi += self.omega * dt
        else:
            self.robot.set_leg_angles(self.initial_angles)

class SpotMicroHardware:
    def __init__(self):
        self.pwm = PCA9685(address=0x40, debug=False)
        self.pwm.setPWMFreq(50)  # Set frequency to 50Hz

        # Servo channel mappings
        self.channels = {
            'LB': {'hip': 2, 'upper': 4, 'lower': 3},
            'RB': {'hip': 13, 'upper': 12, 'lower': 11},
            'LF': {'hip': 6, 'upper': 7, 'lower': 5},
            'RF': {'hip': 9, 'upper': 10, 'lower': 8}
        }

        # Servo calibration data
        self.calibration = {
            'LB': {'hip': (1000, -1), 'upper': (2200, 1), 'lower': (1000, -1)},
            'RB': {'hip': (1000, 1), 'upper': (1100, 1), 'lower': (2100, -1)},
            'LF': {'hip': (1950, 1), 'upper': (1500, 1), 'lower': (950, -1)},
            'RF': {'hip': (2100, -1), 'upper': (1150, 1), 'lower': (2000, -1)}
        }

    def angle_to_pwm(self, leg, joint, angle):
        zero_pwm, direction = self.calibration[leg][joint]
        pwm_per_degree = 1000 / 90.0  # 1000 PWM units per 90 degrees
        pwm = zero_pwm + direction * angle * pwm_per_degree
        return int(pwm)

    def set_leg_angles(self, leg_angles):
        legs = ['RB', 'RF', 'LF', 'LB']
        joints = ['hip', 'upper', 'lower']
        
        for i, leg in enumerate(legs):
            for j, joint in enumerate(joints):
                angle = leg_angles[i][j] * r2d  # Convert to degrees
                pwm = self.angle_to_pwm(leg, joint, angle)
                self.pwm.setServoPulse(self.channels[leg][joint], pwm)

def main():
    sm = SpotMicroStickFigure(x=0, y=0.10, z=0, phi=0, theta=0, psi=0)
    gait_controller = BezierGaitController(sm)
    hardware = SpotMicroHardware()

    try:
        while True:
            # Simulate key presses here or implement actual control method
            gait_controller.v_x = 0.1  # Example: constant forward motion
            gait_controller.moving = True

            gait_controller.update(0.05)  # 50ms update interval
            hardware.set_leg_angles(sm.get_leg_angles())
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopping the robot")
        # Set the robot to a neutral standing position
        hardware.set_leg_angles(gait_controller.initial_angles)

if __name__ == "__main__":
    main()