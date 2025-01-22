# main_gait_control.py

import numpy as np
import time
from math import pi, sqrt, atan2, acos
from PCA9685 import PCA9685
from spot_micro_kinematics.spot_micro_stick_figure import SpotMicroStickFigure

# Constants for angle conversions
d2r = pi / 180
r2d = 180 / pi

class BezierGaitController:
    def __init__(self, robot):
        self.robot = robot
        self.stance_height = -0.14
        self.step_length = 0.08
        self.step_height = 0.04
        self.phase = 0
        self.freq = 1.0
        self.v_x = 0.1  # Set forward velocity
        self.v_y = 0
        self.omega = 0
        self.moving = False  # Initially not moving
        self.initial_angles = self.get_initial_angles()

        # Initialize PCA9685
        self.pwm = PCA9685(address=0x40, debug=False)
        self.pwm.setPWMFreq(50)  # Set frequency to 50Hz

        # Servo channel mappings
        self.channels = {
            'LB': {'hip': 2, 'upper': 4, 'lower': 3},
            'RB': {'hip': 13, 'upper': 12, 'lower': 11},
            'LF': {'hip': 6, 'upper': 7, 'lower': 5},
            'RF': {'hip': 9, 'upper': 10, 'lower': 8}
        }

        # Base pulse widths and direction multipliers for each servo
        self.servo_mappings = {
            'LB': {'hip': (1000, -1), 'upper': (2200, -1), 'lower': (1000, -1)},
            'RB': {'hip': (1000, 1), 'upper': (1100, -1), 'lower': (2100, -1)},
            'LF': {'hip': (1950, 1), 'upper': (1500, -1), 'lower': (950, -1)},
            'RF': {'hip': (2100, -1), 'upper': (1150, -1), 'lower': (2000, -1)}
        }

        # PWM per degree conversion
        self.pwm_per_degree = 1000.0 / 90.0  # 1000µs for 90 degrees

    def get_initial_angles(self):
        return [
            [0, -30 * d2r, 60 * d2r],  # RB
            [0, -30 * d2r, 60 * d2r],  # RF
            [0, 30 * d2r, -60 * d2r],  # LF
            [0, 30 * d2r, -60 * d2r]   # LB
        ]

    def bezier_curve(self, t, P0, P1, P2, P3):
        return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3

    def leg_trajectory(self, phase, x_offset, y_offset):
        if not self.moving:
            return x_offset, y_offset, self.stance_height

        if phase < 0.5:  # Stance phase
            x = x_offset + self.step_length * (0.5 - phase)
            y = y_offset
            z = self.stance_height
        else:  # Swing phase
            t = (phase - 0.5) * 2
            P0 = np.array([x_offset - self.step_length / 2, y_offset, self.stance_height])
            P1 = np.array([x_offset - self.step_length / 4, y_offset, self.stance_height])
            P2 = np.array([x_offset + self.step_length / 4, y_offset, self.stance_height + self.step_height])
            P3 = np.array([x_offset + self.step_length / 2, y_offset, self.stance_height])
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

    def angle_to_pulse(self, angle_deg, servo, leg):
        """
        Convert an angle in degrees to a pulse width in microseconds for the specified servo.
        """
        base_pulse, direction = self.servo_mappings[leg][servo]
        delta_pulse = direction * angle_deg * self.pwm_per_degree
        pulse = base_pulse + delta_pulse

        # Clamp pulse to [500, 2500] µs to prevent extreme positions
        pulse = max(500, min(pulse, 2500))

        return pulse

    def update_pca9685(self, leg_angles):
        """
        Update the PCA9685 PWM signals based on the leg angles.
        """
        for leg, joints in leg_angles.items():
            for joint, angle in joints.items():
                servo = joint
                angle_deg = angle  # angle is already in degrees
                pulse = self.angle_to_pulse(angle_deg, servo, leg)
                channel = self.channels[leg][servo]
                self.pwm.setServoPulse(channel, pulse)
                # Optional: Print servo PWM settings for debugging
                # print(f"Setting {leg} {joint} to pulse {pulse}µs on channel {channel}")

    def inverse_kinematics(self, x, y, z, leg_index):
        # Leg dimensions
        l1 = self.robot.hip_length
        l2 = self.robot.upper_leg_length
        l3 = self.robot.lower_leg_length

        # Adjust for hip offset
        leg_names = ['RB', 'RF', 'LF', 'LB']
        leg_name = leg_names[leg_index]

        if leg_name in ['RB', 'RF']:  # Right legs
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
            print(f"Warning: Position out of reach for leg {leg_index} ({leg_name})")
            return self.initial_angles[leg_index]

        # Calculate angles for knee and ankle
        cos_ankle = (l2**2 + l3**2 - r**2) / (2 * l2 * l3)
        cos_ankle = max(min(cos_ankle, 1), -1)  # Clamp to [-1, 1]
        ankle_angle = acos(cos_ankle)

        a1 = atan2(z, d)
        a2 = acos((l2**2 + r**2 - l3**2) / (2 * l2 * r))
        knee_angle = a1 + a2

        # Adjust angles based on leg side and orientation
        if leg_name in ['RB', 'RF']:  # Right legs
            hip_angle = -hip_angle  # Negate for right side
            knee_angle = pi / 2 - knee_angle
            ankle_angle = ankle_angle - pi
        else:  # Left legs
            knee_angle = knee_angle - pi / 2
            ankle_angle = pi - ankle_angle

        # Clamp angles to reasonable ranges
        hip_angle = max(min(hip_angle, pi / 4), -pi / 4)  # Limit hip angle to ±45 degrees
        knee_angle = max(min(knee_angle, pi / 2), -pi / 2)
        ankle_angle = max(min(ankle_angle, pi), -pi)

        # Convert angles from radians to degrees
        hip_deg = hip_angle * r2d
        knee_deg = knee_angle * r2d
        ankle_deg = ankle_angle * r2d

        return hip_deg, knee_deg, ankle_deg

    def set_initial_position(self):
        """
        Set all servos to the initial stable position.
        """
        leg_angles_deg = {
            'RB': {'hip': self.initial_angles[0][0] * r2d, 'upper': self.initial_angles[0][1] * r2d, 'lower': self.initial_angles[0][2] * r2d},
            'RF': {'hip': self.initial_angles[1][0] * r2d, 'upper': self.initial_angles[1][1] * r2d, 'lower': self.initial_angles[1][2] * r2d},
            'LF': {'hip': self.initial_angles[2][0] * r2d, 'upper': self.initial_angles[2][1] * r2d, 'lower': self.initial_angles[2][2] * r2d},
            'LB': {'hip': self.initial_angles[3][0] * r2d, 'upper': self.initial_angles[3][1] * r2d, 'lower': self.initial_angles[3][2] * r2d}
        }
        self.update_pca9685(leg_angles_deg)
        self.robot.set_leg_angles([
            [angle * d2r for angle in leg]
            for leg in self.initial_angles
        ])

    def update(self, dt):
        if self.moving:
            self.phase += self.freq * dt
            if self.phase >= 1:
                self.phase -= 1

            # Calculate foot positions
            rf = self.leg_trajectory(self.phase, self.robot.body_length / 2, self.robot.body_width / 2)
            lf = self.leg_trajectory((self.phase + 0.5) % 1, self.robot.body_length / 2, -self.robot.body_width / 2)
            rb = self.leg_trajectory((self.phase + 0.5) % 1, -self.robot.body_length / 2, self.robot.body_width / 2)
            lb = self.leg_trajectory(self.phase, -self.robot.body_length / 2, -self.robot.body_width / 2)

            # Adjust for forward motion
            motion_adjust = self.v_x * dt
            rf = (rf[0] + motion_adjust, rf[1], rf[2])
            lf = (lf[0] + motion_adjust, lf[1], lf[2])
            rb = (rb[0] + motion_adjust, rb[1], rb[2])
            lb = (lb[0] + motion_adjust, lb[1], lb[2])

            # Calculate leg angles using inverse kinematics
            rb_angles_deg = self.inverse_kinematics(*rb, 0)
            rf_angles_deg = self.inverse_kinematics(*rf, 1)
            lf_angles_deg = self.inverse_kinematics(*lf, 2)
            lb_angles_deg = self.inverse_kinematics(*lb, 3)

            # Create a dictionary of angles for PWM update
            leg_angles = {
                'RB': {'hip': rb_angles_deg[0], 'upper': rb_angles_deg[1], 'lower': rb_angles_deg[2]},
                'RF': {'hip': rf_angles_deg[0], 'upper': rf_angles_deg[1], 'lower': rf_angles_deg[2]},
                'LF': {'hip': lf_angles_deg[0], 'upper': lf_angles_deg[1], 'lower': lf_angles_deg[2]},
                'LB': {'hip': lb_angles_deg[0], 'upper': lb_angles_deg[1], 'lower': lb_angles_deg[2]}
            }

            # Set leg angles in the robot model (assumed to be in radians)
            self.robot.set_leg_angles([
                [rb_angles_deg[0] * d2r, rb_angles_deg[1] * d2r, rb_angles_deg[2] * d2r],
                [rf_angles_deg[0] * d2r, rf_angles_deg[1] * d2r, rf_angles_deg[2] * d2r],
                [lf_angles_deg[0] * d2r, lf_angles_deg[1] * d2r, lf_angles_deg[2] * d2r],
                [lb_angles_deg[0] * d2r, lb_angles_deg[1] * d2r, lb_angles_deg[2] * d2r]
            ])

            # Update PWM signals
            self.update_pca9685(leg_angles)

            # Update robot position and orientation
            self.robot.x += self.v_x * dt
            self.robot.y += self.v_y * dt
            self.robot.psi += self.omega * dt
        else:
            # Set to initial angles
            self.robot.set_leg_angles([
                [angle * d2r for angle in leg]
                for leg in self.initial_angles
            ])
            # Send initial angles to PWM
            leg_angles_deg_initial = {
                'RB': {'hip': self.initial_angles[0][0] * r2d, 'upper': self.initial_angles[0][1] * r2d, 'lower': self.initial_angles[0][2] * r2d},
                'RF': {'hip': self.initial_angles[1][0] * r2d, 'upper': self.initial_angles[1][1] * r2d, 'lower': self.initial_angles[1][2] * r2d},
                'LF': {'hip': self.initial_angles[2][0] * r2d, 'upper': self.initial_angles[2][1] * r2d, 'lower': self.initial_angles[2][2] * r2d},
                'LB': {'hip': self.initial_angles[3][0] * r2d, 'upper': self.initial_angles[3][1] * r2d, 'lower': self.initial_angles[3][2] * r2d}
            }
            self.update_pca9685(leg_angles_deg_initial)

def main():
    # Instantiate SpotMicroStickFigure
    sm = SpotMicroStickFigure(x=0, y=0.1, z=0, phi=0, theta=0, psi=0)

    # Create gait controller
    gait_controller = BezierGaitController(sm)

    # Initialize to initial position
    print("Setting initial stable position...")
    gait_controller.set_initial_position()
    time.sleep(1)  # 1 second to stabilize

    # Start moving forward
    print("Starting forward movement...")
    gait_controller.moving = True
    start_time = time.time()

    try:
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # After 3 seconds, stop moving
            if elapsed_time >= 3.0 and gait_controller.moving:
                print("3 seconds elapsed. Stopping movement and returning to stable position.")
                gait_controller.moving = False

            # Update the gait controller
            gait_controller.update(0.02)  # dt = 20ms

            # Get current leg angles in radians
            leg_angles_rad = sm.get_leg_angles()
            leg_names = ['RB', 'RF', 'LF', 'LB']

            # Convert angles to degrees for printing
            leg_angles_deg = [
                [angle * r2d for angle in leg]
                for leg in leg_angles_rad
            ]

            # Print angle tuples
            time_sec = elapsed_time
            print(f"\nTime: {time_sec:.2f}s")
            print("Leg Angles (degrees):")
            print("    Hip     Upper    Lower")
            for i, leg in enumerate(leg_names):
                hip, upper, lower = leg_angles_deg[i]
                print(f"{leg}: {hip:6.2f} {upper:7.2f} {lower:7.2f}")

            # Check if movement has stopped and returned to stable position
            if not gait_controller.moving and elapsed_time >= 3.0:
                break

            # Sleep for 20ms to simulate 50Hz control loop
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nStopping gait control due to KeyboardInterrupt...")

    finally:
        # Ensure the robot returns to initial stable position
        if gait_controller.moving:
            gait_controller.moving = False
            gait_controller.update(0.02)
        gait_controller.set_initial_position()
        # Optionally, stop all PWM signals to servos
        gait_controller.pwm.setPWMFreq(0)
        print("Gait control stopped and servos turned off.")

if __name__ == "__main__":
    main()