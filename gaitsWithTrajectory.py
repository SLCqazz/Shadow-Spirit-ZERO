from spot_micro_kinematics.spot_micro_stick_figure import SpotMicroStickFigure
from math import pi, sin, cos, sqrt, atan2, acos
import time
from PCA9685 import PCA9685
import numpy as np
import sys
import select
import tty
import termios

# Conversion constants
d2r = pi / 180
r2d = 180 / pi

def getkey():
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                key = sys.stdin.read(1)
                return key
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

class SimpleGaitController:
    def __init__(self, robot):
        self.robot = robot
        self.stance_height = 0.14
        self.step_length = 0.04
        self.step_height = 0.08
        self.phase = 0
        self.freq = 1.0
        self.direction = 0  # 0 for stationary, 1 for forward, -1 for backward
        self.initial_angles = self.get_initial_angles()
        self.current_waypoint = 0
        self.total_waypoints = 16
        self.initial_foot_positions = self.get_initial_foot_positions()

        self.servo_mappings = {
            'LB': {'hip': (1000, -1), 'upper': (2250, -1), 'lower': (950, -1)},
            'RB': {'hip': (950, 1), 'upper': (1100, -1), 'lower': (1950, -1)},
            'LF': {'hip': (1900, 1), 'upper': (1500, -1), 'lower': (950, -1)},
            'RF': {'hip': (2100, -1), 'upper': (1350, -1), 'lower': (1900, -1)}
        }

        self.pwm_per_degree = 1000.0 / 90.0

        self.phase_offsets = {
            'RB': 0.0,
            'RF': 0.4,
            'LF': 0.0,
            'LB': 0.4
        }

        self.max_angles = {
            'hip': 30 * d2r,
            'upper': 90 * d2r,
            'lower': 120 * d2r
        }

        # Initialize PCA9685
        self.pwm = PCA9685(address=0x40, debug=False)
        self.pwm.setPWMFreq(330)  # Set frequency to 330Hz

        # Servo channel mappings
        self.channels = {
            'LB': {'hip': 2, 'upper': 4, 'lower': 3},
            'RB': {'hip': 13, 'upper': 12, 'lower': 11},
            'LF': {'hip': 6, 'upper': 7, 'lower': 5},
            'RF': {'hip': 9, 'upper': 10, 'lower': 8}
        }

    def get_initial_angles(self):
        return [
            [0, -45 * d2r, 60 * d2r],  # RB
            [0, -45 * d2r, 60 * d2r],  # RF
            [0, 45 * d2r, -60 * d2r],  # LF
            [0, 45 * d2r, -60 * d2r]   # LB
        ]

    def get_initial_foot_positions(self):
        coords = self.robot.get_leg_coordinates()
        return {leg: coords[i][-1] for i, leg in enumerate(['RB', 'RF', 'LF', 'LB'])}

    def leg_trajectory(self, phase, leg):
        swing_phase = 0.6
        stance_phase = 1.0 - swing_phase

        a = 0.8

        # 反转相位
        reversed_phase = (phase + 0.5) % 1.0

        if self.direction == 0:  # 原地踏步
            if reversed_phase < swing_phase:
                t = reversed_phase / swing_phase
                t_scaled = t**a
                z = self.stance_height + self.step_height * sin(pi * t_scaled)
                x = 0
            else:
                z = self.stance_height
                x = 0
        else:  # 前进或后退
            if reversed_phase < swing_phase:
                t = reversed_phase / swing_phase
                t_scaled = t**a
                z = self.stance_height + self.step_height * sin(pi * t_scaled)
                x = self.direction * self.step_length * (1 - cos(pi * t_scaled)) / 2
            else:
                t = (reversed_phase - swing_phase) / stance_phase
                z = self.stance_height
                x = self.direction * self.step_length * (0.5 - t)

        y = 0  # 假设没有横向运动
        hip_angle = 0  # 假设hip角度保持不变

        return x, y, z, hip_angle

    def inverse_kinematics(self, x, y, z, leg_index, hip_angle):
        config = self.robot
        h = config.hip_length
        hu = config.upper_leg_length
        hl = config.lower_leg_length

        dyz = sqrt(y**2 + z**2)
        if dyz < h:
            print(f"Leg {leg_index}: Warning - dyz ({dyz:.3f}) < h ({h:.3f}), setting lyz to small positive value to avoid sqrt of negative.")
            lyz = 1e-6
        else:
            lyz = sqrt(dyz**2 - h**2)

        gamma_yz = -atan2(y, z)
        gamma_h_offset = -atan2(h, lyz)
        gamma = gamma_yz - gamma_h_offset

        lxzp = sqrt(lyz**2 + x**2)
        n_numerator = lxzp**2 - hl**2 - hu**2
        n_denominator = 2 * hu
        n = n_numerator / n_denominator

        try:
            beta_angle = -acos(max(min(n / hl, 1.0), -1.0))
        except ValueError:
            beta_angle = 0.0
            print(f"Leg {leg_index}: Warning - Invalid value for beta, setting to 0")

        alfa_xzp = -atan2(x, lyz)
        try:
            alfa_off = acos(max(min((hu + n) / lxzp, 1.0), -1.0))
        except ValueError:
            alfa_off = 0.0
            print(f"Leg {leg_index}: Warning - Invalid value for alfa_off, setting to 0")

        alfa = alfa_xzp + alfa_off

        if leg_index in [2, 3]:  # LF and LB legs
            alfa = -alfa
            beta_angle = -beta_angle

        hip_angle = max(min(gamma, self.max_angles['hip']), -self.max_angles['hip'])
        upper_angle = max(min(alfa, self.max_angles['upper']), -self.max_angles['upper'])
        knee_angle = max(min(beta_angle, self.max_angles['lower']), -self.max_angles['lower'])

        return hip_angle, upper_angle, knee_angle

    def calculate_pwm_duty_cycles(self, leg_angles):
        leg_names = ['RB', 'RF', 'LF', 'LB']
        joint_names = ['hip', 'upper', 'lower']
        pwm_duty_cycles = {}

        for leg, angles in zip(leg_names, leg_angles):
            angles_deg = [angle * r2d for angle in angles]
            for joint, angle_deg in zip(joint_names, angles_deg):
                base_pulse, direction = self.servo_mappings[leg][joint]
                pwm_change = direction * angle_deg * self.pwm_per_degree
                pwm = base_pulse + pwm_change
                pwm = max(500, min(2500, pwm))
                pwm_duty_cycles[f"{leg}_{joint}"] = int(pwm)

        return pwm_duty_cycles

    def set_servo_pulses(self, pwm_duty_cycles):
        for leg, joints in self.channels.items():
            for joint, channel in joints.items():
                pulse = pwm_duty_cycles[f"{leg}_{joint}"]
                self.pwm.setServoPulse(channel, pulse)

    def update(self):
        self.phase = self.current_waypoint / self.total_waypoints
        phases = {leg: (self.phase + self.phase_offsets[leg]) % 1 for leg in ['RB', 'RF', 'LF', 'LB']}
        foot_positions = {leg: self.leg_trajectory(phases[leg], leg) for leg in ['RB', 'RF', 'LF', 'LB']}

        rb_angles = self.inverse_kinematics(*foot_positions['RB'][:3], 0, foot_positions['RB'][3])
        rf_angles = self.inverse_kinematics(*foot_positions['RF'][:3], 1, foot_positions['RF'][3])
        lf_angles = self.inverse_kinematics(*foot_positions['LF'][:3], 2, foot_positions['LF'][3])
        lb_angles = self.inverse_kinematics(*foot_positions['LB'][:3], 3, foot_positions['LB'][3])

        # 左右对调
        rb_angles, lb_angles = lb_angles, rb_angles
        rf_angles, lf_angles = lf_angles, rf_angles

        self.robot.set_leg_angles([rb_angles, rf_angles, lf_angles, lb_angles])
        pwm_duty_cycles = self.calculate_pwm_duty_cycles([rb_angles, rf_angles, lf_angles, lb_angles])
        self.set_servo_pulses(pwm_duty_cycles)

        self.current_waypoint = (self.current_waypoint + 1) % self.total_waypoints

        return [rb_angles, rf_angles, lf_angles, lb_angles], pwm_duty_cycles, foot_positions

    def initialize_position(self):
        print("Initializing position...")
        initial_angles = self.get_initial_angles()
        
        # Set upper angles first
        upper_angles = [[0, angles[1], 0] for angles in initial_angles]
        pwm_duty_cycles = self.calculate_pwm_duty_cycles(upper_angles)
        self.set_servo_pulses(pwm_duty_cycles)
        print("Upper angles set. Waiting 1 second...")
        time.sleep(1)

        # Set lower angles
        pwm_duty_cycles = self.calculate_pwm_duty_cycles(initial_angles)
        self.set_servo_pulses(pwm_duty_cycles)
        print("Lower angles set. Initialization complete.")

    def move_forward(self):
        self.direction = 1

    def move_backward(self):
        self.direction = -1

    def stay_stationary(self):
        self.direction = 0

def main():
    sm = SpotMicroStickFigure(x=0, y=0.16, z=0, phi=0, theta=0, psi=0)
    gait_controller = SimpleGaitController(sm)

    # Initialize position
    gait_controller.initialize_position()

    print("Use arrow keys to control the robot:")
    print("Up: Move forward")
    print("Down: Move backward")
    print("Space: Stay stationary")
    print("Q: Quit")

    try:
        while True:
            key = getkey()
            if key == 'q':
                break
            elif key == '\x1b[A':  # Up arrow
                gait_controller.move_forward()
                print("Moving forward")
            elif key == '\x1b[B':  # Down arrow
                gait_controller.move_backward()
                print("Moving backward")
            elif key == ' ':  # Space
                gait_controller.stay_stationary()
                print("Staying stationary")

            leg_angles, pwm_duty_cycles, foot_positions = gait_controller.update()

            print("\nLeg Angles (degrees), PWM Duty Cycles, and Foot Positions:")
            print("    Hip    Upper   Lower      PWM Hip  PWM Upper PWM Lower    X       Y       Z")
            leg_names = ['RB', 'RF', 'LF', 'LB']
            joint_names = ['hip', 'upper', 'lower']
            for i, leg in enumerate(leg_names):
                angles = [angle * r2d for angle in leg_angles[i]]
                pwm_values = [pwm_duty_cycles.get(f'{leg}_{joint}', 1500) for joint in joint_names]
                initial_pos = gait_controller.initial_foot_positions[leg]
                current_pos = foot_positions[leg][:3]
                relative_pos = tuple(current - initial for current, initial in zip(current_pos, initial_pos))
                print(f"{leg}: {angles[0]:6.2f} {angles[1]:6.2f} {angles[2]:6.2f}   {pwm_values[0]:5d} {pwm_values[1]:5d} {pwm_values[2]:5d}   {relative_pos[0]:6.3f} {relative_pos[1]:6.3f} {relative_pos[2]:6.3f}")

    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        print("Program completed")

if __name__ == "__main__":
    main()