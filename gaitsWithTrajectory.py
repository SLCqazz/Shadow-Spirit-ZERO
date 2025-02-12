from spot_micro_kinematics.spot_micro_stick_figure import SpotMicroStickFigure
from math import pi, sin, cos, sqrt, atan2, acos
import time
from PCA9685 import PCA9685
import numpy as np

# Conversion constants
d2r = pi / 180
r2d = 180 / pi

class SimpleGaitController:
    def __init__(self, robot):
        self.robot = robot
        self.stance_height = -0.18
        self.step_length = 0.06  
        self.step_height = 0.08  
        self.phase = 0
        self.freq = 1.0
        self.initial_angles = self.get_initial_angles()
        self.current_waypoint = 0
        self.total_waypoints = 18
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
            'RF': 0.5,
            'LF': 0.0,
            'LB': 0.5
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

        # 定义非对称参数
        a = 0.6

        if phase < swing_phase:
            t = phase / swing_phase
            # 应用非线性缩放以获得非对称轨迹
            t_scaled = t**a
            z = self.stance_height + self.step_height * sin(pi * t_scaled)
            x = self.step_length * (1 - cos(pi * t_scaled)) / 2 - self.step_length / 4
        else:
            t = (phase - swing_phase) / stance_phase
            z = self.stance_height
            forward_offset = 0.02  # 添加前向偏移
            x = (self.step_length / 4) - (self.step_length * t / 2) + forward_offset

        return x, 0, z

    def inverse_kinematics(self, x, y, z, leg_index):
        l1 = self.robot.hip_length
        l2 = self.robot.upper_leg_length
        l3 = self.robot.lower_leg_length

        # Calculate hip angle
        hip_angle = atan2(y, x)

        # Adjust x and y for hip rotation
        x_adj = sqrt(x**2 + y**2) - l1
        y_adj = z

        # Now use adjusted coordinates to calculate other angles
        d = sqrt(x_adj**2 + y_adj**2)

        if d > l2 + l3:
            print(f"Warning: Position out of reach for leg {leg_index}")
            return self.initial_angles[leg_index]

        cos_knee = (l2**2 + l3**2 - d**2) / (2 * l2 * l3)
        cos_knee = max(min(cos_knee, 1), -1)
        knee_angle = pi - acos(cos_knee)

        alpha = atan2(y_adj, x_adj)
        beta = acos((l2**2 + d**2 - l3**2) / (2 * l2 * d))
        upper_angle = alpha + beta

        if leg_index in [2, 3]:  # Left side legs
            upper_angle = -upper_angle
            knee_angle = -knee_angle

        hip_angle = max(min(hip_angle, self.max_angles['hip']), -self.max_angles['hip'])
        upper_angle = max(min(upper_angle, self.max_angles['upper']), -self.max_angles['upper'])
        knee_angle = max(min(knee_angle, self.max_angles['lower']), -self.max_angles['lower'])

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

        rb_angles = self.inverse_kinematics(*foot_positions['RB'], 0)
        rf_angles = self.inverse_kinematics(*foot_positions['RF'], 1)
        lf_angles = self.inverse_kinematics(*foot_positions['LF'], 2)
        lb_angles = self.inverse_kinematics(*foot_positions['LB'], 3)

        self.robot.set_leg_angles([rb_angles, rf_angles, lf_angles, lb_angles])
        pwm_duty_cycles = self.calculate_pwm_duty_cycles([rb_angles, rf_angles, lf_angles, lb_angles])
        self.set_servo_pulses(pwm_duty_cycles)

        self.current_waypoint = (self.current_waypoint + 1) % self.total_waypoints

        return [rb_angles, rf_angles, lf_angles, lb_angles], pwm_duty_cycles, foot_positions

def main():
    sm = SpotMicroStickFigure(x=0, y=0.16, z=0, phi=0, theta=0, psi=0)
    gait_controller = SimpleGaitController(sm)

    try:
        while True:
            input("Press Enter to move to the next waypoint...")
            leg_angles, pwm_duty_cycles, foot_positions = gait_controller.update()

            print("\nLeg Angles (degrees), PWM Duty Cycles, and Foot Positions:")
            print("    Hip    Upper   Lower      PWM Hip  PWM Upper PWM Lower    X       Y       Z")
            leg_names = ['RB', 'RF', 'LF', 'LB']
            joint_names = ['hip', 'upper', 'lower']
            for i, leg in enumerate(leg_names):
                angles = [angle * r2d for angle in leg_angles[i]]
                pwm_values = [pwm_duty_cycles.get(f'{leg}_{joint}', 1500) for joint in joint_names]
                initial_pos = gait_controller.initial_foot_positions[leg]
                current_pos = foot_positions[leg]
                relative_pos = tuple(current - initial for current, initial in zip(current_pos, initial_pos))
                print(f"{leg}: {angles[0]:6.2f} {angles[1]:6.2f} {angles[2]:6.2f}   {pwm_values[0]:5d} {pwm_values[1]:5d} {pwm_values[2]:5d}   {relative_pos[0]:6.3f} {relative_pos[1]:6.3f} {relative_pos[2]:6.3f}")

    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        print("Program completed")

if __name__ == "__main__":
    main()