from spot_micro_kinematics.spot_micro_stick_figure import SpotMicroStickFigure
from math import pi, sin, cos, sqrt, atan2, acos
import time
from PCA9685 import PCA9685

# Conversion constants
d2r = pi / 180
r2d = 180 / pi

class SimpleGaitController:
    def __init__(self, robot):
        self.robot = robot
        self.stance_height = -0.20
        self.step_length = 0.1
        self.step_height = 0.08
        self.phase = 0
        self.freq = 1.0
        self.initial_angles = self.get_initial_angles()
        self.current_waypoint = 0
        self.total_waypoints = 18

        self.servo_mappings = {
            'LB': {'hip': (1000, -1), 'upper': (2200, -1), 'lower': (1000, -1)},
            'RB': {'hip': (1000, 1), 'upper': (1100, -1), 'lower': (2100, -1)},
            'LF': {'hip': (1950, 1), 'upper': (1500, -1), 'lower': (950, -1)},
            'RF': {'hip': (2100, -1), 'upper': (1150, -1), 'lower': (2000, -1)}
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

    def leg_trajectory(self, phase, leg):
        swing_phase = 0.4
        stance_phase = 1.0 - swing_phase

        # Get initial position
        initial_x, initial_y, initial_z = 0, 0, self.stance_height
        initial_hip_angle = 0

        # Calculate target position
        if phase < swing_phase:
            t = phase / swing_phase
            target_z = self.stance_height + self.step_height * sin(pi * t)
            target_x = self.step_length * (1 - cos(pi * t)) / 2 - self.step_length / 4
            target_y = 0
            target_hip_angle = 0
        else:
            t = (phase - swing_phase) / stance_phase
            target_z = self.stance_height
            target_x = (self.step_length / 4) - (self.step_length * t / 2)
            target_y = 0
            target_hip_angle = 0

        # Apply smooth transition
        transition_factor = min(self.phase * 5, 1.0)  # Transition over 0.2 seconds (1 / 5)
        x = initial_x + (target_x - initial_x) * transition_factor
        y = initial_y + (target_y - initial_y) * transition_factor
        z = initial_z + (target_z - initial_z) * transition_factor
        hip_angle = initial_hip_angle + (target_hip_angle - initial_hip_angle) * transition_factor

        return x, y, z, hip_angle

    def inverse_kinematics(self, x, y, z, leg_index, hip_angle):
        l1 = self.robot.hip_length
        l2 = self.robot.upper_leg_length
        l3 = self.robot.lower_leg_length

        x_adj = x * cos(hip_angle) - y * sin(hip_angle)
        y_adj = x * sin(hip_angle) + y * cos(hip_angle)

        x_plane = sqrt(x_adj**2 + y_adj**2) - l1
        z_plane = z
        d = sqrt(x_plane**2 + z_plane**2)

        if d > l2 + l3:
            print(f"Warning: Position out of reach for leg {leg_index}")
            return self.initial_angles[leg_index]

        cos_knee = (l2**2 + l3**2 - d**2) / (2 * l2 * l3)
        cos_knee = max(min(cos_knee, 1), -1)
        knee_angle = pi - acos(cos_knee)

        alpha = atan2(z_plane, x_plane)
        beta = acos((l2**2 + d**2 - l3**2) / (2 * l2 * d))
        upper_angle = alpha + beta

        if leg_index in [2, 3]:
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

        rb_angles = self.inverse_kinematics(*foot_positions['RB'][:3], 0, foot_positions['RB'][3])
        rf_angles = self.inverse_kinematics(*foot_positions['RF'][:3], 1, foot_positions['RF'][3])
        lf_angles = self.inverse_kinematics(*foot_positions['LF'][:3], 2, foot_positions['LF'][3])
        lb_angles = self.inverse_kinematics(*foot_positions['LB'][:3], 3, foot_positions['LB'][3])

        self.robot.set_leg_angles([rb_angles, rf_angles, lf_angles, lb_angles])
        pwm_duty_cycles = self.calculate_pwm_duty_cycles([rb_angles, rf_angles, lf_angles, lb_angles])
        self.set_servo_pulses(pwm_duty_cycles)

        self.current_waypoint = (self.current_waypoint + 1) % self.total_waypoints

        return [rb_angles, rf_angles, lf_angles, lb_angles], pwm_duty_cycles, foot_positions

def main():
    sm = SpotMicroStickFigure(x=0, y=0.18, z=0, phi=0, theta=0, psi=0)
    gait_controller = SimpleGaitController(sm)

    try:
        while True:
            input("Press Enter to move to the next waypoint...")
            leg_angles, pwm_duty_cycles, foot_positions = gait_controller.update()

            print("\nLeg Angles (degrees), PWM Duty Cycles, and Foot Positions:")
            print("    Hip    Upper   Lower      PWM Hip  PWM Upper  PWM Lower    X       Y       Z")
            leg_names = ['RB', 'RF', 'LF', 'LB']
            joint_names = ['hip', 'upper', 'lower']
            for i, leg in enumerate(leg_names):
                angles = [angle * r2d for angle in leg_angles[i]]
                pwm_values = [pwm_duty_cycles.get(f'{leg}_{joint}', 1500) for joint in joint_names]
                foot_pos = foot_positions[leg][:3]
                print(f"{leg}: {angles[0]:6.2f} {angles[1]:6.2f} {angles[2]:6.2f}   {pwm_values[0]:5d}     {pwm_values[1]:5d}     {pwm_values[2]:5d}    {foot_pos[0]:6.3f} {foot_pos[1]:6.3f} {foot_pos[2]:6.3f}")

    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        print("Program completed")

if __name__ == "__main__":
    main()
