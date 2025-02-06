import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
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
        self.stance_height = -0.16
        self.step_length = 0.08
        self.step_height = 0.04
        self.phase = 0
        self.freq = 1.0
        self.v_x = 0
        self.v_y = 0
        self.omega = 0
        self.moving = False
        self.initial_angles = self.get_initial_angles()

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

        self.sideways_step_length = 0.03
        self.turn_radius = 0.2

        # Initialize PCA9685
        self.pca = PCA9685(address=0x40, debug=False)
        self.pca.setPWMFreq(50)  # Set frequency to 50Hz

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

        x = 0
        y = 0
        z = self.stance_height
        hip_angle = 0

        if phase < swing_phase:
            t = phase / swing_phase
            z = self.stance_height + self.step_height * sin(pi * t)
            x = self.step_length * (1 - cos(pi * t)) / 2 - self.step_length / 4
        else:
            t = (phase - swing_phase) / stance_phase
            z = self.stance_height
            x = (self.step_length / 4) - (self.step_length * t / 2)

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

        upper_ratio = 0.7
        lower_ratio = 1 - upper_ratio

        upper_angle = upper_angle * upper_ratio
        knee_angle = knee_angle * lower_ratio

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
                self.pca.setServoPulse(channel, pulse)

    def update(self, dt):
        if self.moving:
            self.phase += self.freq * dt
            if self.phase >= 1:
                self.phase -= 1

            phases = {leg: (self.phase + self.phase_offsets[leg]) % 1 for leg in ['RB', 'RF', 'LF', 'LB']}
            foot_positions = {leg: self.leg_trajectory(phases[leg], leg) for leg in ['RB', 'RF', 'LF', 'LB']}

            rb_angles = self.inverse_kinematics(*foot_positions['RB'][:3], 0, foot_positions['RB'][3])
            rf_angles = self.inverse_kinematics(*foot_positions['RF'][:3], 1, foot_positions['RF'][3])
            lf_angles = self.inverse_kinematics(*foot_positions['LF'][:3], 2, foot_positions['LF'][3])
            lb_angles = self.inverse_kinematics(*foot_positions['LB'][:3], 3, foot_positions['LB'][3])

            self.robot.set_leg_angles([rb_angles, rf_angles, lf_angles, lb_angles])
            pwm_duty_cycles = self.calculate_pwm_duty_cycles([rb_angles, rf_angles, lf_angles, lb_angles])
            self.set_servo_pulses(pwm_duty_cycles)

            self.robot.x += self.v_x * cos(self.robot.psi) * dt
            self.robot.y += self.v_x * sin(self.robot.psi) * dt

            return pwm_duty_cycles
        else:
            self.robot.set_leg_angles(self.initial_angles)
            pwm_duty_cycles = self.calculate_pwm_duty_cycles(self.initial_angles)
            self.set_servo_pulses(pwm_duty_cycles)
            return pwm_duty_cycles

    def set_motion(self, forward=0.0, sideways=0.0, turn=0.0):
        self.v_x = forward
        self.v_y = sideways
        self.omega = turn
        self.moving = any([forward, sideways, turn])

def update_pose(frame, sm, gait_controller, lines, angle_lines, angle_data, ax, angle_ax, start_time):
    dt = 0.02

    current_time = time.time() - start_time
    if current_time <= 10:
        gait_controller.set_motion(forward=0.025)
    else:
        gait_controller.set_motion()

    pwm_duty_cycles = gait_controller.update(dt)

    coords = sm.get_leg_coordinates()

    body_points = [coord[0] for coord in coords] + [coords[0][0]]
    lines[0].set_data([p[0] for p in body_points], [p[1] for p in body_points])
    lines[0].set_3d_properties([p[2] for p in body_points])

    for i, leg in enumerate(coords):
        lines[i+1].set_data([p[0] for p in leg], [p[1] for p in leg])
        lines[i+1].set_3d_properties([p[2] for p in leg])

    ax.set_xlim(sm.x - 0.3, sm.x + 0.3)
    ax.set_ylim(sm.y - 0.3, sm.y + 0.3)
    ax.set_zlim(0, 0.4)

    leg_angles = sm.get_leg_angles()
    current_angles = np.array(leg_angles).flatten() * r2d
    angle_data.append(current_angles)

    leg_names = ['RB', 'RF', 'LF', 'LB']
    joint_names = ['hip', 'upper', 'lower']
    
    print(f"\nFrame: {frame}")
    print("Leg Angles (degrees) and PWM Duty Cycles:")
    print("    Hip    Upper   Lower")
    for i, leg in enumerate(leg_names):
        angles = current_angles[i*3:(i+1)*3]
        print(f"{leg}: {angles[0]:6.2f} {angles[1]:6.2f} {angles[2]:6.2f}")
        pwm_values = [pwm_duty_cycles.get(f'{leg}_{joint}', 1500) for joint in joint_names]
        print(f"PWM: {pwm_values[0]:5d} {pwm_values[1]:5d} {pwm_values[2]:5d}")

    angle_data_array = np.array(angle_data)
    for i in range(12):
        angle_lines[i].set_data(range(len(angle_data)), angle_data_array[:, i])

    angle_ax.relim()
    angle_ax.autoscale_view()

    return lines + angle_lines

def main():
    sm = SpotMicroStickFigure(x=0, y=0.16, z=0, phi=0, theta=0, psi=0)
    gait_controller = SimpleGaitController(sm)

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(211, projection='3d')
    angle_ax = fig.add_subplot(212)

    lines = [ax.plot([], [], [], 'k-', linewidth=2)[0]]
    colors = ['r', 'g', 'b', 'y']
    for color in colors:
        lines.append(ax.plot([], [], [], f'{color}-', linewidth=1)[0])

    angle_lines = []
    leg_names = ['RB_Hip', 'RB_Upper', 'RB_Lower',
                 'RF_Hip', 'RF_Upper', 'RF_Lower',
                 'LF_Hip', 'LF_Upper', 'LF_Lower',
                 'LB_Hip', 'LB_Upper', 'LB_Lower']
    for name in leg_names:
        line, = angle_ax.plot([], [], label=name)
        angle_lines.append(line)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('SpotMicroStickFigure Gait (Moving forward for 10 seconds)')

    angle_ax.set_xlabel('Frame')
    angle_ax.set_ylabel('Angle (degrees)')
    angle_ax.set_title('Leg Joint Angles')
    angle_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0, 0.4)

    plt.tight_layout()

    angle_data = []

    start_time = time.time()

    anim = animation.FuncAnimation(
        fig,
        update_pose,
        frames=500,
        fargs=(sm, gait_controller, lines, angle_lines, angle_data, ax, angle_ax, start_time),
        interval=20,
        blit=False
    )

    plt.show()

    print("Animation complete. If you can see this, the script has run to completion.")

if __name__ == "__main__":
    main()