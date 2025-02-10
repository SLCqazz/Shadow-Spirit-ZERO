import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from spot_micro_kinematics.spot_micro_stick_figure import SpotMicroStickFigure
from math import pi, sin, cos, sqrt, atan2, acos
from pynput import keyboard

# Conversion constants
d2r = pi / 180
r2d = 180 / pi

class SimpleGaitController:
    def __init__(self, robot):
        self.robot = robot
        self.stance_height = -0.16
        self.step_length = 0.1
        self.step_height = 0.08
        self.phase = 0
        self.freq = 1.0
        self.v_x = 0
        self.v_y = 0
        self.omega = 0
        self.moving = False
        self.initial_angles = self.get_initial_angles()
        self.current_waypoint = 0
        self.total_waypoints = 18  # 将一个完整的步态周期分为8个途径点
        self.leg_trajectories = {leg: [] for leg in ['RB', 'RF', 'LF', 'LB']}
        self.initial_foot_positions = self.get_initial_foot_positions()

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

    def update(self):
        if self.moving:
            self.phase = self.current_waypoint / self.total_waypoints
            phases = {leg: (self.phase + self.phase_offsets[leg]) % 1 for leg in ['RB', 'RF', 'LF', 'LB']}
            foot_positions = {leg: self.leg_trajectory(phases[leg], leg) for leg in ['RB', 'RF', 'LF', 'LB']}

            rb_angles = self.inverse_kinematics(*foot_positions['RB'][:3], 0, foot_positions['RB'][3])
            rf_angles = self.inverse_kinematics(*foot_positions['RF'][:3], 1, foot_positions['RF'][3])
            lf_angles = self.inverse_kinematics(*foot_positions['LF'][:3], 2, foot_positions['LF'][3])
            lb_angles = self.inverse_kinematics(*foot_positions['LB'][:3], 3, foot_positions['LB'][3])

            self.robot.set_leg_angles([rb_angles, rf_angles, lf_angles, lb_angles])

            for leg in ['RB', 'RF', 'LF', 'LB']:
                self.leg_trajectories[leg].append(foot_positions[leg][:3])

            self.current_waypoint = (self.current_waypoint + 1) % self.total_waypoints
            self.moving = False

            return [rb_angles, rf_angles, lf_angles, lb_angles], foot_positions
        else:
            return self.initial_angles, {leg: (0, 0, self.stance_height) for leg in ['RB', 'RF', 'LF', 'LB']}

    def move_forward(self):
        self.moving = True

def update_pose(frame, sm, gait_controller, lines, trajectory_lines):
    leg_angles, foot_positions = gait_controller.update()

    coords = sm.get_leg_coordinates()

    body_points = [coord[0] for coord in coords] + [coords[0][0]]
    lines[0].set_data([p[0] for p in body_points], [p[1] for p in body_points])
    lines[0].set_3d_properties([p[2] for p in body_points])

    for i, leg in enumerate(coords):
        lines[i+1].set_data([p[0] for p in leg], [p[1] for p in leg])
        lines[i+1].set_3d_properties([p[2] for p in leg])

    for i, leg in enumerate(['RB', 'RF', 'LF', 'LB']):
        trajectory = gait_controller.leg_trajectories[leg]
        if trajectory:
            trajectory_lines[i].set_data([p[0] for p in trajectory], [p[1] for p in trajectory])
            trajectory_lines[i].set_3d_properties([p[2] for p in trajectory])

    return lines + trajectory_lines

def on_press(key, gait_controller):
    if key == keyboard.Key.up:
        gait_controller.move_forward()
        leg_angles, foot_positions = gait_controller.update()
        print("\nLeg Angles (degrees) and Foot Positions:")
        print("    Hip    Upper   Lower      X       Y       Z")
        leg_names = ['RB', 'RF', 'LF', 'LB']
        for i, leg in enumerate(leg_names):
            angles = [angle * r2d for angle in leg_angles[i]]
            initial_pos = gait_controller.initial_foot_positions[leg]
            current_pos = foot_positions[leg][:3]
            relative_pos = tuple(current - initial for current, initial in zip(current_pos, initial_pos))
            print(f"{leg}: {angles[0]:6.2f} {angles[1]:6.2f} {angles[2]:6.2f}   {relative_pos[0]:6.3f} {relative_pos[1]:6.3f} {relative_pos[2]:6.3f}")

def main():
    sm = SpotMicroStickFigure(x=0, y=0.14, z=0, phi=0, theta=0, psi=0)
    gait_controller = SimpleGaitController(sm)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    lines = [ax.plot([], [], [], 'k-', linewidth=2)[0]]
    colors = ['r', 'g', 'b', 'y']
    for color in colors:
        lines.append(ax.plot([], [], [], f'{color}-', linewidth=1)[0])

    trajectory_lines = [ax.plot([], [], [], f'{color}--', linewidth=1)[0] for color in colors]

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('SpotMicroStickFigure Gait (Press Up Arrow to Move)')

    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0, 0.4)

    listener = keyboard.Listener(
        on_press=lambda key: on_press(key, gait_controller))
    listener.start()

    anim = animation.FuncAnimation(
        fig,
        update_pose,
        frames=1000,
        fargs=(sm, gait_controller, lines, trajectory_lines),
        interval=50,
        blit=False
    )

    plt.show()

if __name__ == "__main__":
    main()