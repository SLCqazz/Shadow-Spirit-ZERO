import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from spot_micro_kinematics.spot_micro_stick_figure import SpotMicroStickFigure
from math import pi, sin, cos, sqrt, atan2, acos
import time
from pca9685 import PCA9685  # Import PCA9685 from the separate module

# Conversion constants
d2r = pi / 180
r2d = 180 / pi

# Bezier Gait Controller Class
class BezierGaitController:
    def __init__(self, robot):
        self.robot = robot
        self.stance_height = -0.14
        self.step_length = 0.08
        self.step_height = 0.04
        self.phase = 0
        self.freq = 1.0
        self.v_x = -0.1  # Set constant forward velocity
        self.v_y = 0
        self.omega = 0
        self.moving = True
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

        # Initialize servos to initial angles
        self.send_angles_to_servo(self.initial_angles)

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

    def send_angles_to_servo(self, leg_angles):
        """
        Converts joint angles to PWM pulses and sends them to the servos.
        
        Parameters:
        - leg_angles: List of 4 legs, each containing 3 joint angles [hip, upper, lower] in radians
        """
        leg_names = ['RB', 'RF', 'LF', 'LB']
        joint_names = ['hip', 'upper', 'lower']

        for leg, angles in zip(leg_names, leg_angles):
            angles_deg = [angle * r2d for angle in angles]  # Convert radians to degrees
            for joint, angle in zip(joint_names, angles_deg):
                base_pulse, direction = self.servo_mappings[leg][joint]
                pwm = base_pulse + direction * angle * self.pwm_per_degree
                pwm = int(min(max(pwm, 500), 2500))  # Clamp pulse to 500-2500µs
                channel = self.channels[leg][joint]
                self.pwm.setServoPulse(channel, pwm)
                if self.pwm.debug:
                    print(f"Leg: {leg} Joint: {joint} Angle: {angle:.2f}° -> PWM: {pwm}µs on Channel: {channel}")

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

            # Send angles to servos
            self.send_angles_to_servo([rb_angles, rf_angles, lf_angles, lb_angles])

            # Update robot position and orientation
            self.robot.x += self.v_x * dt
            self.robot.y += self.v_y * dt
            self.robot.psi += self.omega * dt
        else:
            self.robot.set_leg_angles(self.initial_angles)
            self.send_angles_to_servo(self.initial_angles)

# Update Pose Function for Animation
def update_pose(frame, sm, gait_controller, lines, angle_lines, angle_data, ax, angle_ax):
    dt = 0.02  # 20ms per frame (50Hz)

    gait_controller.update(dt)

    # Update robot's pose
    sm.set_absolute_body_pose(sm.ht_body)

    # Get updated leg coordinates
    coords = sm.get_leg_coordinates()

    # Update body lines
    body_points = [coord[0] for coord in coords] + [coords[0][0]]
    lines[0].set_data([p[0] for p in body_points], [p[1] for p in body_points])
    lines[0].set_3d_properties([p[2] for p in body_points])

    # Update leg lines
    for i, leg in enumerate(coords):
        lines[i+1].set_data([p[0] for p in leg], [p[1] for p in leg])
        lines[i+1].set_3d_properties([p[2] for p in leg])

    # Update plot limits
    ax.set_xlim(sm.x - 0.3, sm.x + 0.3)
    ax.set_ylim(sm.y - 0.2, sm.y + 0.2)
    ax.set_zlim(0, 0.4)

    # Get and store leg angles
    leg_angles = sm.get_leg_angles()
    for i, leg in enumerate(leg_angles):
        for j, angle in enumerate(leg):
            angle_data[i, j] = angle * r2d

    # Print angle tuples
    print(f"\nTime: {frame*dt:.2f}s")
    print("Leg Angles (degrees):")
    print("    Hip    Upper   Lower")
    for i, leg in enumerate(['RB', 'RF', 'LF', 'LB']):
        print(f"{leg}: {angle_data[i,0]:6.2f} {angle_data[i,1]:6.2f} {angle_data[i,2]:6.2f}")

    # Update angle plot
    time_history = np.arange(frame+1)
    for i in range(12):
        # Each leg has 3 joints, so we can plot them sequentially
        leg = i // 3
        joint = i % 3
        angle_lines[i].set_data(time_history, angle_data.flatten()[leg, joint] * np.ones_like(time_history))
    
    angle_ax.relim()
    angle_ax.autoscale_view()

    return lines + angle_lines

def main():
    # Instantiate SpotMicroStickFigure
    sm = SpotMicroStickFigure(x=0, y=0.1, z=0, phi=0, theta=0, psi=0)

    # Create gait controller
    gait_controller = BezierGaitController(sm)

    # Set up the figures and axes
    fig = plt.figure(figsize=(10, 12))

    # 3D Plot for Robot Gait
    ax = fig.add_subplot(211, projection='3d')

    # 2D Plot for Joint Angles
    angle_ax = fig.add_subplot(212)

    # Initialize lines for 3D plot
    lines = [ax.plot([], [], [], 'k-')[0]]  # Body line
    colors = ['r', 'g', 'b', 'y']
    for color in colors:
        lines.append(ax.plot([], [], [], f'{color}-')[0])  # Leg lines

    # Initialize lines for angle plot
    angle_lines = []
    leg_names = ['RB_Hip', 'RB_Upper', 'RB_Lower',
                 'RF_Hip', 'RF_Upper', 'RF_Lower',
                 'LF_Hip', 'LF_Upper', 'LF_Lower',
                 'LB_Hip', 'LB_Upper', 'LB_Lower']
    for name in leg_names:
        line, = angle_ax.plot([], [], label=name)
        angle_lines.append(line)

    # Set labels and title for 3D plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SpotMicroStickFigure Bezier Gait')

    # Set labels and title for angle plot
    angle_ax.set_xlabel('Frame')
    angle_ax.set_ylabel('Angle (degrees)')
    angle_ax.set_title('Leg Joint Angles')
    angle_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Set initial axis limits
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(0, 0.4)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Global variables for angle data
    angle_data = np.zeros((4, 3))  # 4 legs, 3 angles each

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        update_pose,
        frames=250,
        fargs=(sm, gait_controller, lines, angle_lines, angle_data, ax, angle_ax),
        interval=20,
        blit=False
    )

    plt.show()

    print("Animation complete. If you can see this, the script has run to completion.")

if __name__ == "__main__":
    main()