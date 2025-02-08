import numpy as np
from math import pi, degrees
from spot_micro_kinematics.spot_micro_stick_figure import SpotMicroStickFigure
from spot_micro_kinematics.utilities import transformations
from pca9685 import PCA9685
import sys
import tty
import termios

# Constants
d2r = pi / 180
r2d = 180 / pi

# Create SpotMicroStickFigure instance
sm = SpotMicroStickFigure(y=0.18)

# Set initial neutral stance
l = sm.body_length
w = sm.body_width
l1 = sm.hip_length
desired_p4_points = np.array([
    [-l/2, 0, w/2 + l1],
    [l/2, 0, w/2 + l1],
    [l/2, 0, -w/2 - l1],
    [-l/2, 0, -w/2 - l1]
])
sm.set_absolute_foot_coordinates(desired_p4_points)

# Updated servo mappings
servo_mappings = {
    'LB': {'hip': (1000, -1), 'upper': (2200, -1), 'lower': (1000, -1)},
    'RB': {'hip': (1000, 1), 'upper': (1100, -1), 'lower': (2100, -1)},
    'LF': {'hip': (1950, 1), 'upper': (1500, -1), 'lower': (950, -1)},
    'RF': {'hip': (2100, -1), 'upper': (1350, -1), 'lower': (1900, -1)}
}

# Servo channel mappings
channels = {
    'LB': {'hip': 2, 'upper': 4, 'lower': 3},
    'RB': {'hip': 13, 'upper': 12, 'lower': 11},
    'LF': {'hip': 6, 'upper': 7, 'lower': 5},
    'RF': {'hip': 9, 'upper': 10, 'lower': 8}
}

pwm_per_degree = 1000.0 / 90.0

# Initialize PCA9685
pwm = PCA9685(address=0x40, debug=False)
pwm.setPWMFreq(330)  # Set frequency to 330Hz

def angle_to_pwm(leg_name, joint_name, angle):
    center_pwm, direction = servo_mappings[leg_name][joint_name]
    angle_deg = degrees(angle)
    pwm = center_pwm + direction * angle_deg * pwm_per_degree
    return int(pwm)

def update_pose_and_servos():
    # Update body position and orientation
    ht_body = np.matmul(transformations.homog_transxyz(x, y, z), 
                        transformations.homog_rotxyz(roll, pitch, yaw))
    
    try:
        sm.set_absolute_body_pose(ht_body)

        # Get and print leg angles and PWM values
        leg_angles = sm.get_leg_angles()
        print("\nLeg Angles (degrees) and PWM values:")
        leg_names = ['RB', 'RF', 'LF', 'LB']
        joint_names = ['hip', 'upper', 'lower']
        for i, angles in enumerate(leg_angles):
            print(f"{leg_names[i]}:")
            for j, angle in enumerate(angles):
                pwm_value = angle_to_pwm(leg_names[i], joint_names[j], angle)
                channel = channels[leg_names[i]][joint_names[j]]
                pwm.setServoPulse(channel, pwm_value)
                print(f"  {joint_names[j]}: {round(degrees(angle), 2)}° - PWM: {pwm_value} - Channel: {channel}")
    
    except ValueError as e:
        print(f"Invalid pose: {e}")

def getch():
    """Gets a single character from standard input. Does not echo to the screen."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def get_key():
    """Gets a key press and returns the key code."""
    ch1 = getch()
    if ch1 == '\x1b':
        ch2 = getch()
        ch3 = getch()
        return ch1 + ch2 + ch3
    return ch1

# Initial pose
x, y, z = 0, 0.18, 0
roll, pitch, yaw = 0, 0, 0
step = 0.005
angle_step = 1 * pi / 180  # 1 degree

print("Commands:")
print("w/s: Move in y direction")
print("a/d: Move in x direction")
print("q/e: Move in z direction")
print("j/l: Roll")
print("i/k: Pitch")
print("u/o: Yaw")
print("x: Quit")

while True:
    key = get_key()
    
    if key == 'x':
        break
    elif key == 'w': y += step
    elif key == 's': y -= step
    elif key == 'a': x -= step
    elif key == 'd': x += step
    elif key == 'q': z += step
    elif key == 'e': z -= step
    elif key == 'j': roll -= angle_step
    elif key == 'l': roll += angle_step
    elif key == 'i': pitch += angle_step
    elif key == 'k': pitch -= angle_step
    elif key == 'u': yaw -= angle_step
    elif key == 'o': yaw += angle_step
    else:
        continue
    
    update_pose_and_servos()

# Clean up
pwm.write(pwm.__MODE1, 0x00)
print("Program ended. PCA9685 reset.")