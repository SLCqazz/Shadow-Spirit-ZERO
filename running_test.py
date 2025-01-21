import time
import numpy as np
from math import pi, sin, cos, sqrt, atan2, acos

d2r = pi / 180
r2d = 180 / pi

class SpotMicroStickFigure:
    def __init__(self, x=0, y=0.14, z=0, phi=0, theta=0, psi=0):
        self.x = x
        self.y = y
        self.z = z
        self.phi = phi
        self.theta = theta
        self.psi = psi
        self.body_length = 0.2
        self.body_width = 0.1
        self.hip_length = 0.04
        self.upper_leg_length = 0.1
        self.lower_leg_length = 0.1
        self.leg_angles = [[0, 0, 0] for _ in range(4)]

    def set_leg_angles(self, leg_angles):
        self.leg_angles = leg_angles

    def get_leg_angles(self):
        return self.leg_angles

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
        self.command = None
        self.start_time = None

    def get_initial_angles(self):
        return [
            [0, -30 * d2r, 60 * d2r],  # Right back
            [0, -30 * d2r, 60 * d2r],  # Right front
            [0, 30 * d2r, -60 * d2r],  # Left front
            [0, 30 * d2r, -60 * d2r]   # Left back
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

        return x, y, z

    def inverse_kinematics(self, x, y, z, leg_index):
        l1 = self.robot.hip_length
        l2 = self.robot.upper_leg_length
        l3 = self.robot.lower_leg_length

        if leg_index in [0, 1]:  # Right legs
            y -= l1
        else:  # Left legs
            y += l1

        hip_angle = atan2(y, abs(x))
        d = sqrt(x**2 + y**2)
        r = sqrt(d**2 + z**2)

        if r > l2 + l3:
            print(f"Warning: Position out of reach for leg {leg_index}")
            return self.initial_angles[leg_index]

        cos_ankle = (l2**2 + l3**2 - r**2) / (2 * l2 * l3)
        cos_ankle = max(min(cos_ankle, 1), -1)
        ankle_angle = acos(cos_ankle)

        a1 = atan2(z, d)
        a2 = acos((l2**2 + r**2 - l3**2) / (2 * l2 * r))
        knee_angle = a1 + a2

        if leg_index in [0, 1]:  # Right legs
            hip_angle = -hip_angle
            knee_angle = pi / 2 - knee_angle
            ankle_angle = ankle_angle - pi
        else:  # Left legs
            knee_angle = knee_angle - pi / 2
            ankle_angle = pi - ankle_angle

        hip_angle = max(min(hip_angle, pi / 4), -pi / 4)
        knee_angle = max(min(knee_angle, pi / 2), -pi / 2)
        ankle_angle = max(min(ankle_angle, pi), -pi)

        return hip_angle, knee_angle, ankle_angle

    def update(self, dt):
        if self.command == 'forward':
            self.v_x = 0.1
            self.moving = True
        elif self.command == 'stop':
            self.v_x = 0
            self.omega = 0
            self.moving = False

        if self.moving:
            self.phase += self.freq * dt
            if self.phase >= 1:
                self.phase -= 1

            rf = self.leg_trajectory(self.phase, self.robot.body_length / 2, self.robot.body_width / 2)
            lf = self.leg_trajectory((self.phase + 0.5) % 1, self.robot.body_length / 2, -self.robot.body_width / 2)
            rb = self.leg_trajectory((self.phase + 0.5) % 1, -self.robot.body_length / 2, self.robot.body_width / 2)
            lb = self.leg_trajectory(self.phase, -self.robot.body_length / 2, -self.robot.body_width / 2)

            motion_adjust = self.v_x * dt
            rf = (rf[0] + motion_adjust, rf[1], rf[2])
            lf = (lf[0] + motion_adjust, lf[1], lf[2])
            rb = (rb[0] + motion_adjust, rb[1], rb[2])
            lb = (lb[0] + motion_adjust, lb[1], lb[2])

            rb_angles = self.inverse_kinematics(*rb, 0)
            rf_angles = self.inverse_kinematics(*rf, 1)
            lf_angles = self.inverse_kinematics(*lf, 2)
            lb_angles = self.inverse_kinematics(*lb, 3)

            self.robot.set_leg_angles([rb_angles, rf_angles, lf_angles, lb_angles])

            self.robot.x += self.v_x * dt
            self.robot.y += self.v_y * dt
            self.robot.psi += self.omega * dt
        else:
            self.robot.set_leg_angles(self.initial_angles)

def print_angle_matrix(leg_angles):
    print("\nCurrent Angle Matrix (degrees):")
    print("    Hip    Upper   Lower")
    legs = ['RB', 'RF', 'LF', 'LB']
    for i, leg in enumerate(legs):
        angles = [angle * r2d for angle in leg_angles[i]]
        print(f"{leg}: {angles[0]:6.2f} {angles[1]:6.2f} {angles[2]:6.2f}")

def main():
    sm = SpotMicroStickFigure(x=0, y=0.14, z=0, phi=0, theta=0, psi=0)
    gait_controller = BezierGaitController(sm)

    try:
        gait_controller.command = 'forward'
        gait_controller.start_time = time.time()

        for _ in range(100):  # Simulate 5 seconds (100 * 0.05s)
            current_time = time.time()
            elapsed_time = current_time - gait_controller.start_time

            if elapsed_time >= 5:
                gait_controller.command = 'stop'

            gait_controller.update(0.05)  # 50ms update interval

            leg_angles = sm.get_leg_angles()
            print_angle_matrix(leg_angles)
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Simulation stopped")

if __name__ == "__main__":
    main()