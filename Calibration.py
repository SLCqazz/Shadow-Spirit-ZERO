#!/usr/bin/python

import time
import math
import smbus
import sys
import tty
import termios

# ============================================================================
# Raspi PCA9685 16-Channel PWM Servo Driver
# ============================================================================

class PCA9685:

    # Registers/etc.
    __SUBADR1            = 0x02
    __SUBADR2            = 0x03
    __SUBADR3            = 0x04
    __MODE1              = 0x00
    __PRESCALE           = 0xFE
    __LED0_ON_L          = 0x06
    __LED0_ON_H          = 0x07
    __LED0_OFF_L         = 0x08
    __LED0_OFF_H         = 0x09

    def __init__(self, address=0x40, debug=False):
        self.bus = smbus.SMBus(1)
        self.address = address
        self.debug = debug
        if (self.debug):
            print("Reseting PCA9685")
        self.write(self.__MODE1, 0x00)
    
    def write(self, reg, value):
        "Writes an 8-bit value to the specified register/address"
        self.bus.write_byte_data(self.address, reg, value)
        if (self.debug):
            print("I2C: Write 0x%02X to register 0x%02X" % (value, reg))
          
    def read(self, reg):
        "Read an unsigned byte from the I2C device"
        result = self.bus.read_byte_data(self.address, reg)
        if (self.debug):
            print("I2C: Device 0x%02X returned 0x%02X from reg 0x%02X" % (self.address, result & 0xFF, reg))
        return result
    
    def setPWMFreq(self, freq):
        "Sets the PWM frequency"
        prescaleval = 25000000.0    # 25MHz
        prescaleval /= 4096.0       # 12-bit
        prescaleval /= float(freq)
        prescaleval -= 1.0
        if (self.debug):
            print("Setting PWM frequency to %d Hz" % freq)
            print("Estimated pre-scale: %d" % prescaleval)
        prescale = math.floor(prescaleval + 0.5)
        if (self.debug):
            print("Final pre-scale: %d" % prescale)

        oldmode = self.read(self.__MODE1);
        newmode = (oldmode & 0x7F) | 0x10        # sleep
        self.write(self.__MODE1, newmode)        # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        "Sets a single PWM channel"
        self.write(self.__LED0_ON_L+4*channel, on & 0xFF)
        self.write(self.__LED0_ON_H+4*channel, on >> 8)
        self.write(self.__LED0_OFF_L+4*channel, off & 0xFF)
        self.write(self.__LED0_OFF_H+4*channel, off >> 8)
        if (self.debug):
            print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel,on,off))
      
    def setServoPulse(self, channel, pulse):
        "Sets the Servo Pulse,The PWM frequency must be 50HZ"
        pulse = pulse * 4096 / 20000        # PWM frequency is 50HZ, the period is 20000us
        self.setPWM(channel, 0, int(pulse))

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

def print_positions(pwm):
    """Prints the current positions of all servos in terms of duty cycle."""
    for channel in range(16):
        on = pwm.read(pwm.__LED0_ON_L + 4 * channel) + (pwm.read(pwm.__LED0_ON_H + 4 * channel) << 8)
        off = pwm.read(pwm.__LED0_OFF_L + 4 * channel) + (pwm.read(pwm.__LED0_OFF_H + 4 * channel) << 8)
        duty_cycle = (off - on) * 20000 / 4096
        print(f"Channel {channel}: Duty Cycle = {duty_cycle:.2f} us")

if __name__ == '__main__':
    pwm = PCA9685(0x40, debug=False)
    pwm.setPWMFreq(50)

    # Initial positions for each servo
    initial_positions = {
        3: 1700,  # Servo 1
        4: 1700,  # Servo 2
        5: 1700,  # Servo 3
        7: 1000,   # Servo 4
        8: 1200,   # Servo 5
        10: 1650, # Servo 6
        11: 1200,  # Servo 7
        12: 1650, # Servo 8
        2: 1000,  # Servo 9
        6: 1950,  # Servo 10
        9: 2050,  # Servo 11
        13: 1000  # Servo 12
    }

    # Set initial positions
    for channel, position in initial_positions.items():
        pwm.setServoPulse(channel, position)

    try:
        while True:
            print("Select a servo (1-12) or press 'q' to quit:")
            selected_servo = input().strip()
            if selected_servo.lower() == 'q':
                break
            if not selected_servo.isdigit() or not 1 <= int(selected_servo) <= 12:
                print("Invalid servo number. Please enter a number between 1 and 12.")
                continue
            selected_servo = int(selected_servo)

            # Map servo number to PCA9685 channel
            channel_map = {
                1: 3, #LB lower
                2: 4, #LB upper
                3: 5, #LF lower
                4: 7, #LF upper
                5: 8, #RF lower
                6: 10, #RF upper
                7: 11, #RB lower
                8: 12, #RB upper
                9: 2, #LB hip
                10: 6, #LF hip
                11: 9, #RF hip
                12: 13 #RB hip
            }
            selected_channel = channel_map[selected_servo]

            print(f"Controlling Servo {selected_servo} (Channel {selected_channel}). Press 'q' to stop.")
            while True:
                key = get_key()
                if key == '\x1b[A':  # Up arrow
                    initial_positions[selected_channel] += 50
                    if initial_positions[selected_channel] > 2600:
                        initial_positions[selected_channel] = 2600
                    pwm.setServoPulse(selected_channel, initial_positions[selected_channel])
                    print(f"Servo {selected_servo} (Channel {selected_channel}) position: {initial_positions[selected_channel]} us")
                elif key == '\x1b[B':  # Down arrow
                    initial_positions[selected_channel] -= 50
                    if initial_positions[selected_channel] < 400:
                        initial_positions[selected_channel] = 400
                    pwm.setServoPulse(selected_channel, initial_positions[selected_channel])
                    print(f"Servo {selected_servo} (Channel {selected_channel}) position: {initial_positions[selected_channel]} us")
                elif key == 'q':
                    break

    except KeyboardInterrupt:
        print("Program interrupted by user.")

    finally:
        print("Current positions of all servos in terms of duty cycle:")
        print_positions(pwm)
        # Stop sending PWM signals
        for channel in range(16):
            pwm.setPWM(channel, 0, 0)
        print("PWM signals stopped and program terminated.")
