# pca9685.py

import time
import math
import smbus

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
            print("Resetting PCA9685")
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

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10        # sleep
        self.write(self.__MODE1, newmode)        # go to sleep
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel, on, off):
        "Sets a single PWM channel"
        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)
        if (self.debug):
            print("channel: %d  LED_ON: %d LED_OFF: %d" % (channel, on, off))
      
    def setServoPulse(self, channel, pulse):
        "Sets the Servo Pulse, The PWM frequency must be 330HZ"
        pulse = pulse * 4096 / 3030.3  # PWM frequency is 330HZ, the period is 3030.3us
        self.setPWM(channel, 0, int(pulse))

    def rampServo(self, channel, target_pulse, steps, delay):
        "Ramps the servo to the target pulse over a number of steps with a delay between each step"
        # Read the current pulse width
        current_pulse = self.readPWM(channel)
        pulse_range = target_pulse - current_pulse
        step_size = pulse_range / float(steps)
        for i in range(steps + 1):
            current_pulse = current_pulse + (step_size * i)
            self.setServoPulse(channel, current_pulse)
            time.sleep(delay)

    def readPWM(self, channel):
        "Reads the current PWM value for a channel"
        on_l = self.read(self.__LED0_ON_L + 4 * channel)
        on_h = self.read(self.__LED0_ON_H + 4 * channel)
        off_l = self.read(self.__LED0_OFF_L + 4 * channel)
        off_h = self.read(self.__LED0_OFF_H + 4 * channel)
        on = (on_h << 8) | on_l
        off = (off_h << 8) | off_l
        return off