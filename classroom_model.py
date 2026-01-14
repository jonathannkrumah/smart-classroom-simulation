# classroom_model.py
import numpy as np

class ClassroomEnvironment:
    def __init__(self, config):
        self.config = config
        
        # Initial conditions
        self.co2 = 400  # ppm (fresh air)
        self.temperature = 22  # °C
        self.humidity = 50  # %
        self.noise = 40  # dB (quiet)
        self.light = 500  # lux
        
        # Occupancy (0 = empty, 1 = full)
        self.occupancy = 0
        self.student_count = 0
        
    def update(self, time_step, student_count, fan_on=False, ac_on=False):
        """Update all environmental parameters for one time step"""
        self.student_count = student_count
        self.occupancy = student_count / self.config["room_capacity"]
        
        # CO₂ accumulation (simplified model)
        co2_production = student_count * self.config["co2_per_person"] * time_step
        air_change_rate = 0.5 if fan_on else 0.1  # ACH (air changes per hour)
        self.co2 += co2_production
        self.co2 -= air_change_rate * (self.co2 - 400) * time_step / 60
        
        # Temperature change
        heat_gain = student_count * self.config["heat_per_person"] * time_step / 3600
        if ac_on:
            heat_gain -= 2000 * time_step / 3600  # AC cooling
        
        self.temperature += heat_gain / (self.config["room_volume"] * 1.2)
        
        # Add some randomness to simulate real conditions
        self.co2 += np.random.normal(0, 5)
        self.temperature += np.random.normal(0, 0.1)
        self.noise = 40 + (student_count * 0.8) + np.random.normal(0, 2)
        
        return {
            "co2": max(400, self.co2),
            "temperature": self.temperature,
            "humidity": self.humidity,
            "noise": max(30, self.noise),
            "light": self.light,
            "occupancy": self.occupancy
        }