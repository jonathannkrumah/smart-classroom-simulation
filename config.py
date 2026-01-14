# config.py
SIMULATION_CONFIG = {
    "simulation_duration": 480,  # minutes (8-hour school day)
    "time_step": 1,  # minutes per simulation step
    "room_capacity": 30,  # number of students
    "room_volume": 200,  # cubic meters
    
    # Thresholds from your ML model
    "thresholds": {
        "co2_max": 1000,  # ppm
        "temp_min": 20,   # °C
        "temp_max": 26,   # °C
        "noise_max": 65,  # dB
        "light_min": 300  # lux
    },
    
    # Environmental change rates
    "co2_per_person": 0.004,  # L/min
    "heat_per_person": 100,   # Watts
}