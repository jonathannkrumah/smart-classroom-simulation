# simulation.py
import simpy
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from config import SIMULATION_CONFIG
from classroom_model import ClassroomEnvironment
from ml_model import LearningEnvironmentClassifier

class SmartClassroomSimulation:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.classroom = ClassroomEnvironment(config)
        self.ml_classifier = LearningEnvironmentClassifier()
        
        # Load or train ML model
        try:
            self.ml_classifier.load_model("trained_model.pkl")
        except:
            print("No pre-trained model found. Please train a model first.")
        
        # Simulation state
        self.interventions = []
        self.log = []
        self.fan_on = False
        self.ac_on = False
        self.lights_on = True
        
        # Start processes
        self.env.process(self.school_day_schedule())
        self.env.process(self.ml_monitoring())
        self.env.process(self.data_logging())
    
    def school_day_schedule(self):
        """Simulate a typical school day schedule"""
        # Morning: empty classroom
        yield self.env.timeout(60)  # 8:00-9:00 AM
        
        # First class: 30 students
        for _ in range(90):  # 90 minutes
            self.update_environment(30)
            yield self.env.timeout(1)
        
        # Break: 15 minutes
        for _ in range(15):
            self.update_environment(5)  # few students remain
            yield self.env.timeout(1)
        
        # Second class: 25 students
        for _ in range(90):
            self.update_environment(25)
            yield self.env.timeout(1)
        
        # And so on...
    
    def update_environment(self, student_count):
        """Update classroom environment"""
        env_data = self.classroom.update(
            time_step=1,
            student_count=student_count,
            fan_on=self.fan_on,
            ac_on=self.ac_on
        )
        return env_data
    
    def ml_monitoring(self):
        """Monitor environment using ML model"""
        while True:
            # Get current environment
            env_data = self.classroom.update(
                time_step=1,
                student_count=self.classroom.student_count,
                fan_on=self.fan_on,
                ac_on=self.ac_on
            )
            
            # Predict if environment is conducive
            prediction = self.ml_classifier.predict(env_data)
            
            # Trigger interventions if not conducive
            if not prediction["conducive"]:
                self.trigger_interventions(env_data, prediction)
            
            yield self.env.timeout(5)  # Check every 5 minutes
    
    def trigger_interventions(self, env_data, prediction):
        """Trigger appropriate interventions"""
        intervention = {
            "time": self.env.now,
            "co2": env_data["co2"],
            "temperature": env_data["temperature"],
            "action": None
        }
        
        # Check each parameter and trigger appropriate response
        if env_data["co2"] > self.config["thresholds"]["co2_max"]:
            intervention["action"] = "activate_ventilation"
            self.fan_on = True
            print(f"[{self.env.now}min] CO₂ high ({env_data['co2']}ppm) - Fan ON")
        
        elif env_data["temperature"] > self.config["thresholds"]["temp_max"]:
            intervention["action"] = "activate_ac"
            self.ac_on = True
            print(f"[{self.env.now}min] Temp high ({env_data['temperature']}°C) - AC ON")
        
        elif env_data["noise"] > self.config["thresholds"]["noise_max"]:
            intervention["action"] = "send_alert"
            print(f"[{self.env.now}min] Noise high ({env_data['noise']}dB) - Alert sent")
        
        self.interventions.append(intervention)
    
    def data_logging(self):
        """Log all simulation data"""
        while True:
            env_data = self.classroom.update(
                time_step=1,
                student_count=self.classroom.student_count,
                fan_on=self.fan_on,
                ac_on=self.ac_on
            )
            
            log_entry = {
                "timestamp": self.env.now,
                "student_count": self.classroom.student_count,
                **env_data,
                "fan_on": self.fan_on,
                "ac_on": self.ac_on
            }
            
            self.log.append(log_entry)
            yield self.env.timeout(1)  # Log every minute
    
    def run(self):
        """Run the simulation"""
        print("Starting simulation...")
        self.env.run(until=self.config["simulation_duration"])
        print(f"Simulation complete. Logged {len(self.log)} entries.")
        
        # Save results
        self.save_results()
        self.visualize_results()
    
    def save_results(self):
        """Save simulation logs to CSV"""
        df = pd.DataFrame(self.log)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/simulation_logs/simulation_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def visualize_results(self):
        """Create visualization plots"""
        df = pd.DataFrame(self.log)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        
        # CO₂ over time
        axes[0, 0].plot(df['timestamp'], df['co2'])
        axes[0, 0].axhline(y=self.config["thresholds"]["co2_max"], color='r', linestyle='--', label='CO₂ Threshold')
        axes[0, 0].set_title('CO₂ Levels Over Time')
        axes[0, 0].set_ylabel('CO₂ (ppm)')
        axes[0, 0].legend()
        
        # Temperature over time
        axes[0, 1].plot(df['timestamp'], df['temperature'])
        axes[0, 1].axhline(y=self.config["thresholds"]["temp_max"], color='r', linestyle='--', label='Max Temp')
        axes[0, 1].axhline(y=self.config["thresholds"]["temp_min"], color='b', linestyle='--', label='Min Temp')
        axes[0, 1].set_title('Temperature Over Time')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].legend()
        
        # Noise over time
        axes[1, 0].plot(df['timestamp'], df['noise'])
        axes[1, 0].axhline(y=self.config["thresholds"]["noise_max"], color='r', linestyle='--', label='Noise Threshold')
        axes[1, 0].set_title('Noise Levels Over Time')
        axes[1, 0].set_ylabel('Noise (dB)')
        axes[1, 0].legend()
        
        # Student count
        axes[1, 1].plot(df['timestamp'], df['student_count'])
        axes[1, 1].set_title('Student Count Over Time')
        axes[1, 1].set_ylabel('Number of Students')
        
        # Interventions
        intervention_times = [i["time"] for i in self.interventions]
        if intervention_times:
            axes[2, 0].vlines(intervention_times, 0, 1, colors='r', label='Interventions')
            axes[2, 0].set_title('Intervention Events')
            axes[2, 0].set_xlabel('Time (minutes)')
            axes[2, 0].set_yticks([])
            axes[2, 0].legend()
        
        # System state
        axes[2, 1].plot(df['timestamp'], df['fan_on'].astype(int), label='Fan')
        axes[2, 1].plot(df['timestamp'], df['ac_on'].astype(int), label='AC')
        axes[2, 1].set_title('System State Over Time')
        axes[2, 1].set_xlabel('Time (minutes)')
        axes[2, 1].set_ylabel('ON/OFF')
        axes[2, 1].legend()
        
        plt.tight_layout()
        plt.savefig('results/simulation_results.png', dpi=300)
        plt.show()

def main():
    """Main function to run the simulation"""
    # Create simulation environment
    env = simpy.Environment()
    
    # Create and run simulation
    simulation = SmartClassroomSimulation(env, SIMULATION_CONFIG)
    simulation.run()

if __name__ == "__main__":
    main()