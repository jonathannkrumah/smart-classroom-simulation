# ml_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

class LearningEnvironmentClassifier:
    def __init__(self):
        self.model = None
        self.thresholds = {}
        
    def train_from_csv(self, csv_path):
        """Train model from historical data"""
        # Load your dataset
        df = pd.read_csv(csv_path)
        
        # Assuming your CSV has columns: co2, temp, noise, light, focus_label
        X = df[['co2', 'temperature', 'noise', 'light']]
        y = df['focus_label']  # 1 = conducive, 0 = non-conducive
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        # Extract thresholds (simplified method)
        self._extract_thresholds(X)
        
        return self.model
    
    def _extract_thresholds(self, X):
        """Extract decision thresholds from the trained model"""
        # Simplified: Use percentiles from conducive conditions
        # In practice, you'd analyze the decision tree structure
        self.thresholds = {
            'co2': X['co2'].quantile(0.8),  # 80th percentile
            'temperature_min': X['temperature'].quantile(0.1),
            'temperature_max': X['temperature'].quantile(0.9),
            'noise': X['noise'].quantile(0.8),
            'light': X['light'].quantile(0.2),
        }
        
        print("Extracted thresholds:", self.thresholds)
    
    def predict(self, environment_data):
        """Predict if environment is conducive for learning"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_from_csv first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([environment_data])
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        return {
            "conducive": bool(prediction),
            "confidence": float(max(probability)),
            "thresholds": self.thresholds
        }
    
    def save_model(self, filename="trained_model.pkl"):
        """Save the trained model to disk"""
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename="trained_model.pkl"):
        """Load a pre-trained model"""
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filename}")