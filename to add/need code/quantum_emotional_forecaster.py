import pennylane as qml
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class EmotionPrediction:
    """Container for emotion prediction results"""
    timestamps: List[int]
    predictions: List[List[float]]
    confidence: List[float]
    dominant_emotions: List[str]

class QuantumEmotionForecaster:
    def __init__(self, processor, temporal_window: int = 5, prediction_horizon: int = 3):
        self.processor = processor
        self.temporal_window = temporal_window
        self.prediction_horizon = prediction_horizon
        self.emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        
        # Quantum circuit parameters
        self.num_qubits = processor.num_qubits
        self.ancilla_qubits = self.num_qubits  # Additional qubits for LSTM gates
        
        # Build quantum circuits
        self.forecast_circuit = self._build_forecast_circuit()
        self.confidence_circuit = self._build_confidence_circuit()
        
        # Training parameters
        self.learning_rate = 0.01
        self.noise_level = 0.1
        
    def _build_forecast_circuit(self):
        """Enhanced quantum LSTM for emotion forecasting with adaptive gates"""
        total_qubits = self.num_qubits * 2
        dev = qml.device("default.qubit", wires=total_qubits)
        
        @qml.qnode(dev, interface="autograd")
        def circuit(history_states, gate_params):
            # Initialize quantum memory
            self._initialize_quantum_memory(history_states[0])
            
            # Process temporal sequence
            for t in range(1, len(history_states)):
                self._apply_input_gate(history_states[t], gate_params['input'], t)
                self._apply_forget_gate(gate_params['forget'], t)
                self._apply_output_gate(gate_params['output'], t)
                
                # Add quantum noise for robustness
                self._add_quantum_noise()
            
            # Final entanglement and measurement
            self._apply_final_entanglement()
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        
        return circuit
    
    def _build_confidence_circuit(self):
        """Circuit to estimate prediction confidence"""
        dev = qml.device("default.qubit", wires=self.num_qubits)
        
        @qml.qnode(dev)
        def circuit(prediction_state):
            # Encode prediction state
            for i, val in enumerate(prediction_state):
                qml.RY(val, wires=i)
            
            # Create entangled state for confidence measurement
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Measure coherence as confidence indicator
            return qml.expval(qml.PauliX(0))
        
        return circuit
    
    def _initialize_quantum_memory(self, initial_state):
        """Initialize quantum memory with initial emotional state"""
        for i, val in enumerate(initial_state):
            qml.RY(np.pi * val, wires=i)
    
    def _apply_input_gate(self, current_state, params, timestep):
        """Quantum input gate with parametric control"""
        for i, val in enumerate(current_state):
            # Parametric input transformation
            qml.RY(params[i] * val, wires=i)
            # Temporal modulation
            qml.RZ(0.1 * timestep * val, wires=i)
    
    def _apply_forget_gate(self, params, timestep):
        """Quantum forget gate with adaptive forgetting"""
        for i in range(self.num_qubits):
            # Controlled forgetting based on temporal distance
            forget_strength = params[i] * np.exp(-0.1 * timestep)
            qml.CRY(forget_strength, wires=[i, i + self.num_qubits])
    
    def _apply_output_gate(self, params, timestep):
        """Quantum output gate with cross-qubit interactions"""
        for i in range(self.num_qubits):
            qml.RY(params[i], wires=i)
            # Cross-emotional interactions
            if i < self.num_qubits - 1:
                qml.CNOT(wires=[i, i + 1])
    
    def _add_quantum_noise(self):
        """Add controlled quantum noise for robustness"""
        for i in range(self.num_qubits):
            qml.BitFlip(self.noise_level, wires=i)
            qml.PhaseFlip(self.noise_level * 0.5, wires=i)
    
    def _apply_final_entanglement(self):
        """Apply final entanglement pattern"""
        qml.Barrier(wires=range(self.num_qubits))
        
        # Create Bell pairs for enhanced correlation
        for i in range(0, self.num_qubits - 1, 2):
            qml.Hadamard(wires=i)
            qml.CNOT(wires=[i, i + 1])
    
    def _generate_gate_parameters(self):
        """Generate adaptive gate parameters"""
        return {
            'input': np.random.uniform(-np.pi, np.pi, self.num_qubits),
            'forget': np.random.uniform(0, 0.5, self.num_qubits),
            'output': np.random.uniform(-np.pi/2, np.pi/2, self.num_qubits)
        }
    
    def predict_emotions(self, steps: int = None) -> EmotionPrediction:
        """Predict emotional state evolution with confidence estimation"""
        if steps is None:
            steps = self.prediction_horizon
            
        # Extract recent history
        if len(self.processor.history) < self.temporal_window:
            raise ValueError(f"Need at least {self.temporal_window} historical states")
        
        history = self._extract_emotion_history()
        predictions = []
        confidences = []
        timestamps = []
        
        # Generate predictions
        current_time = len(self.processor.history)
        gate_params = self._generate_gate_parameters()
        
        for step in range(steps):
            # Predict next state
            forecast = self.forecast_circuit(history, gate_params)
            predictions.append(forecast)
            
            # Calculate confidence
            confidence = abs(self.confidence_circuit(forecast))
            confidences.append(confidence)
            
            # Update history for next prediction
            history = history[1:] + [forecast]
            timestamps.append(current_time + step + 1)
            
            # Adapt parameters based on prediction confidence
            if confidence < 0.3:
                gate_params = self._generate_gate_parameters()
        
        # Identify dominant emotions
        dominant_emotions = self._identify_dominant_emotions(predictions)
        
        return EmotionPrediction(
            timestamps=timestamps,
            predictions=predictions,
            confidence=confidences,
            dominant_emotions=dominant_emotions
        )
    
    def _extract_emotion_history(self) -> List[List[float]]:
        """Extract normalized emotion values from processor history"""
        history = []
        for state in self.processor.history[-self.temporal_window:]:
            if isinstance(state, dict) and 'output' in state:
                emotion_values = list(state['output'].values())
            else:
                emotion_values = state if isinstance(state, list) else [0.0] * self.num_qubits
            
            # Normalize to [-1, 1] range
            normalized = [2 * (v - 0.5) for v in emotion_values]
            history.append(normalized)
        
        return history
    
    def _identify_dominant_emotions(self, predictions: List[List[float]]) -> List[str]:
        """Identify dominant emotion for each prediction"""
        dominant = []
        for pred in predictions:
            max_idx = np.argmax(np.abs(pred))
            if max_idx < len(self.emotion_labels):
                dominant.append(self.emotion_labels[max_idx])
            else:
                dominant.append('neutral')
        return dominant
    
    def visualize_predictions(self, prediction_result: EmotionPrediction, 
                            save_path: Optional[str] = None):
        """Visualize emotion predictions with confidence intervals"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot emotion predictions
        predictions_array = np.array(prediction_result.predictions)
        for i, emotion in enumerate(self.emotion_labels[:predictions_array.shape[1]]):
            ax1.plot(prediction_result.timestamps, predictions_array[:, i], 
                    marker='o', label=emotion, alpha=0.7)
        
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Emotion Intensity')
        ax1.set_title('Quantum Emotion Forecasting')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot confidence levels
        ax2.plot(prediction_result.timestamps, prediction_result.confidence, 
                'r-', marker='s', linewidth=2, label='Prediction Confidence')
        ax2.fill_between(prediction_result.timestamps, 0, prediction_result.confidence, 
                        alpha=0.3, color='red')
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Prediction Confidence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_emotional_trends(self, prediction_result: EmotionPrediction) -> Dict:
        """Analyze trends in emotional predictions"""
        predictions_array = np.array(prediction_result.predictions)
        
        analysis = {
            'trend_direction': {},
            'volatility': {},
            'peak_emotions': {},
            'average_confidence': np.mean(prediction_result.confidence),
            'stability_score': 1 - np.std(prediction_result.confidence)
        }
        
        # Analyze trends for each emotion
        for i, emotion in enumerate(self.emotion_labels[:predictions_array.shape[1]]):
            emotion_values = predictions_array[:, i]
            
            # Calculate trend direction
            if len(emotion_values) > 1:
                trend = np.polyfit(range(len(emotion_values)), emotion_values, 1)[0]
                analysis['trend_direction'][emotion] = 'increasing' if trend > 0 else 'decreasing'
            
            # Calculate volatility
            analysis['volatility'][emotion] = np.std(emotion_values)
            
            # Find peak moments
            peak_idx = np.argmax(np.abs(emotion_values))
            analysis['peak_emotions'][emotion] = {
                'peak_value': emotion_values[peak_idx],
                'peak_time': prediction_result.timestamps[peak_idx]
            }
        
        return analysis
    
    def calibrate_model(self, validation_data: List[Dict], epochs: int = 100):
        """Calibrate the quantum forecaster using validation data"""
        print(f"Calibrating quantum emotion forecaster over {epochs} epochs...")
        
        best_error = float('inf')
        best_params = None
        
        for epoch in range(epochs):
            # Generate random parameters
            gate_params = self._generate_gate_parameters()
            
            # Calculate prediction error
            total_error = 0
            for data_point in validation_data:
                try:
                    # Make prediction
                    history = data_point['history']
                    actual = data_point['actual']
                    predicted = self.forecast_circuit(history, gate_params)
                    
                    # Calculate error
                    error = np.mean([(p - a)**2 for p, a in zip(predicted, actual)])
                    total_error += error
                except Exception as e:
                    continue
            
            # Update best parameters
            if total_error < best_error:
                best_error = total_error
                best_params = gate_params
                
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Best error = {best_error:.4f}")
        
        print(f"Calibration complete. Final error: {best_error:.4f}")
        return best_params

# Example usage and testing
def demo_quantum_emotion_forecaster():
    """Demonstrate the quantum emotion forecaster"""
    # Mock processor for demonstration
    class MockProcessor:
        def __init__(self):
            self.num_qubits = 7
            self.history = []
            
            # Generate synthetic emotional history
            for i in range(10):
                state = {
                    'output': {
                        'joy': np.random.beta(2, 2),
                        'sadness': np.random.beta(2, 2),
                        'anger': np.random.beta(1, 3),
                        'fear': np.random.beta(1, 3),
                        'surprise': np.random.beta(1, 2),
                        'disgust': np.random.beta(1, 4),
                        'neutral': np.random.beta(3, 2)
                    }
                }
                self.history.append(state)
    
    # Create forecaster
    processor = MockProcessor()
    forecaster = QuantumEmotionForecaster(processor, temporal_window=5, prediction_horizon=3)
    
    # Make predictions
    try:
        predictions = forecaster.predict_emotions(steps=5)
        
        print("Quantum Emotion Forecasting Results:")
        print(f"Predicted {len(predictions.predictions)} time steps")
        print(f"Average confidence: {np.mean(predictions.confidence):.3f}")
        print(f"Dominant emotions: {predictions.dominant_emotions}")
        
        # Analyze trends
        analysis = forecaster.analyze_emotional_trends(predictions)
        print(f"\nTrend Analysis:")
        print(f"Average confidence: {analysis['average_confidence']:.3f}")
        print(f"Stability score: {analysis['stability_score']:.3f}")
        
        # Visualize results
        forecaster.visualize_predictions(predictions)
        
    except Exception as e:
        print(f"Error in demonstration: {e}")

if __name__ == "__main__":
    demo_quantum_emotion_forecaster()