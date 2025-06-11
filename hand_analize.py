import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import random
import argparse
import os
import pygame.midi
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt


class ImprovedHandModel(nn.Module):
    """Improved neural network model for hand opening prediction"""

    def __init__(self, input_size=42, hidden_size=256):
        super(ImprovedHandModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
            # Remove Sigmoid - use raw output with proper loss function
        )

    def forward(self, x):
        return self.network(x)


class ImprovedHandAnalyzer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Reduced for better performance
            min_tracking_confidence=0.5    # Reduced for better performance
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Model components
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # MIDI setup
        self.midi_out = None
        self.setup_midi()

        # Improved parameters
        self.DATA_COLLECTION_ROUNDS = 12  # More data points
        self.PREPARATION_TIME = 4
        self.CSV_FILE = 'improved_hand_data.csv'
        
        # Data validation
        self.min_confidence_threshold = 0.5  # Reduced for better performance
        
        # Cache for column names
        self._columns_cache = None

    def setup_midi(self):
        """Initialize MIDI"""
        try:
            pygame.midi.init()
            if pygame.midi.get_count() > 0:
                self.midi_out = pygame.midi.Output(0)
                print("MIDI initialized successfully")
            else:
                print("No MIDI devices found")
        except Exception as e:
            print(f"MIDI initialization error: {e}")

    def setup_optimized_camera(self):
        """Setup camera with optimal settings for smooth preview"""
        cap = cv2.VideoCapture(0)
        
        # Force lower resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Set frame rate
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Reduce buffer size to minimize lag
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Auto exposure (if supported)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        
        return cap

    def calculate_hand_features(self, landmarks) -> dict:
        """Calculate comprehensive hand features"""
        if len(landmarks) < 21:
            return {}

        features = {}
        
        # Key points
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]

        # Primary distance (thumb-index)
        thumb_index_dist = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        features['thumb_index_distance'] = thumb_index_dist

        # Additional finger distances for context
        features['thumb_middle_distance'] = np.sqrt((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)
        features['thumb_ring_distance'] = np.sqrt((thumb_tip.x - ring_tip.x)**2 + (thumb_tip.y - ring_tip.y)**2)
        features['thumb_pinky_distance'] = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)

        # Finger angles (more robust features)
        # Thumb angle
        thumb_vector1 = np.array([thumb_ip.x - thumb_mcp.x, thumb_ip.y - thumb_mcp.y])
        thumb_vector2 = np.array([thumb_tip.x - thumb_ip.x, thumb_tip.y - thumb_ip.y])
        thumb_angle = np.arccos(np.clip(np.dot(thumb_vector1, thumb_vector2) / 
                                       (np.linalg.norm(thumb_vector1) * np.linalg.norm(thumb_vector2)), -1, 1))
        features['thumb_angle'] = thumb_angle

        # Index finger angle
        index_vector1 = np.array([index_pip.x - index_mcp.x, index_pip.y - index_mcp.y])
        index_vector2 = np.array([index_tip.x - index_pip.x, index_tip.y - index_pip.y])
        index_angle = np.arccos(np.clip(np.dot(index_vector1, index_vector2) / 
                                       (np.linalg.norm(index_vector1) * np.linalg.norm(index_vector2)), -1, 1))
        features['index_angle'] = index_angle

        # Hand span (normalized by hand size)
        hand_size = np.sqrt((wrist.x - middle_tip.x)**2 + (wrist.y - middle_tip.y)**2)
        features['hand_size'] = hand_size
        features['normalized_thumb_index'] = thumb_index_dist / hand_size if hand_size > 0 else 0

        return features

    def landmarks_to_enhanced_array(self, landmarks) -> np.array:
        """Convert landmarks to enhanced feature array"""
        if landmarks is None:
            return np.zeros(47)  # 42 coordinates + 5 calculated features

        # Original coordinates
        coords = []
        for landmark in landmarks:
            coords.extend([landmark.x, landmark.y])

        # Add calculated features
        features = self.calculate_hand_features(landmarks)
        enhanced_features = [
            features.get('thumb_index_distance', 0),
            features.get('thumb_angle', 0),
            features.get('index_angle', 0),
            features.get('hand_size', 0),
            features.get('normalized_thumb_index', 0)
        ]

        return np.array(coords + enhanced_features)

    def improved_distance_to_percentage(self, landmarks) -> float:
        """Improved distance to percentage conversion using multiple features"""
        features = self.calculate_hand_features(landmarks)
        
        # Use normalized distance for better consistency
        normalized_dist = features.get('normalized_thumb_index', 0)
        
        # Empirically determined bounds (you may need to adjust these)
        min_normalized = 0.05  # Closed hand
        max_normalized = 0.35  # Fully open hand
        
        percentage = ((normalized_dist - min_normalized) / (max_normalized - min_normalized)) * 100
        return max(0, min(100, percentage))

    def get_columns_cache(self):
        """Get cached column names"""
        if self._columns_cache is None:
            self._columns_cache = []
            for point_id in range(21):
                self._columns_cache.extend([f'x_{point_id}', f'y_{point_id}'])
            self._columns_cache.extend(['thumb_index_dist', 'thumb_angle', 'index_angle', 'hand_size', 'normalized_dist'])
            self._columns_cache.append('target_angle')
        return self._columns_cache

    def optimized_countdown_with_preview(self, cap, target_angle, countdown_time):
        """Smooth countdown with real-time preview"""
        start_time = time.time()
        last_mediapipe_time = 0
        mediapipe_interval = 0.1  # Process MediaPipe only every 100ms
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = countdown_time - elapsed
            
            if remaining <= 0:
                break
                
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            
            # Only run MediaPipe processing occasionally to reduce lag
            if current_time - last_mediapipe_time > mediapipe_interval:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                last_mediapipe_time = current_time
                
                # Draw landmarks if found
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Always show countdown (this is fast)
            cv2.putText(frame, f"Target: {target_angle}% - Ready: {int(remaining)+1}s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Position your hands", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            # Non-blocking key check
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
                
            # Small sleep to prevent overwhelming the CPU
            time.sleep(0.01)  # 10ms = roughly 100fps max
        
        return True

    def capture_and_validate_data_fast(self, frame, target_angle) -> int:
        """FAST capture with minimal processing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        samples_saved = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            columns = self.get_columns_cache()

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Quick confidence check
                if handedness.classification[0].score < self.min_confidence_threshold:
                    continue

                # Get features (this is still the bottleneck, but necessary)
                landmarks_array = self.landmarks_to_enhanced_array(hand_landmarks.landmark)
                
                # Quick validation
                if np.any(np.isnan(landmarks_array)) or np.any(np.isinf(landmarks_array)):
                    continue

                # Fast CSV append
                data_row = list(landmarks_array) + [target_angle]
                df_new = pd.DataFrame([data_row], columns=columns)

                if os.path.exists(self.CSV_FILE):
                    df_new.to_csv(self.CSV_FILE, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(self.CSV_FILE, index=False)

                samples_saved += 1

        return samples_saved

    def collect_data_smooth(self):
        """Smooth data collection with optimized camera handling"""
        print(f"=== SMOOTH DATA COLLECTION ===")
        print(f"Collecting {self.DATA_COLLECTION_ROUNDS} samples with optimized performance")
        print("Each hand will be saved as a separate sample")
        
        cap = self.setup_optimized_camera()
        
        # Let camera adjust (important!)
        print("Camera warming up...")
        for _ in range(30):  # Skip first 30 frames
            cap.read()
        
        collected_samples = 0
        target_angles = []

        # Generate more diverse target angles
        for i in range(self.DATA_COLLECTION_ROUNDS):
            if i < 4:
                # Ensure we have corner cases
                angles = [0, 25, 75, 100]
                target_angles.append(angles[i])
            else:
                # Random angles for the rest
                target_angles.append(random.randint(0, 100))

        for round_num, target_angle in enumerate(target_angles):
            print(f"\nRound {round_num + 1}/{self.DATA_COLLECTION_ROUNDS}")
            print(f"Set both hands to {target_angle}% opening")
            
            # Smooth countdown
            if not self.optimized_countdown_with_preview(cap, target_angle, self.PREPARATION_TIME):
                break  # User pressed 'q'
            
            print("CAPTURE!")
            
            # Clear any buffered frames
            for _ in range(5):
                cap.read()
            
            # Single clean capture
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                samples = self.capture_and_validate_data_fast(frame, target_angle)
                collected_samples += samples
                print(f"✓ Saved {samples} samples for {target_angle}%")
                
                # Brief feedback
                feedback_frame = frame.copy()
                cv2.putText(feedback_frame, f"CAPTURED! Samples: {samples}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('Data Collection', feedback_frame)
                cv2.waitKey(1000)  # Show for 1 second

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nData collection complete! Total samples: {collected_samples}")
        self.analyze_collected_data()

    def analyze_collected_data(self):
        """Analyze the collected data quality"""
        if not os.path.exists(self.CSV_FILE):
            print("No data file found!")
            return

        df = pd.read_csv(self.CSV_FILE)
        print(f"\n=== DATA ANALYSIS ===")
        print(f"Total samples: {len(df)}")
        print(f"Target angle distribution:")
        print(df['target_angle'].value_counts().sort_index())
        
        # Check for data quality issues
        if df.isnull().any().any():
            print("⚠️ Warning: Found NaN values in data")
        
        # Analyze feature ranges
        print(f"\nFeature ranges:")
        print(f"Thumb-index distance: {df['thumb_index_dist'].min():.4f} - {df['thumb_index_dist'].max():.4f}")
        print(f"Normalized distance: {df['normalized_dist'].min():.4f} - {df['normalized_dist'].max():.4f}")

    def load_improved_model(self):
        """Load the improved trained model"""
        try:
            input_size = 47  # 42 coordinates + 5 features
            self.model = ImprovedHandModel(input_size=input_size).to(self.device)
            self.model.load_state_dict(torch.load('improved_hand_model_best.pth', map_location=self.device))
            self.model.eval()

            self.scaler = joblib.load('improvedd_scaler.pkl')
            print("Improved model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict_hand_opening_improved(self, hand_landmarks) -> float:
        """Improved prediction using enhanced features"""
        try:
            landmarks_array = self.landmarks_to_enhanced_array(hand_landmarks.landmark)
            landmarks_scaled = self.scaler.transform([landmarks_array])
            
            landmarks_tensor = torch.FloatTensor(landmarks_scaled).to(self.device)
            with torch.no_grad():
                prediction = self.model(landmarks_tensor).cpu().numpy()[0][0]
            
            # Clamp prediction to valid range
            return max(0, min(100, prediction))
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.0

    def inference_mode_smooth(self):
        """Smooth inference mode with optimized camera handling"""
        if not self.load_improved_model():
            print("Cannot load model. Train the model first.")
            return

        print("=== SMOOTH INFERENCE MODE ===")
        print("Enhanced model with additional features and optimized performance")
        print("Press 'q' to quit, 'c' to show confidence, 'r' to reset MIDI")
        
        cap = self.setup_optimized_camera()
        
        # Camera warm-up
        print("Warming up camera...")
        for _ in range(30):
            cap.read()
        
        show_confidence = False
        last_process_time = 0
        process_interval = 0.05  # Process every 50ms (20 FPS processing)
        
        # Store last predictions to avoid flickering
        last_left = 0
        last_right = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            current_time = time.time()
            
            # Only process MediaPipe at controlled intervals
            if current_time - last_process_time > process_interval:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                last_process_time = current_time
                
                left_prediction = 0
                right_prediction = 0

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Determine hand side
                        is_left = handedness.classification[0].label == 'Right'  # MediaPipe flips this
                        hand_label = "Left" if is_left else "Right"
                        confidence = handedness.classification[0].score

                        # Prediction
                        prediction = self.predict_hand_opening_improved(hand_landmarks)

                        if is_left:
                            left_prediction = prediction
                            y_pos = 30
                            color = (0, 255, 0)  # Green for left
                        else:
                            right_prediction = prediction
                            y_pos = 60
                            color = (255, 0, 0)  # Blue for right

                        # Display prediction
                        text = f"{hand_label}: {prediction:.1f}%"
                        if show_confidence:
                            text += f" (conf: {confidence:.2f})"
                        
                        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Update stored predictions
                if left_prediction > 0:
                    last_left = left_prediction
                if right_prediction > 0:
                    last_right = right_prediction
                    
                # Send MIDI data
                self.send_midi_data(last_left, last_right)
            
            else:
                # Use last known predictions for display when not processing
                cv2.putText(frame, f"Left: {last_left:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Right: {last_right:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Additional UI
            cv2.putText(frame, "Smooth Improved Model with Enhanced Features", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 'c' for confidence", 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"MIDI: L={last_left:.0f}% R={last_right:.0f}%", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Smooth Hand Analysis', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_confidence = not show_confidence
                print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")
            elif key == ord('r'):
                print("MIDI reset")
                last_left = 0
                last_right = 0

        cap.release()
        cv2.destroyAllWindows()

    def send_midi_data(self, left_percentage: float, right_percentage: float):
        """Send MIDI data with hand opening percentages"""
        if self.midi_out is None:
            return

        try:
            left_midi = int(np.clip(left_percentage * 127 / 100, 0, 127))
            right_midi = int(np.clip(right_percentage * 127 / 100, 0, 127))

            # Send Control Change messages
            self.midi_out.write([[[0xB0, 1, left_midi], pygame.midi.time()]])
            self.midi_out.write([[[0xB0, 2, right_midi], pygame.midi.time()]])

        except Exception as e:
            print(f"MIDI send error: {e}")

    def cleanup(self):
        """Clean up resources"""
        if self.midi_out:
            self.midi_out.close()
        pygame.midi.quit()

    # Legacy methods for backward compatibility
    def collect_data_with_validation(self):
        """Legacy method - redirects to smooth version"""
        print("Using optimized smooth data collection...")
        self.collect_data_smooth()

    def inference_mode_improved(self):
        """Legacy method - redirects to smooth version"""
        print("Using optimized smooth inference mode...")
        self.inference_mode_smooth()


def main():
    parser = argparse.ArgumentParser(description='Optimized Hand Opening Analysis')
    parser.add_argument('--mode', choices=['collect', 'inference'], 
                       default='collect', help='Operation mode')

    args = parser.parse_args()

    analyzer = ImprovedHandAnalyzer()

    try:
        if args.mode == 'collect':
            analyzer.collect_data_smooth()
        else:
            analyzer.inference_mode_smooth()
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()