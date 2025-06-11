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
            min_detection_confidence=0.8,  # Increased for better detection
            min_tracking_confidence=0.7    # Increased for better tracking
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
        self.min_confidence_threshold = 0.7

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

    def collect_data_with_validation(self):
        """Enhanced data collection with validation"""
        print(f"=== IMPROVED DATA COLLECTION ===")
        print(f"Collecting {self.DATA_COLLECTION_ROUNDS} samples with validation")
        print("Each hand will be saved as a separate sample")

        cap = cv2.VideoCapture(0)
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
            print(f"Preparation time: {self.PREPARATION_TIME} seconds...")

            # Countdown with live preview
            for i in range(self.PREPARATION_TIME, 0, -1):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)
                    
                    # Show live preview with hand tracking
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    cv2.putText(frame, f"Target: {target_angle}% - Get ready: {i}s", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(1)
                
                time.sleep(1)

            print("CAPTURE!")

            # Multiple captures for better data
            samples_this_round = 0
            for capture_attempt in range(3):  # Take 3 quick captures
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    samples = self.capture_and_validate_data(frame, target_angle)
                    samples_this_round += samples
                    time.sleep(0.1)  # Brief pause between captures

            collected_samples += samples_this_round
            print(f"✓ Saved {samples_this_round} samples for {target_angle}%")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nData collection complete! Total samples: {collected_samples}")
        self.analyze_collected_data()

    def capture_and_validate_data(self, frame, target_angle) -> int:
        """Capture and validate data before saving"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        samples_saved = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            columns = []
            for point_id in range(21):
                columns.extend([f'x_{point_id}', f'y_{point_id}'])
            # Add feature columns
            columns.extend(['thumb_index_dist', 'thumb_angle', 'index_angle', 'hand_size', 'normalized_dist'])
            columns.append('target_angle')

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Validate hand detection confidence
                confidence = handedness.classification[0].score
                if confidence < self.min_confidence_threshold:
                    continue

                # Convert to enhanced feature array
                landmarks_array = self.landmarks_to_enhanced_array(hand_landmarks.landmark)
                
                # Data validation - check for reasonable values
                if np.any(np.isnan(landmarks_array)) or np.any(np.isinf(landmarks_array)):
                    continue

                # Prepare data row
                data_row = list(landmarks_array) + [target_angle]

                # Save to CSV
                df_new = pd.DataFrame([data_row], columns=columns)

                if os.path.exists(self.CSV_FILE):
                    df_new.to_csv(self.CSV_FILE, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(self.CSV_FILE, index=False)

                samples_saved += 1

        return samples_saved

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
            self.model.load_state_dict(torch.load('improved_hand_model.pth', map_location=self.device))
            self.model.eval()

            self.scaler = joblib.load('improved_scaler.pkl')
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

    def inference_mode_improved(self):
        """Improved inference mode with better visualization"""
        if not self.load_improved_model():
            print("Cannot load model. Train the model first.")
            return

        print("=== IMPROVED INFERENCE MODE ===")
        print("Enhanced model with additional features")
        print("Press 'q' to quit, 'c' to show confidence, 'r' to reset MIDI")

        cap = cv2.VideoCapture(0)
        show_confidence = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

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

                # Send MIDI data
                self.send_midi_data(left_prediction, right_prediction)

            # Additional UI
            cv2.putText(frame, "Improved Model with Enhanced Features", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit, 'c' for confidence", 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"MIDI: L={left_prediction:.0f}% R={right_prediction:.0f}%", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Improved Hand Analysis', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_confidence = not show_confidence
                print(f"Confidence display: {'ON' if show_confidence else 'OFF'}")
            elif key == ord('r'):
                print("MIDI reset")

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


def main():
    parser = argparse.ArgumentParser(description='Improved Hand Opening Analysis')
    parser.add_argument('--mode', choices=['collect', 'inference'], 
                       default='collect', help='Operation mode')

    args = parser.parse_args()

    analyzer = ImprovedHandAnalyzer()

    try:
        if args.mode == 'collect':
            analyzer.collect_data_with_validation()
        else:
            analyzer.inference_mode_improved()
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()