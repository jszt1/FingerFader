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


class HandModel(nn.Module):
    """Model neuronowy do predykcji kąta rozwarcia dłoni"""

    def __init__(self, input_size=84, hidden_size=128):  # 21 punktów * 2 ręce * 2 koordynaty = 84
        super(HandModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalizujemy do [0, 1], potem skalujemy do [0, 100]
        )

    def forward(self, x):
        return self.network(x) * 100  # Skalowanie do [0, 100]


class HandAnalyzer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Dane do trenowania/predykcji
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # MIDI setup
        self.midi_out = None
        self.setup_midi()

        # Parametry
        self.DATA_COLLECTION_ROUNDS = 5
        self.PREPARATION_TIME = 5
        self.CSV_FILE = 'data.csv'

    def setup_midi(self):
        """Inicjalizacja MIDI"""
        try:
            pygame.midi.init()
            if pygame.midi.get_count() > 0:
                self.midi_out = pygame.midi.Output(0)
                print("MIDI zainicjalizowane pomyślnie")
            else:
                print("Nie znaleziono urządzeń MIDI")
        except Exception as e:
            print(f"Błąd inicjalizacji MIDI: {e}")

    def calculate_finger_distance(self, landmarks) -> float:
        """Oblicza odległość między kciukiem a palcem wskazującym"""
        if len(landmarks) < 21:
            return 0.0

        # Kciuk (punkt 4) i palec wskazujący (punkt 8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]

        distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
        return distance

    def landmarks_to_array(self, landmarks) -> np.array:
        """Konwertuje landmarki do tablicy numpy"""
        if landmarks is None:
            return np.zeros(42)  # 21 punktów * 2 koordynaty

        coords = []
        for landmark in landmarks:
            coords.extend([landmark.x, landmark.y])

        return np.array(coords)

    def normalize_distance_to_percentage(self, distance: float) -> float:
        """Konwertuje odległość na procent rozwarcia (0-100%)"""
        # Normalizacja oparta na empirycznych wartościach
        # Minimalna odległość (zamknięta dłoń) ≈ 0.01
        # Maksymalna odległość (otwarta dłoń) ≈ 0.15
        min_dist = 0.01
        max_dist = 0.15

        normalized = (distance - min_dist) / (max_dist - min_dist)
        percentage = max(0, min(100, normalized * 100))

        return percentage

    def send_midi_data(self, left_percentage: float, right_percentage: float):
        """Wysyła dane MIDI z procentem rozwarcia dla każdej ręki"""
        if self.midi_out is None:
            return

        try:
            # Konwertuj procenty na wartości MIDI (0-127)
            left_midi = int(left_percentage * 127 / 100)
            right_midi = int(right_percentage * 127 / 100)

            # Wyślij Control Change messages
            # CC 1 dla lewej ręki, CC 2 dla prawej ręki
            self.midi_out.write([[[0xB0, 1, left_midi], pygame.midi.time()]])
            self.midi_out.write([[[0xB0, 2, right_midi], pygame.midi.time()]])

        except Exception as e:
            print(f"Błąd wysyłania MIDI: {e}")

    def collect_data_mode(self):
        """Tryb zbierania danych treningowych"""
        print(f"=== TRYB ZBIERANIA DANYCH ===")
        print(f"Będę zadawać {self.DATA_COLLECTION_ROUNDS} pytań o różne kąty rozwarcia")

        cap = cv2.VideoCapture(0)

        for round_num in range(self.DATA_COLLECTION_ROUNDS):
            # Wylosuj kąt
            target_angle = random.randint(0, 10) * 10  # 0, 10, 20, ..., 100

            print(f"\nRunda {round_num + 1}/{self.DATA_COLLECTION_ROUNDS}")
            print(f"Ustaw obie ręce na {target_angle}% rozwarcia")
            print(f"Masz {self.PREPARATION_TIME} sekund...")

            # Odliczanie
            for i in range(self.PREPARATION_TIME, 0, -1):
                print(f"Przygotowanie: {i}s")
                time.sleep(1)

            print("ZDJĘCIE!")

            # Zrób zdjęcie i zapisz dane
            ret, frame = cap.read()
            if ret:
                self.capture_and_save_data(frame, target_angle)

            time.sleep(1)  # Krótka pauza między rundami

        cap.release()
        print(f"\nZbieranie danych zakończone! Dane zapisane w {self.CSV_FILE}")

    def capture_and_save_data(self, frame, target_angle):
        """Przechwytuje dane z klatki i zapisuje do CSV"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hands_data = []

            # Przetwórz wszystkie wykryte ręce
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_array = self.landmarks_to_array(hand_landmarks.landmark)
                hands_data.extend(landmarks_array)

            # Uzupełnij do 2 rąk jeśli potrzeba
            while len(hands_data) < 84:  # 2 ręce * 42 koordynaty
                hands_data.extend([0] * 42)

            # Przygotuj dane do zapisu
            data_row = hands_data + [target_angle]

            # Nazwy kolumn
            columns = []
            for hand_id in ['left', 'right']:
                for point_id in range(21):
                    columns.extend([f'{hand_id}_x_{point_id}', f'{hand_id}_y_{point_id}'])
            columns.append('target_angle')

            # Zapisz do CSV
            df_new = pd.DataFrame([data_row], columns=columns)

            if os.path.exists(self.CSV_FILE):
                df_new.to_csv(self.CSV_FILE, mode='a', header=False, index=False)
            else:
                df_new.to_csv(self.CSV_FILE, index=False)

            print(f"✓ Zapisano dane dla kąta {target_angle}%")
        else:
            print("✗ Nie wykryto rąk - spróbuj ponownie")

    def load_model(self):
        """Ładuje wytrenowany model"""
        try:
            self.model = HandModel().to(self.device)
            self.model.load_state_dict(torch.load('hand_model.pth', map_location=self.device))
            self.model.eval()

            self.scaler = joblib.load('scaler.pkl')
            print("Model załadowany pomyślnie")
            return True
        except Exception as e:
            print(f"Błąd ładowania modelu: {e}")
            return False

    def inference_mode(self):
        """Tryb inferencji z wyświetlaniem i MIDI"""
        if not self.load_model():
            print("Nie można załadować modelu. Uruchom najpierw trening.")
            return

        print("=== TRYB INFERENCJI ===")
        print("Naciśnij 'q' aby zakończyć")

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Odbij obraz horizontalnie dla lepszego UX
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            left_percentage = 0
            right_percentage = 0

            if results.multi_hand_landmarks and results.multi_handedness:
                hands_data = np.zeros(84)  # 2 ręce * 42 koordynaty

                for idx, (hand_landmarks, handedness) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)):
                    # Rysuj landmarki
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Określ która ręka (uwaga: MediaPipe odwraca left/right)
                    is_left = handedness.classification[0].label == 'Right'  # Odwrócone!
                    hand_offset = 0 if is_left else 42

                    # Zapisz dane ręki
                    landmarks_array = self.landmarks_to_array(hand_landmarks.landmark)
                    hands_data[hand_offset:hand_offset + 42] = landmarks_array

                    # Oblicz procent rozwarcia dla tej ręki
                    distance = self.calculate_finger_distance(hand_landmarks.landmark)
                    percentage = self.normalize_distance_to_percentage(distance)

                    if is_left:
                        left_percentage = percentage
                    else:
                        right_percentage = percentage

                    # Wyświetl informacje na ekranie
                    hand_label = "Lewa" if is_left else "Prawa"
                    cv2.putText(frame, f"{hand_label}: {percentage:.1f}%",
                                (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                # Predykcja modelem (opcjonalnie)
                try:
                    hands_data_scaled = self.scaler.transform([hands_data])
                    hands_tensor = torch.FloatTensor(hands_data_scaled).to(self.device)

                    with torch.no_grad():
                        prediction = self.model(hands_tensor).cpu().numpy()[0][0]

                    cv2.putText(frame, f"Predykcja modelu: {prediction:.1f}%",
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 0), 2)
                except Exception as e:
                    pass  # Ignoruj błędy predykcji

                # Wyślij dane MIDI
                self.send_midi_data(left_percentage, right_percentage)

            # Wyświetl obraz
            cv2.imshow('Hand Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def cleanup(self):
        """Czyszczenie zasobów"""
        if self.midi_out:
            self.midi_out.close()
        pygame.midi.quit()


def main():
    parser = argparse.ArgumentParser(description='Analiza rozwarcia dłoni')
    parser.add_argument('--mode', choices=['collect', 'inference'],
                        default='collect', help='Tryb pracy')

    args = parser.parse_args()

    analyzer = HandAnalyzer()

    try:
        if args.mode == 'collect':
            analyzer.collect_data_mode()
        else:
            analyzer.inference_mode()
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    main()