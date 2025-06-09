import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os


class HandModel(nn.Module):
    """Model neuronowy do predykcji kąta rozwarcia pojedynczej dłoni"""

    def __init__(self, input_size=42, hidden_size=128):  # 21 punktów * 2 koordynaty = 42 (jedna ręka)
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
            nn.Sigmoid()  # Normalizujemy do [0, 1]
        )

    def forward(self, x):
        return self.network(x) * 100  # Skalowanie do [0, 100]


class HandModelTrainer:
    def __init__(self, csv_file='hand_data.csv'):
        self.csv_file = csv_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Używam urządzenia: {self.device}")

        self.model = None
        self.scaler = StandardScaler()

        # Hiperparametry
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 100
        self.validation_split = 0.2

    def load_data(self):
        """Ładuje i przygotowuje dane z CSV (każda ręka jako osobna próbka)"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Plik {self.csv_file} nie istnieje. Uruchom najpierw zbieranie danych.")

        print(f"Ładowanie danych z {self.csv_file}...")
        df = pd.read_csv(self.csv_file)

        print(f"Załadowano {len(df)} próbek (każda ręka osobno)")
        
        # Analiza rozkładu danych
        print(f"Rozkład kątów docelowych:")
        print(df['target_angle'].value_counts().sort_index())
        
        # Sprawdź czy mamy wystarczająco danych
        if len(df) < 10:
            print("⚠️  Uwaga: Bardzo mało danych treningowych. Zalecane minimum to 50+ próbek.")

        # Przygotuj dane wejściowe (wszystkie kolumny oprócz target_angle)
        feature_columns = [col for col in df.columns if col != 'target_angle']
        X = df[feature_columns].values
        y = df['target_angle'].values

        # Sprawdź czy są wartości NaN
        if np.isnan(X).any() or np.isnan(y).any():
            print("⚠️  Wykryto wartości NaN - czyszczenie danych...")
            df_clean = df.dropna()
            X = df_clean[feature_columns].values
            y = df_clean['target_angle'].values
            print(f"Po czyszczeniu zostało {len(X)} próbek")

        # Informacje o rozmiarze danych
        print(f"Rozmiar danych wejściowych: {X.shape}")
        print(f"Rozmiar danych wyjściowych: {y.shape}")

        return X, y

    def prepare_data(self, X, y):
        """Przygotowuje dane do treningu"""
        # Podział na zbiór treningowy i walidacyjny
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42, stratify=None
        )

        # Normalizacja danych
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Normalizacja etykiet do [0, 1]
        y_train_norm = y_train / 100.0
        y_val_norm = y_val / 100.0

        # Konwersja do tensorów
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train_norm).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val_norm).unsqueeze(1)

        # DataLoadery
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Dane treningowe: {len(X_train)} próbek")
        print(f"Dane walidacyjne: {len(X_val)} próbek")

        return train_loader, val_loader, X_val, y_val

    def train_model(self, train_loader, val_loader):
        """Trenuje model"""
        # Inicjalizacja modelu
        input_size = 42  # 21 punktów * 2 koordynaty (jedna ręka)
        self.model = HandModel(input_size=input_size).to(self.device)

        # Optymalizator i funkcja straty
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Historia treningu
        train_losses = []
        val_losses = []

        print(f"\nRozpoczynanie treningu na {self.epochs} epok...")
        print(f"Model dla pojedynczej ręki - uniwersalny (input_size={input_size})")

        for epoch in range(self.epochs):
            # Faza treningowa
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X) / 100.0  # Normalizacja do [0, 1]
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Faza walidacyjna
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X) / 100.0
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            # Średnie straty
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Wyświetl postęp co 10 epok
            if (epoch + 1) % 10 == 0:
                print(f"Epoka {epoch + 1}/{self.epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        return train_losses, val_losses

    def evaluate_model(self, X_val, y_val):
        """Ewaluuje model na danych walidacyjnych"""
        self.model.eval()

        X_val_scaled = self.scaler.transform(X_val)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_val_tensor).cpu().numpy().flatten()

        # Metryki
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)

        print(f"\n=== EWALUACJA MODELU (UNIWERSALNY) ===")
        print(f"Mean Absolute Error: {mae:.2f}%")
        print(f"R² Score: {r2:.4f}")

        # Analiza predykcji
        print(f"\nPorównanie predykcji:")
        for i in range(min(10, len(y_val))):
            print(f"Prawdziwy: {y_val[i]:5.1f}% | Predykcja: {predictions[i]:5.1f}% | "
                  f"Błąd: {abs(y_val[i] - predictions[i]):4.1f}%")

        return predictions, mae, r2

    def plot_training_history(self, train_losses, val_losses):
        """Rysuje wykres historii treningu"""
        plt.figure(figsize=(12, 4))

        # Wykres strat
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History - Universal Hand Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Wykres predykcji vs rzeczywiste wartości (jeśli dostępne)
        plt.subplot(1, 2, 2)
        plt.title('Model Performance')
        plt.text(0.5, 0.5, 'Uruchom ewaluację\ndla wykresu predykcji',
                 ha='center', va='center', transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.savefig('training_history_universal.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("Wykres zapisany jako 'training_history_universal.png'")

    def plot_predictions(self, y_true, y_pred):
        """Rysuje wykres porównania predykcji z rzeczywistymi wartościami"""
        plt.figure(figsize=(12, 8))

        # Wykres scatter
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.7)
        plt.plot([0, 100], [0, 100], 'r--', lw=2)
        plt.xlabel('Rzeczywiste wartości (%)')
        plt.ylabel('Predykcje (%)')
        plt.title('Predykcje vs Rzeczywiste wartości')
        plt.grid(True)

        # Histogram błędów
        plt.subplot(2, 2, 2)
        errors = y_pred - y_true
        plt.hist(errors, bins=20, alpha=0.7)
        plt.xlabel('Błąd predykcji (%)')
        plt.ylabel('Liczba próbek')
        plt.title('Rozkład błędów')
        plt.grid(True)

        # Rozkład prawdziwych wartości
        plt.subplot(2, 2, 3)
        plt.hist(y_true, bins=20, alpha=0.7, color='green')
        plt.xlabel('Rzeczywiste wartości (%)')
        plt.ylabel('Liczba próbek')
        plt.title('Rozkład prawdziwych wartości')
        plt.grid(True)

        # Rozkład predykcji
        plt.subplot(2, 2, 4)
        plt.hist(y_pred, bins=20, alpha=0.7, color='orange')
        plt.xlabel('Predykcje (%)')
        plt.ylabel('Liczba próbek')
        plt.title('Rozkład predykcji')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('model_evaluation_universal.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("Wykres ewaluacji zapisany jako 'model_evaluation_universal.png'")

    def save_model(self):
        """Zapisuje wytrenowany model i scaler"""
        torch.save(self.model.state_dict(), 'hand_model.pth')
        joblib.dump(self.scaler, 'scaler.pkl')

        print(f"\n✓ Model zapisany jako 'hand_model.pth'")
        print(f"✓ Scaler zapisany jako 'scaler.pkl'")
        print(f"✓ Model przygotowany jako uniwersalny dla dowolnej ręki (input_size=42)")

    def analyze_data_distribution(self, df):
        """Analizuje rozkład danych treningowych"""
        print(f"\n=== ANALIZA DANYCH TRENINGOWYCH ===")
        
        print("Rozkład kątów w całym datasecie:")
        angles = df['target_angle'].value_counts().sort_index()
        print(angles)
        
        print(f"\nŁączna liczba próbek: {len(df)}")
        print(f"Średni kąt: {df['target_angle'].mean():.1f}%")
        print(f"Mediana kąta: {df['target_angle'].median():.1f}%")

    def train_full_pipeline(self):
        """Pełny pipeline treningu"""
        try:
            # Załaduj dane
            X, y = self.load_data()
            
            # Dodatkowa analiza jeśli potrzebna
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                self.analyze_data_distribution(df)

            # Przygotuj dane
            train_loader, val_loader, X_val, y_val = self.prepare_data(X, y)

            # Trenuj model
            train_losses, val_losses = self.train_model(train_loader, val_loader)

            # Ewaluuj model
            predictions, mae, r2 = self.evaluate_model(X_val, y_val)

            # Rysuj wykresy
            self.plot_training_history(train_losses, val_losses)
            self.plot_predictions(y_val, predictions)

            # Zapisz model
            self.save_model()

            print(f"\n🎉 Trening zakończony pomyślnie!")
            print(f"📊 Końcowe metryki: MAE = {mae:.2f}%, R² = {r2:.4f}")
            print(f"📈 Model został wytrenowany na {len(X)} próbkach (uniwersalny dla dowolnej ręki)")

            return True

        except Exception as e:
            print(f"❌ Błąd podczas treningu: {e}")
            return False


def main():
    print("=== TRENING MODELU ANALIZY DŁONI (UNIWERSALNY) ===\n")

    trainer = HandModelTrainer()
    success = trainer.train_full_pipeline()

    if success:
        print("\n🚀 Model gotowy do użycia!")
        print("Uruchom: python hand_analyze.py --mode inference")
        print("\n💡 Zalety nowego podejścia:")
        print("• Każda ręka jest osobną próbką treningową")
        print("• Podwójnie więcej danych z tego samego wysiłku") 
        print("• Jeden uniwersalny model dla dowolnej ręki")
        print("• Prostsze zarządzanie modelami")
        print("• Lepsze wykorzystanie danych treningowych")
    else:
        print("\n💡 Spróbuj zebrać więcej danych treningowych:")
        print("python hand_analyze.py --mode collect")


if __name__ == "__main__":
    main()