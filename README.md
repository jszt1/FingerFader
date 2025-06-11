# System Analizy Rozwarcia Dłoni

System wykorzystujący MediaPipe do analizy estymacji pozycji dłoni i przewidywania stopnia rozwarcia między kciukiem a palcem wskazującym.

## Funkcje

- **Zbieranie danych treningowych** z kamerki w czasie rzeczywistym
- **Trening modelu neuronowego** na GPU NVIDIA
- **Inferencja w czasie rzeczywistym** z wizualizacją
- **Wysyłanie danych przez MIDI** dla integracji z oprogramowaniem muzycznym

## Instalacja

1. **Zainstaluj wymagane biblioteki:**
```bash
pip install -r requirements.txt
```

2. **Sprawdź dostępność GPU NVIDIA (opcjonalnie):**
```python
import torch
print(torch.cuda.is_available())
```

## Użytkowanie

### 1. Zbieranie danych treningowych

```bash
python hand_analize.py --mode collect
```

Program wykona 5 rund zbierania danych:
- Wylosuje kąt rozwarcia (0-100% w odstępach 10%)
- Da 5 sekund na przygotowanie pozycji obu rąk
- Zrobi "zdjęcie" i zapisze landmarki do `data.csv`

**Wskazówki dla zbierania danych:**
- Upewnij się, że obie ręce są widoczne w kadrze
- Ustaw oświetlenie równomierne
- Trzymaj ręce w podobnej odległości od kamery
- Zbierz dane kilka razy dla lepszej jakości modelu

### 2. Trenowanie modelu

```bash
python train_model.py
```

Skrypt automatycznie:
- Załaduje dane z `data.csv`
- Przygotuje dane (normalizacja, podział train/validation)
- Wytrenuje model neuronowy na GPU/CPU
- Wyświetli metryki ewaluacji
- Zapisze model (`hand_model.pth`) i scaler (`scaler.pkl`)
- Wygeneruje wykresy treningu

### 3. Inferencja w czasie rzeczywistym

```bash
python hand_analysis.py --mode inference
```

W tym trybie system:
- Ładuje wytrenowany model
- Analizuje obraz z kamery w czasie rzeczywistym
- Wyświetla landmarki dłoni
- Pokazuje procent rozwarcia dla każdej ręki
- Wysyła dane MIDI (Control Change):
  - CC 1: Lewa ręka (0-127)
  - CC 2: Prawa ręka (0-127)

**Sterowanie:**
- `Q` - wyjście z programu

## Konfiguracja MIDI

System automatycznie wykrywa dostępne urządzenia MIDI. Jeśli nie masz fizycznego interfejsu MIDI, możesz użyć wirtualnego:

### Windows:
- Zainstaluj LoopMIDI lub podobne oprogramowanie

### Linux:
```bash
sudo apt-get install qjackctl
# lub użyj ALSA MIDI
```

### macOS:
- Użyj wbudowanego IAC Driver w Audio MIDI Setup

## Struktura plików

```
├── hand_analysis.py      # Program główny
├── train_model.py        # Skrypt trenowania
├── requirements.txt      # Biblioteki
├── data.csv             # Dane treningowe (generowane)
├── hand_model.pth       # Wytrenowany model (generowany)
├── scaler.pkl           # Scaler do normalizacji (generowany)
├── training_history.png # Wykres treningu (generowany)
└── model_evaluation.png # Wykres ewaluacji (generowany)
```

## Parametry konfiguracyjne

### W `hand_analysis.py`:
```python
DATA_COLLECTION_ROUNDS = 5    # Liczba rund zbierania danych
PREPARATION_TIME = 5          # Czas przygotowania (sekundy)
CSV_FILE = 'data.csv'        # Nazwa pliku danych
```

### W `train_model.py`:
```python
batch_size = 32              # Rozmiar batcha
learning_rate = 0.001        # Współczynnik uczenia
epochs = 100                 # Liczba epok
validation_split = 0.2       # Procent danych walidacyjnych
```

## Architektura modelu

Model neuronowy składa się z:
- **Warstwa wejściowa:** 84 neurony (2 ręce × 21 punktów × 2 koordynaty)
- **Warstwy ukryte:** 128 → 128 → 64 neurony z ReLU i Dropout
- **Warstwa wyjściowa:** 1 neuron z aktywacją Sigmoid (0-100%)

## Rozwiązywanie problemów

### Błędy kamery:
```bash
# Sprawdź dostępne kamery
ls /dev/video*  # Linux
# lub użyj innych indeksów kamery w cv2.VideoCapture(1), (2), etc.
```

### Problemy z MIDI:
```python
import pygame.midi
pygame.midi.init()
print(f"Liczba urządzeń MIDI: {pygame.midi.get_count()}")
for i in range(pygame.midi.get_count()):
    print(f"Urządzenie {i}: {pygame.midi.get_device_info(i)}")
```

### Słaba jakość modelu:
1. Zbierz więcej danych (zalecane 50+ próbek)
2. Zapewnij różnorodność pozycji rąk
3. Zwiększ liczbę epok treningu
4. Sprawdź oświetlenie podczas zbierania danych

## Metryki modelu

- **MAE (Mean Absolute Error):** Średni błąd bezwzględny w procentach
- **R² Score:** Współczynnik determinacji (0-1, wyższy = lepszy)

**Dobre wyniki:**
- MAE < 10%
- R² > 0.8

## Przykładowe użycie MIDI

System wysyła Control Change messages, które można mapować w DAW:
- **Ableton Live:** MIDI Learn → wybierz parametr
- **FL Studio:** Tools → Last tweaked → Link to controller
- **Reaper:** Learn → Right-click parametr

## Rozszerzenia

Możliwe ulepszenia systemu:
1. **Kalibracja użytkownika** - personalizacja zakresów rozwarcia
2. **Więcej gestów** - analiza innych konfiguracji palców  
3. **Filtrowanie sygnału** - wygładzanie danych MIDI
4. **Interfejs GUI** - graficzny panel kontrolny
5. **OSC zamiast MIDI** - większa precyzja transmisji danych

## Wymagania systemowe

- **Python:** 3.8+
- **RAM:** 4GB+ (8GB+ zalecane dla treningu)
- **GPU:** NVIDIA z CUDA (opcjonalne, ale zalecane)
- **Kamera:** Dowolna kamera USB/wbudowana
- **System:** Windows 10+, Ubuntu 18.04+, macOS 10.15+

## Licencja

Projekt open source - można swobodnie modyfikować i dostosowywać do własnych potrzeb.