# Fast Vision Transformer (ViT) Implementations

Dieses Repository enthält drei verschiedene Implementierungen von Vision Transformern (ViT) für die Bildklassifizierung auf dem CIFAR-10 Datensatz. Jede Implementierung verfolgt einen anderen Ansatz, um die Trainingszeit zu optimieren und die Leistung zu verbessern.

## Überblick der Implementierungen

### 1. FastViT (Basis-Implementierung)
- Verwendet das kleinste ViT-Modell (vit_tiny)
- Implementiert eine zusätzliche Feature-Projektionsschicht
- Reduzierter Datensatzumfang für schnelleres Training
- Einfache Datennormalisierung

### 2. FastTransferViT (Transfer Learning)
- Nutzt vortrainiertes ViT-Basismodell
- Implementiert Transfer Learning mit Layer-Freezing
- Enthält Data Augmentation
- Optimierter Klassifizierungskopf

### 3. FastCustomViT (Custom Architecture)
- Trainiert ein angepasstes ViT-Modell ohne Vortraining
- Fügt eine Faltungsschicht vor dem Embedding hinzu
- Verwendet 20% des CIFAR-10 Datensatzes
- Angepasste Architektur für verbesserte Leistung

## Technische Details

### Gemeinsame Merkmale
- Framework: PyTorch
- Basis-Architektur: Vision Transformer (ViT)
- Datensatz: CIFAR-10 (reduziert)
- Input-Größe: 224x224 Pixel

### Abhängigkeiten
```python
torch
torchvision
timm
matplotlib
```

### Hardware-Anforderungen
- CUDA-fähige GPU (empfohlen)
- Mindestens 8GB RAM
- CPU-Training möglich, aber deutlich langsamer

## Installation und Verwendung

1. Repository klonen:
```bash
git clone [repository-url]
```

2. Abhängigkeiten installieren:
```bash
pip install torch torchvision timm matplotlib
```

3. Modell auswählen und trainieren:
```bash
python fast_vit.py        # Basis-Implementierung
python fast_transfer_vit.py    # Transfer Learning
python fast_custom_vit.py      # Custom Architecture
```

## Modellvergleich

### FastViT
- 🔹 Schnellstes Training
- 🔹 Geringster Speicherverbrauch
- 🔸 Möglicherweise geringere Genauigkeit

### FastTransferViT
- 🔹 Beste Genauigkeit
- 🔹 Stabile Konvergenz
- 🔸 Höherer Speicherverbrauch

### FastCustomViT
- 🔹 Gutes Gleichgewicht zwischen Geschwindigkeit und Genauigkeit
- 🔹 Anpassbare Architektur
- 🔸 Benötigt mehr Feinabstimmung

## Modell-Ausgaben

Jedes Modell speichert:
- Trainierte Gewichte (.pth Datei)
- Trainings-Metriken
- Validierungs-Genauigkeit
