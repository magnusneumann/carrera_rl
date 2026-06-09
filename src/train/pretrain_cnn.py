import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_PATH = "data/lidar_agent_dataset_20260609_133207.npz"
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/pretrained_vision_backbone.pth"

print(f"Nutze Gerät: {DEVICE}")

# --- 1. PYTORCH DATASET DEFINIEREN ---
class CarreraDataset(Dataset):
    def __init__(self, states, actions):
        # Konvertierung zu Float32 und Normalisierung auf [0, 1] erst hier im Dataset,
        # um RAM zu sparen.
        self.states = torch.tensor(states, dtype=torch.float32) / 255.0
        self.actions = torch.tensor(actions, dtype=torch.long)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# --- 2. CNN ARCHITEKTUR (SB3-kompatibel) ---
class PretrainCNN(nn.Module):
    def __init__(self, num_actions):
        super(PretrainCNN, self).__init__()
        
        # Feature Extractor (Standard NatureCNN Struktur aus SB3)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # LazyLinear berechnet die Eingangsdimension nach dem Flatten automatisch,
        # unabhängig von Rundungsfehlern bei 100x166er Bildern.
        self.linear_layer = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU()
        )
        
        # Der finale Klassifikations-Kopf für die Aktionen
        self.action_head = nn.Linear(512, num_actions)

    def forward(self, x):
        features = self.cnn(x)
        features = self.linear_layer(features)
        action_logits = self.action_head(features)
        return action_logits

# --- 3. DATEN LADEN & VORBEREITEN ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Datensatz nicht gefunden unter: {DATA_PATH}")

print("Lade Datensatz...")
data = np.load(DATA_PATH)
X_data = data["states"]   # (N, 3, 100, 166)
Y_data = data["actions"]  # (N,)

# Ermitteln, wie viele diskrete Aktionen es gibt
num_classes = len(np.unique(Y_data))
print(f"Datensatz erfolgreich geladen. Samples: {X_data.shape[0]}, Erkannte Aktionen: {num_classes}")

# Train-Validation-Split (80% Train, 20% Validierung)
X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

train_dataset = CarreraDataset(X_train, Y_train)
val_dataset = CarreraDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. TRAINING REPRODUZIERBAR INITIALISIEREN ---
model = PretrainCNN(num_actions=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss() # Da wir diskrete Aktionen klassifizieren
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Einmaliger Dummy-Forward-Pass, damit LazyLinear die Shapes initialisiert
dummy_input = torch.zeros(1, 3, 100, 166).to(DEVICE)
_ = model(dummy_input)

# --- 5. TRAINING SCHLEIFE ---
print("\nStarte Vortraining (Behavioral Cloning)...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for states, actions in train_loader:
        states, actions = states.to(DEVICE), actions.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * states.size(0)
        _, predicted = outputs.max(1)
        total_train += actions.size(0)
        correct_train += predicted.eq(actions).sum().item()
        
    train_loss /= len(train_loader.dataset)
    train_acc = 100.0 * correct_train / total_train
    
    # Validierung
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for states, actions in val_loader:
            states, actions = states.to(DEVICE), actions.to(DEVICE)
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            val_loss += loss.item() * states.size(0)
            _, predicted = outputs.max(1)
            total_val += actions.size(0)
            correct_val += predicted.eq(actions).sum().item()
            
    val_loss /= len(val_loader.dataset)
    val_acc = 100.0 * correct_val / total_val
    
    print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] -> "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# --- 6. SPEICHERN ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nVortraining abgeschlossen! Modellgewichte gespeichert unter: {MODEL_SAVE_PATH}")