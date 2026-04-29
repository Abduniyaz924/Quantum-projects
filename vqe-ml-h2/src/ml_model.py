import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Load dataset
# -----------------------------
#data = np.load("data/vqe_dataset.npy", allow_pickle=True)

import os
import numpy as np

# -----------------------------
# Load dataset (robust path)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "vqe_dataset.npy")

data = np.load(DATA_PATH, allow_pickle=True)

print(f"Loaded dataset from: {DATA_PATH}")

X = np.array([d["theta_init"] for d in data])
y = np.array([d["theta_opt"] for d in data])

# convert to torch
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# -----------------------------
# Define model
# -----------------------------
class VQEModel(nn.Module):
    def __init__(self):
        super(VQEModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.net(x)

model = VQEModel()

# -----------------------------
# Training setup
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training loop
# -----------------------------
EPOCHS = 10000

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    predictions = model(X)
    loss = criterion(predictions, y)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# -----------------------------
# Save model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_PATH = os.path.join(DATA_DIR, "vqe_model.pth")

# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH)

print(f"Model saved at: {MODEL_PATH}")
