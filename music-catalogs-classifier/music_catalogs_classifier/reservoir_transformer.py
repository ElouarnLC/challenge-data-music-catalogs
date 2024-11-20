import torch
import torch.nn as nn
import torch.optim as optim
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.model_selection import train_test_split
import numpy as np

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EMBEDDING_SIZE = 128  # Taille des embeddings pour tous les inputs
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
LEARNING_RATE = 1e-4
EPOCHS = 10


# Simulation des données d'entrée
def load_data():
    # Simule les données d'input (genres, instruments, moods)
    X_genres = np.random.rand(1000, 90)
    X_instruments = np.random.rand(1000, 112)
    X_moods = np.random.rand(1000, 46)

    # Simule les données de sortie (étiquettes binaires pour chaque catégorie)
    y_genres = np.random.randint(0, 2, (1000, 90))
    y_instruments = np.random.randint(0, 2, (1000, 112))
    y_moods = np.random.randint(0, 2, (1000, 46))

    return (X_genres, X_instruments, X_moods), (y_genres, y_instruments, y_moods)


# Charger les données
(X_genres, X_instruments, X_moods), (y_genres, y_instruments, y_moods) = load_data()

# Train-test split
X_genres_train, X_genres_test, y_genres_train, y_genres_test = train_test_split(
    X_genres, y_genres, test_size=0.2, random_state=42
)
X_instruments_train, X_instruments_test, y_instruments_train, y_instruments_test = (
    train_test_split(X_instruments, y_instruments, test_size=0.2, random_state=42)
)
X_moods_train, X_moods_test, y_moods_train, y_moods_test = train_test_split(
    X_moods, y_moods, test_size=0.2, random_state=42
)

# Préparation des données
X_train = np.concatenate([X_genres_train, X_instruments_train, X_moods_train], axis=1)
X_test = np.concatenate([X_genres_test, X_instruments_test, X_moods_test], axis=1)

y_train = np.concatenate([y_genres_train, y_instruments_train, y_moods_train], axis=1)
y_test = np.concatenate([y_genres_test, y_instruments_test, y_moods_test], axis=1)

# Convertir les données en tensors PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

# Initialisation des réservoirs (genre, instrument, mood)
reservoir_Genre = Reservoir(
    units=150,
    sr=0.9,  # Spectral radius
    lr=1,  # Leak rate
    input_scaling=1.0,
)

reservoir_Instrument = Reservoir(units=150, sr=0.9, lr=1, input_scaling=1.0)

reservoir_Mood = Reservoir(units=150, sr=0.9, lr=1, input_scaling=1.0)

# Readout pour chaque réservoir
readout_Genre = Ridge(ridge=1e-4)
readout_Instrument = Ridge(ridge=1e-4)
readout_Mood = Ridge(ridge=1e-4)

# Création des modèles
model_Genre = reservoir_Genre >> readout_Genre
model_Instrument = reservoir_Instrument >> readout_Instrument
model_Mood = reservoir_Mood >> readout_Mood

# Entraîner les réservoirs
model_Genre.fit(X_genres_train)
model_Instrument.fit(X_instruments_train)
model_Mood.fit(X_moods_train)

# Obtenir les sorties des réservoirs
y_genres_train_pred = model_Genre.run(X_genres_train)
y_instruments_train_pred = model_Instrument.run(X_instruments_train)
y_moods_train_pred = model_Mood.run(X_moods_train)

# Combine les sorties des réservoirs
X_train_reservoirs = np.concatenate(
    [y_genres_train_pred, y_instruments_train_pred, y_moods_train_pred], axis=1
)
X_test_reservoirs = np.concatenate(
    [
        model_Genre.run(X_genres_test),
        model_Instrument.run(X_instruments_test),
        model_Mood.run(X_moods_test),
    ],
    axis=1,
)

# Dataset et DataLoader pour les sorties des réservoirs
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train_reservoirs, dtype=torch.float32).to(DEVICE), y_train_tensor
)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_test_reservoirs, dtype=torch.float32).to(DEVICE), y_test_tensor
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# Modèle de Transformeur
class MultiTaskTransformer(nn.Module):
    def __init__(
        self, input_size, embedding_size, num_heads, num_layers, num_labels, dropout
    ):
        super(MultiTaskTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, embedding_size)  # Embedding Layer
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(embedding_size, num_labels)  # Final Classifier

    def forward(self, x):
        # Pass input through the embedding layer
        embedded = self.embedding(x)

        # Add a positional encoding (if needed)
        embedded = embedded.unsqueeze(1)  # Add sequence dimension

        # Transformer expects (batch, seq_len, embedding_size)
        transformer_output = self.transformer(embedded, embedded)

        # Take only the output of the first token (classification token equivalent)
        output = transformer_output[:, 0, :]  # Extract first token

        # Pass through the classifier
        predictions = self.classifier(output)
        return predictions


# Initialiser le modèle
model = MultiTaskTransformer(
    input_size=X_train_reservoirs.shape[
        1
    ],  # Taille totale des sorties concaténées des réservoirs
    embedding_size=EMBEDDING_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_labels=y_train.shape[1],  # Nombre total de catégories en sortie
    dropout=DROPOUT,
).to(DEVICE)

# Optimiseur et fonction de perte
criterion = nn.BCEWithLogitsLoss()  # Fonction de perte pour les étiquettes binaires
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Entraînement
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")


# Évaluation
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(test_loader)}")


# Entraîner le modèle
train_model(model, train_loader, criterion, optimizer, EPOCHS)

# Évaluer le modèle
evaluate_model(model, test_loader, criterion)

# Prédictions
with torch.no_grad():
    sample_input = X_test_tensor[:5]  # Exemple de données
    sample_predictions = model(sample_input)
    predictions_binary = (
        torch.sigmoid(sample_predictions) > 0.5
    ).int()  # Convertir en 0/1
    print("Predictions (binary):", predictions_binary)
