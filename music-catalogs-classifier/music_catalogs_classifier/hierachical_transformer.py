# Mewo Multi-Label Music Tagging Classification
# Comprehensive Implementation with Improvements

# Import required libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)


# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed()


# Configuration
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EMBEDDING_SIZE = 256
    NUM_HEADS = 4
    NUM_LAYERS = 3
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    PATIENCE = 10
    WEIGHT_DECAY = 1e-5


# Custom Focal Loss for Multi-Label Classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()


# Advanced Multi-Label Transformer Model
class HierarchicalMusicTagger(nn.Module):
    def __init__(self, input_dims, num_labels, config):
        super(HierarchicalMusicTagger, self).__init__()

        # Input embeddings
        self.input_embeddings = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, config.EMBEDDING_SIZE),
                    nn.ReLU(),
                    nn.Dropout(config.DROPOUT),
                )
                for name, dim in input_dims.items()
            }
        )

        # Multi-head attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_SIZE,
            nhead=config.NUM_HEADS,
            dropout=config.DROPOUT,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.NUM_LAYERS
        )

        # Classification head with hierarchical features
        self.classification_head = nn.Sequential(
            nn.Linear(config.EMBEDDING_SIZE, config.EMBEDDING_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.EMBEDDING_SIZE // 2, num_labels),
        )

    def forward(self, inputs):
        # Embed inputs
        embedded_inputs = [
            self.input_embeddings[key](input_tensor)
            for key, input_tensor in inputs.items()
        ]

        # Stack and process through transformer
        sequence = torch.stack(embedded_inputs, dim=1)
        transformed = self.transformer(sequence)

        # Aggregate features
        pooled = transformed.mean(dim=1)

        # Final classification
        return self.classification_head(pooled)


# Custom Dataset
class MusicTagDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.inputs.items()}, self.labels[idx]


# Data Loading and Preprocessing
def load_and_preprocess_data(data_path="../data/train"):
    # Load data similarly to original implementation
    X_genres = pd.read_csv(os.path.join(data_path, "input_genres_tags_data.csv"))
    X_instruments = pd.read_csv(
        os.path.join(data_path, "input_instruments_tags_data.csv")
    )
    X_moods = pd.read_csv(os.path.join(data_path, "input_moods_tags_data.csv"))

    y_genres = pd.read_csv(os.path.join(data_path, "output_genres_tags_data.csv"))
    y_instruments = pd.read_csv(
        os.path.join(data_path, "output_instruments_tags_data.csv")
    )
    y_moods = pd.read_csv(os.path.join(data_path, "output_moods_tags_data.csv"))

    # Drop ChallengeID
    X_cols_to_drop = ["ChallengeID"]
    y_cols_to_drop = ["ChallengeID"]

    X_genres = X_genres.drop(
        columns=[col for col in X_cols_to_drop if col in X_genres.columns]
    )
    X_instruments = X_instruments.drop(
        columns=[col for col in X_cols_to_drop if col in X_instruments.columns]
    )
    X_moods = X_moods.drop(
        columns=[col for col in X_cols_to_drop if col in X_moods.columns]
    )

    y_genres = y_genres.drop(
        columns=[col for col in y_cols_to_drop if col in y_genres.columns]
    )
    y_instruments = y_instruments.drop(
        columns=[col for col in y_cols_to_drop if col in y_instruments.columns]
    )
    y_moods = y_moods.drop(
        columns=[col for col in y_cols_to_drop if col in y_moods.columns]
    )

    # Combine all labels
    y_combined = pd.concat([y_genres, y_instruments, y_moods], axis=1)

    # Preprocessing
    scaler = StandardScaler()
    X_genres_scaled = scaler.fit_transform(X_genres)
    X_instruments_scaled = scaler.fit_transform(X_instruments)
    X_moods_scaled = scaler.fit_transform(X_moods)

    return {
        "X_genres": X_genres_scaled,
        "X_instruments": X_instruments_scaled,
        "X_moods": X_moods_scaled,
    }, y_combined.values


# Training Function
def train_model(model, train_loader, val_loader, config):
    criterion = FocalLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        train_losses = []

        for batch_inputs, batch_labels in train_loader:
            batch_inputs = {k: v.to(config.DEVICE) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = {k: v.to(config.DEVICE) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(config.DEVICE)

                outputs = model(batch_inputs)
                val_loss = criterion(outputs, batch_labels)
                val_losses.append(val_loss.item())

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered")
            break

    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))
    return model


# Evaluation Function
def evaluate_model(model, test_loader, config):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = {k: v.to(config.DEVICE) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(config.DEVICE)

            outputs = torch.sigmoid(model(batch_inputs))
            preds = (outputs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    metrics = {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, average="micro"),
        "Recall": recall_score(all_labels, all_preds, average="micro"),
        "F1 Score": f1_score(all_labels, all_preds, average="micro"),
        "Average Precision": average_precision_score(all_labels, all_preds),
    }

    return metrics


# Main Execution
def main():
    # Load and preprocess data
    inputs, labels = load_and_preprocess_data()

    # Split data
    X_train_inputs, X_test_inputs, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.2, random_state=42
    )
    X_train_inputs, X_val_inputs, y_train, y_val = train_test_split(
        X_train_inputs, y_train, test_size=0.2, random_state=42
    )

    # Prepare input dimensions
    input_dims = {k: v.shape[1] for k, v in X_train_inputs.items()}

    # Convert to torch tensors
    X_train_inputs = {k: torch.FloatTensor(v) for k, v in X_train_inputs.items()}
    X_val_inputs = {k: torch.FloatTensor(v) for k, v in X_val_inputs.items()}
    X_test_inputs = {k: torch.FloatTensor(v) for k, v in X_test_inputs.items()}

    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    y_test = torch.FloatTensor(y_test)

    # Create datasets and loaders
    train_dataset = MusicTagDataset(X_train_inputs, y_train)
    val_dataset = MusicTagDataset(X_val_inputs, y_val)
    test_dataset = MusicTagDataset(X_test_inputs, y_test)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Initialize model
    model = HierarchicalMusicTagger(
        input_dims=input_dims, num_labels=y_train.shape[1], config=Config
    ).to(Config.DEVICE)

    # Train model
    trained_model = train_model(model, train_loader, val_loader, Config)

    # Evaluate model
    metrics = evaluate_model(trained_model, test_loader, Config)

    # Print results
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
