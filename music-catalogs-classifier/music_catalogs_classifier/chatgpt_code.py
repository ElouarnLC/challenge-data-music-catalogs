import torch
import torch.nn as nn

# Taille des embeddings pour chaque catégorie
EMBEDDING_SIZE = 128

# Exemple : embedding pour Genres
genre_embedding = nn.Linear(input_genres_tags_data.shape[1], EMBEDDING_SIZE)
instrument_embedding = nn.Linear(input_instruments_tags_data.shape[1], EMBEDDING_SIZE)
mood_embedding = nn.Linear(input_moods_tags_data.shape[1], EMBEDDING_SIZE)

# Exemple : calcul des embeddings combinés
instrument_genre_embedding = nn.Linear(EMBEDDING_SIZE * 2, EMBEDDING_SIZE)
mood_genre_embedding = nn.Linear(EMBEDDING_SIZE * 2, EMBEDDING_SIZE)
mood_genre_instrument_embedding = nn.Linear(EMBEDDING_SIZE * 3, EMBEDDING_SIZE)


# Fonction d'embedding
def compute_combined_embeddings(genre, instrument, mood):
    combined_embeddings = []
    combined_embeddings.append(genre_embedding(genre))  # Genre
    combined_embeddings.append(instrument_embedding(instrument))  # Instrument
    combined_embeddings.append(mood_embedding(mood))  # Mood

    # Combinations
    combined_embeddings.append(
        instrument_genre_embedding(torch.cat([genre, instrument], dim=-1))
    )
    combined_embeddings.append(mood_genre_embedding(torch.cat([genre, mood], dim=-1)))
    combined_embeddings.append(
        mood_genre_instrument_embedding(torch.cat([genre, instrument, mood], dim=-1))
    )

    return torch.stack(
        combined_embeddings, dim=1
    )  # Shape: (Batch, Num_Tokens, EMBEDDING_SIZE)


class TransformerModel(nn.Module):
    def __init__(self, embedding_size, num_tokens, num_heads, num_layers):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.classifier = nn.Linear(
            embedding_size, output_genres_tags_data.shape[1]
        )  # Adapte à la sortie cible

    def forward(self, inputs):
        # Transformer expects (Batch, Num_Tokens, Embedding_Size)
        transformer_output = self.transformer(inputs, inputs)
        # Pooling ou classification sur la séquence complète
        classification_output = self.classifier(transformer_output.mean(dim=1))
        return classification_output


# Exemple : pipeline complet
embedding_size = EMBEDDING_SIZE
num_heads = 4  # Nombre de têtes d'attention
num_layers = 2  # Nombre de couches Transformer

# Initialiser le modèle
model = TransformerModel(
    embedding_size, num_tokens=6, num_heads=num_heads, num_layers=num_layers
)

# Données d'exemple
genres = torch.rand((batch_size, input_genres_tags_data.shape[1]))
instruments = torch.rand((batch_size, input_instruments_tags_data.shape[1]))
moods = torch.rand((batch_size, input_moods_tags_data.shape[1]))

# Calcul des embeddings
inputs = compute_combined_embeddings(genres, instruments, moods)

# Passage au Transformer
output = model(inputs)


criterion = nn.BCEWithLogitsLoss()  # Si multi-étiquettes
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, y_true)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")


# Embeddings des données d'entrée brutes
input_genre_embed = genre_embedding(input_genres_tags_data)
input_instrument_embed = instrument_embedding(input_instruments_tags_data)
input_mood_embed = mood_embedding(input_moods_tags_data)

# Embeddings des sorties des réservoirs
reservoir_genre_embed = reservoir_Genre.run(input_genres_tags_data)
reservoir_instrument_embed = reservoir_Instrument.run(input_instruments_tags_data)
reservoir_mood_embed = reservoir_Mood.run(input_moods_tags_data)

# Concatenation des embeddings (entrées brutes + sorties des réservoirs)
inputs = torch.cat(
    [
        input_genre_embed,
        input_instrument_embed,
        input_mood_embed,
        reservoir_genre_embed,
        reservoir_instrument_embed,
        reservoir_mood_embed,
    ],
    dim=1,
)  # (Batch, Num_Tokens, EMBEDDING_SIZE)


class TransformerWithReservoirs(nn.Module):
    def __init__(self, embedding_size, num_heads, num_layers):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.classifier = nn.Linear(
            embedding_size, output_genres_tags_data.shape[1]
        )  # Adapté à la sortie

    def forward(self, inputs):
        # Transformer les entrées
        transformer_output = self.transformer(inputs, inputs)
        # Moyenne sur les tokens pour la classification finale
        pooled_output = transformer_output.mean(dim=1)
        classification_output = self.classifier(pooled_output)
        return classification_output


# Initialisation du modèle
model = TransformerWithReservoirs(embedding_size=128, num_heads=4, num_layers=2)

# Fonction de perte et optimiseur
criterion = nn.BCEWithLogitsLoss()  # Pour multi-étiquettes
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Données d'exemple
inputs_combined = compute_combined_embeddings(inputs, reservoir_outputs)

# Entraînement
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(inputs_combined)
    loss = criterion(predictions, y_true)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
