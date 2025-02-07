
# Projet IA - Challenge Data

## Interpreting Neural Network Predictions for Multi-Label Classification of a Music Catalog

### Run the Project

1.  **Clone the project**
    
    ```bash
    git clone https://github.com/ElouarnLC/challenge-data-music-catalogs.git
    
    ```
    
2.  **Navigate to the project folder**
    
    ```bash
    cd music-catalogs-classifier
    
    ```
    
3.  **Install dependencies**  
    Ensure you have [Poetry](https://python-poetry.org/docs/) installed, then run:
    
    ```bash
    poetry install
    
    ```
    
4.  **Open and run Jupyter notebooks**  
    Start Jupyter Notebook:
    
    ```bash
    poetry run jupyter notebook
    
    ```
    
    The main notebooks include:
    
    -   **`prepare_data.ipynb`** → Data visualization and preparation of dataframes
    -   **`train_transformer_final.ipynb`** → Training ESNs and the Transformer, including hyperparameter search
    -   **`run_model.ipynb`** → Running trained models on test data

### Get Data and Models

To run the model without training, download the preformatted data and trained models:

-   **[Download Data](https://drive.google.com/drive/folders/1biucwLWDea-wITQfaxgIH3Jl3tQ-3Qn4?usp=sharing)**
-   **[Download Models](https://drive.google.com/drive/folders/1_Fp9EG5nx4nFKh3TqbQ2q0iVf_h2I4os?usp=sharing)**

After downloading, place the files in the following directories within the `music-catalogs-classifier` subfolder:

-   **Data** → Place in the `data/` directory
-   **Models** → Place in the `models/` directory

