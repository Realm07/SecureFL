# Secure Federated Learning Demonstration

This repository contains the source code for a proof-of-concept application demonstrating Secure Federated Learning (SFL). The application provides a visual simulation and interactive analysis of a federated learning system that uses Homomorphic Encryption (HE) to ensure client privacy during model training.

The primary objective of this project is to illustrate the functional viability and performance characteristics of SFL compared to traditional plaintext federated learning and centralized training methodologies.

## Table of Contents

- [Conceptual Background](#conceptual-background)
- [Technical Architecture](#technical-architecture)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Setup and Execution](#setup-and-execution)
- [License](#license)

## Background

### Centralized Training Limitations

Traditional machine learning requires the aggregation of training data on a single, central server. This approach is untenable in domains handling sensitive information, such as healthcare, due to significant legal and ethical constraints.

-   **Regulatory Compliance:** Data privacy regulations like the **Health Insurance Portability and Accountability Act (HIPAA)** in the United States and the **General Data Protection Regulation (GDPR)** in Europe impose strict rules on the storage, processing, and transfer of Personally Identifiable Information (PII) and Protected Health Information (PHI). Centralizing patient data from multiple institutions often violates these statutes.
-   **Data Sovereignty:** Data may be subject to the laws of the country in which it was collected, prohibiting it from leaving that jurisdiction.
-   **Competitive and Ethical Concerns:** Institutions are often unwilling to share proprietary data, as it represents a competitive asset and a potential liability if breached.

### Federated Learning (FL)

Federated Learning is a decentralized machine learning paradigm that enables model training on distributed data without requiring the data to be moved from its source. The process is iterative:

1.  A central server initializes a global model.
2.  The model is distributed to a subset of clients.
3.  Each client trains the model locally on its private data partition.
4.  Clients send their updated model parameters (e.g., weights and biases) back to the server.
5.  The server aggregates these updates (e.g., via federated averaging) to produce an improved global model for the next round.

This methodology ensures that raw training data never leaves the client's local environment.

### Homomorphic Encryption (HE)

While standard FL protects the raw data, the model updates themselves can potentially leak information about the underlying training data. Homomorphic Encryption is a cryptographic method that mitigates this risk.

-   **Definition:** HE allows for specific mathematical operations to be performed directly on encrypted data (ciphertexts). When the result of these computations is decrypted, it is identical to the result of performing the same operations on the original, unencrypted data (plaintexts).
-   **Application:** In this project, clients encrypt their model updates before sending them to the server. The server performs the federated averaging computation on these encrypted updates. Because the server only ever handles ciphertexts, it learns nothing about the individual contributions from any client, providing a strong guarantee of privacy for the model parameters.

## Technical Architecture

The system is composed of three primary components simulated within the application:

1.  **Clients:** Entities (e.g., hospitals) that hold private data. They receive the global model, train it locally, encrypt the resulting model weight updates using HE, and send the ciphertexts to the server.
2.  **Central Server:** An aggregation server that orchestrates the FL process. It selects clients, distributes the global model, and receives encrypted updates. It performs federated averaging on the ciphertexts and then decrypts the final averaged result to produce the next iteration of the global model.
3.  **Simulation & Analysis Dashboard:** A Streamlit-based web interface that provides a real-time animation of the FL process and presents a detailed analysis of the results, including comparisons of accuracy, performance, and privacy.

## Key Features

-   **Real-Time Simulation:** A step-by-step visual animation of the federated learning rounds, including client selection, training, encryption, and aggregation states.
-   **Privacy Visualization:** A "Privacy Sniffer" component that explicitly displays the data received by the server during aggregation, contrasting human-readable plaintext updates with unintelligible encrypted ciphertexts.
-   **Interactive Model Testing:** The final, securely trained models can be tested interactively:
    -   **MNIST:** A drawable canvas for real-time digit recognition.
    -   **Arrhythmia:** A diagnostic tool for classifying sample patient data, which includes model-explainability (XAI) outputs generated via the SHAP library.
-   **Performance Benchmarking:** The dashboard provides a comprehensive comparison of three training methodologies:
    1.  **Secure Federated Learning (SFL)**
    2.  **Plaintext Federated Learning** (insecure baseline)
    3.  **Centralized Training** (non-private theoretical performance upper-bound)

## Technology Stack

-   **Backend & ML Framework:** Python 3.11+, PyTorch
-   **Web Framework / Dashboard:** Streamlit
-   **Homomorphic Encryption:** [Microsoft SEAL](https://github.com/microsoft/SEAL) (via the [TenSEAL](https://github.com/OpenMined/TenSEAL) Python wrapper)
-   **Model Explainability:** SHAP
-   **Data Manipulation & Numerics:** Pandas, NumPy, scikit-learn
-   **Core Dependencies:** `joblib`, `matplotlib`, `seaborn`, `Pillow`, `st-shap`, `streamlit-drawable-canvas`

## Repository Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── assets/             # Static image assets for the UI
    ├── data/               # Default directory for datasets
    ├── centralized_trainer.py # Script to train the centralized baseline model
    ├── config.py           # Hyperparameter and model configurations
    ├── dashboard.py        # Main Streamlit application file
    ├── data_loader.py      # Data loading and partitioning logic
    ├── fl_logic.py         # Core federated training and aggregation functions
    ├── he_tenseal.py       # Homomorphic encryption/decryption logic using TenSEAL
    ├── main.py             # Main script to run FL simulations and generate results
    ├── models.py           # PyTorch model definitions
    ├── simulation.py       # High-level simulation orchestration
    └── utils.py            # Utility functions for evaluation and plotting
```

## Setup and Execution

### Prerequisites

-   Python (version 3.10+ recommended)
-   `pip` and `venv` for package management

### 1. Clone the Repository

```bash
git clone https://github.com/Realm07/SecureFL.git
cd SecureFL
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies

Install all required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

The Arrhythmia dataset must be manually downloaded and placed in the appropriate directory.

-   Download the dataset from a source such as the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/arrhythmia) or [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/heart-disease-dataset-from-uci).
-   Ensure the file is named `arrhythmia.csv`.
-   Place the `arrhythmia.csv` file inside the `src/data/` directory.

### 5. Generate Benchmark and Model Files

Before running the dashboard, you must first run the training scripts to generate the necessary result files (`.json`), model weights (`.pth`), and the data scaler (`.joblib`). Execute the following commands from the root directory:

```bash
# Generate centralized model benchmarks
python src/centralized_trainer.py --dataset mnist
python src/centralized_trainer.py --dataset arrhythmia

# Generate federated learning benchmarks and models
python src/main.py --dataset mnist
python src/main.py --dataset arrhythmia
```

### 6. Launch the Dashboard

Once all result files have been generated, launch the Streamlit application.

```bash
streamlit run src/dashboard.py
```

The application will be accessible at `http://localhost:8501`.

## License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.
