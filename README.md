# QFL-Fraud-Detection

## Overview
This project was developed as part of the *Qinnovision World Challenge 2025*, specifically for the *Quantum Federated Learning for Fraud Detection* challenge. Our approach integrates quantum computing techniques to enhance fraud detection models, particularly focusing on improving recall—ensuring fewer fraudulent transactions go undetected.

## Methodology
We implemented a **Quantum Neural Network (QNN)** with an **Intermediate Quantum Layer (IQL)** to boost the performance of traditional machine learning models in fraud detection. The core idea was to integrate quantum-enhanced representations within a classical Federated Learning (FL) framework, thereby leveraging quantum advantages in feature learning and model optimization.

### Key Contributions
- **Quantum Neural Network (QNN)**: A hybrid quantum-classical model that utilizes quantum circuits as part of a deep learning architecture.
- **Intermediate Quantum Layer (IQL)**: A dedicated quantum processing unit within the network to transform classical feature representations into an enhanced quantum-embedded space.
- **Improved Recall in Fraud Detection**: By integrating quantum representations, our model achieves better fraud recall rates compared to classical deep learning approaches.
- **Federated Learning Integration**: Our model is designed to function in a federated learning environment, ensuring privacy and decentralization in fraud detection applications.

## Implementation Details
### 1. Data Processing
- The dataset used consists of transactional fraud detection records, preprocessed for compatibility with both classical and quantum models.
- Feature engineering included both classical transformations and quantum feature embedding.

### 2. Model Architecture
Our model consists of three main components:
1. **Classical Input Layer**: Processes raw transactional data into feature vectors.
2. **Intermediate Quantum Layer (IQL)**: A parameterized quantum circuit (PQC) implemented using IBM Qiskit and Pennylane, introducing quantum entanglement into the model's feature transformation.
3. **Classical Output Layer**: A standard neural network classifier trained using Federated Learning principles, improving both accuracy and recall.

### 3. Training Strategy
- **Federated Learning Framework**: The QNN model was trained across multiple clients using federated learning techniques to preserve data privacy.
- **Quantum Circuit Optimization**: The quantum layer parameters were optimized using hybrid quantum-classical gradient descent methods.
- **Performance Metrics**: Standard fraud detection metrics such as Precision, Recall, and F1-score were used, with a particular emphasis on maximizing Recall.

## Results
- **Increased Recall**: Our approach improved recall rates by **X%** compared to baseline classical models.
- **Better Feature Representation**: The Intermediate Quantum Layer contributed to a more effective separation of fraudulent and non-fraudulent transactions.
- **Scalability in Federated Learning**: The model demonstrated efficiency in a federated environment, proving its potential for real-world deployment.

## Future Work
- **Enhancing Quantum Layer Depth**: Exploring more complex parameterized quantum circuits to further improve accuracy.
- **Testing on Real-World Data**: Extending the framework to large-scale real-world fraud datasets.
- **Optimizing Federated Quantum Learning**: Investigating ways to reduce computational overhead and improve scalability in federated quantum machine learning applications.

## Technologies Used
- **Quantum Computing**: Pennylane, IBM Quantum Platform
- **Federated Learning**: Blockchain QHash, QKD
- **Machine Learning**: Tensorflow
- **Deployment**: IBM Cloud Quantum Services

## Team
Developed by *The Basement* as part of the *Qinnovision World Challenge 2025*.

---
This project represents a significant step towards integrating quantum computing into real-world fraud detection applications, demonstrating the potential for Quantum Federated Learning to revolutionize financial security systems.
