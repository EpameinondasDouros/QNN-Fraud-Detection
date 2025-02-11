from itertools import combinations
from ..imports import pd, np, tf, plt, sns, qml, Sequential, Dense, Dropout
from planqk import PlanqkQuantumProvider
n_qubits = 3

def initialize_device(n_qubits, device_type="default"):
    """Initialize the quantum device based on the backend type."""
    
    if device_type == "simulator":
        provider = PlanqkQuantumProvider(access_token="your_token")
        backend = provider.get_backend("azure.ionq.simulator")
        dev = qml.device("qiskit.remote", wires=n_qubits+1, backend=backend, shots=100)

    elif device_type == "hardware":
        provider = PlanqkQuantumProvider(access_token="your_token")
        backend = provider.get_backend("azure.ionq.simulator")
        dev = qml.device("qiskit.remote", wires=n_qubits+1, backend=backend, shots=100)

    else:  # Default to PennyLane simulator
        dev = qml.device("default.qubit", wires=n_qubits+1)

    print(f"Quantum device initialized: {dev}")
    return dev


def custom_layer(weights, n_qubits):
    index = 0  # Initialize index to track unique weights

    for i in range(n_qubits + 1):
        qml.RY(weights[index], wires=i)
        index += 1  # Increment index

    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for pair in pairs:
        qml.CNOT(wires=pair)

    for i in range(n_qubits + 1):
        qml.RY(weights[index], wires=i)
        index += 1  # Increment index

    pairs = [(0, 2), (2, 0)]
    for pair in pairs:
        qml.CNOT(wires=pair)

    qml.RY(weights[index], wires=3)
    index += 1  
    qml.RY(weights[index], wires=3)
    index += 1  

    for j in range(2):
        for i in range(n_qubits):
            qml.RY(weights[index], wires=i)
            index += 1  # Increment index

    # Apply third set of CNOT gates
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for pair in pairs:
        qml.CNOT(wires=pair)

    # Apply final set of RZ gates
    for i in range(n_qubits + 1):
        qml.RZ(weights[index], wires=i)
        index += 1  # Increment index

def custom_layer_long(weights, n_qubits):
    index = 0  # Start index for weights

    # First block of RY
    for i in range(n_qubits + 1):
        qml.RY(weights[index], wires=i)
        index += 1

    # First set of CNOT pairs
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for pair in pairs:
        qml.CNOT(wires=pair)

    # Second block of RY
    for i in range(n_qubits + 1):
        qml.RY(weights[index], wires=i)
        index += 1

    # Second set of CNOT pairs
    pairs = [(0, 2), (2, 0)]
    for pair in pairs:
        qml.CNOT(wires=pair)

    # Third block of RY (single qubit repeated)
    qml.RY(weights[index], wires=3)
    index += 1
    qml.RY(weights[index], wires=3)
    index += 1

    # Nested loop of RY
    for j in range(2):
        for i in range(n_qubits):
            qml.RY(weights[index], wires=i)
            index += 1

    # Third set of CNOT pairs
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for pair in pairs:
        qml.CNOT(wires=pair)

    # First block of RZ
    for i in range(n_qubits + 1):
        qml.RZ(weights[index], wires=i)
        index += 1

    # Fourth set of CNOT pairs
    pairs = [(0, 2), (2, 0)]
    for pair in pairs:
        qml.CNOT(wires=pair)

    # Fourth block of RY (single qubit repeated)
    qml.RY(weights[index], wires=3)
    index += 1
    qml.RY(weights[index], wires=3)
    index += 1

    # Second block of RZ
    for i in range(n_qubits):
        qml.RZ(weights[index], wires=i)
        index += 1

    # Third block of RY
    for i in range(n_qubits):
        qml.RY(weights[index], wires=i)
        index += 1

    # Fifth set of CNOT pairs
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for pair in pairs:
        qml.CNOT(wires=pair)

    # Final block of RZ
    for i in range(n_qubits + 1):
        qml.RZ(weights[index], wires=i)
        index += 1

    return index  # Total number of indices used

def create_qnode_long(dev):
    
    @qml.qnode(dev)
    def qnode_long(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits+1))

        for w in weights:
            custom_layer_long(w,n_qubits)
        outputs = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        return outputs

    return qnode_long

def create_qnode(dev):
    @qml.qnode(dev)
    def qnode(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits+1))
        for w in weights:
            custom_layer(w,n_qubits)
        outputs = [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        return outputs
    
    return qnode



def create_qlayer(X_train, n_qubits):
    n_layers = 1
    n_qubits=3
    total_weights = 3 * (n_qubits + 1) + 2 * n_qubits + 2
    weight_shapes = {"weights": (n_layers, total_weights+1)}
    weights = np.random.random(size=(n_layers, total_weights))
    fig, ax = qml.draw_mpl(qnode)(X_train[:, :4], weights)
    plt.show()
    qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

    return qlayer

def create_qlayer_long(n_qubits=3,runtype="default"):

    dev = initialize_device(n_qubits, device_type=runtype)

    n_layers = 1
    total_weights_long = 32
    print("Total weights required:", total_weights_long)
    weight_shapes_long = {"weights": (n_layers, total_weights_long+1)}


    weights = np.random.random(size=(n_layers, total_weights_long))

    qnode_long = create_qnode_long(dev)

    qlayer_long = qml.qnn.KerasLayer(qnode_long, weight_shapes_long, output_dim=n_qubits)

    return qlayer_long