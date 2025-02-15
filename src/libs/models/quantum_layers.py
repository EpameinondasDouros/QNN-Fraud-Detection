from asyncio.log import logger
from itertools import combinations
from ..imports import pd, np, tf, plt, sns, qml, Sequential, Dense, Dropout

n_qubits = 3

from qiskit_ibm_runtime import QiskitRuntimeService

# Save your IBM Quantum account
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="Input-IBM-API-Token",
    overwrite=True
)

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-deps"])

def initialize_device(n_qubits, device_type="default",device_name="ibm_brisbane"):

    install("planqk-quantum==2.15.0")

    from planqk import PlanqkQuantumProvider

    """Initialize the quantum device, handling authentication errors properly."""
    logger.info("Initializing quantum device")

    if device_type == "simulator":
        try:
            # Initialize the Planqk provider with your access token
            provider = PlanqkQuantumProvider(access_token="Input-Planqk-API-Token")
            
            # Retrieve and print available backends
            available_backends = provider.backends()
            print(f"Available backends: {available_backends}")
            if not available_backends:
                raise ValueError("PlanQK authentication failed: No backends available.")
            
            # Select the backend (ensure the name matches one of the available backends)
            backend_name = "azure.ionq.simulator"
            backend = provider.get_backend(backend_name)
            if backend is None:
                raise ValueError(f"PlanQK backend '{backend_name}' not available.")

            # Instantiate QiskitRuntimeService and print the active account info
            service = QiskitRuntimeService()
            print("Active Qiskit account:", service.active_account())

            # Create a PennyLane device using the Qiskit remote backend
            dev = qml.device("qiskit.remote", wires=n_qubits + 1, backend=backend, shots=100)
            print("PennyLane device created:", dev)
            
        except Exception as e:
            print("An error occurred:", e)

    elif device_type=="hardware":
        
        service = QiskitRuntimeService(
            channel="ibm_quantum",
            token="Your_IBMQ_API_Token"
        )
        print(service.backends())

        backend = service.backend("ibm_brisbane")

        dev = qml.device("qiskit.remote", wires=127, backend=backend,shots=1000)
    
    else:
        dev = qml.device("default.qubit", wires=n_qubits+1)
    print(f"✅ Quantum device initialized: {dev}")
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

def create_qlayer_long(n_qubits=3,runtype="default",device_name="ibm_brisbane"):

    dev = initialize_device(n_qubits, device_type=runtype,device_name=device_name)

    n_layers = 1
    total_weights_long = 32
    print("Total weights required:", total_weights_long)
    weight_shapes_long = {"weights": (n_layers, total_weights_long+1)}


    weights = np.random.random(size=(n_layers, total_weights_long))

    qnode_long = create_qnode_long(dev)

    qlayer_long = qml.qnn.KerasLayer(qnode_long, weight_shapes_long, output_dim=n_qubits)

    return qlayer_long