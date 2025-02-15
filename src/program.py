"""
Template for implementing services running on the PlanQK platform
"""
import qiskit
import tensorflow as tf
import time
from loguru import logger
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
import qiskit_ibm_runtime as qir
from typing import Dict, Any, Union
from .pipeline import run_pipeline
from .libs.return_objects import ResultResponse, ErrorResponse



def run(data: Dict[str, Any] = None, params: Dict[str, Any] = None) -> Union[ResultResponse, ErrorResponse]:
    """
    Default entry point of your code. Start coding here!

    Parameters:
        data (Dict[str, Any]): The input data sent by the client
        params (Dict[str, Any]): Contains parameters, which can be set by the client to configure the execution

    Returns:
        response: (ResultResponse | ErrorResponse): Response as arbitrary json-serializable dict or an error to be passed back to the client
    """

    print("IN THE RUN FUNCTION")
    # logger.info(f"Received data: {data}")

    finance_df = pd.DataFrame()

    if isinstance(data, dict):

        if "transactions" in data and isinstance(data["transactions"], list):
            finance_df = pd.DataFrame(data["transactions"])
            # logger.info(f"Multiple transactions detected. DataFrame created with {finance_df.shape[0]} rows and {finance_df.shape[1]} columns.")
        else:
            logger.warning("Empty or invalid JSON object received. Creating an empty DataFrame.")

    else:
        logger.warning("Invalid data format. Expected a JSON object.")

    # logger.info(f"DataFrame preview:\n{finance_df.head()}")
    
    # Retrieve the 'type' parameter from params
    # Retrieve the 'device' parameter from params
    if params and "device" in params:
        run_type = params["device"]
        logger.info(f"Run type: {run_type}")
    else:
        run_type = "default"
        logger.warning("No 'device' parameter provided. Using default run type.")

    # Retrieve the 'hardware' parameter from params
    if params and "hardware" in params:
        device_name = params["hardware"]
        logger.info(f"Device name: {device_name}")
    else:
        device_name = "ibm_brisbane"
        logger.warning("No 'hardware' parameter provided. Using default device name.")

    
    

##############################################################################################################
    print("Starting the pipeline")

    print("PennyLane version:", qml.__version__)
    print("Qiskit version:", qiskit.__version__)
    print("TensorFlow version:", tf.__version__)
    print("qiskit-ibm-runtime version:", qir.__version__)

 
    start_time = time.time()
 
    results, testing = run_pipeline(finance_df, run_type=run_type,device_name=device_name)
    exec_time = time.time() - start_time

    logger.info("Printing the results")

    print(results)

    if testing:
        result = {
            "Testing Results": results # Keep as list if results is already a NumPy array or list
        }
        metadata = {
            "execution_time": exec_time,
            "tools": f"Pennylane, Using the device: {run_type}",
        }
    else:
        result ={
            "Predictions": results 
        }

        metadata = {
            "execution_time": exec_time,
            "tools": f"Pennylane,Using the device: {run_type}",
        }
    
    return ResultResponse(result=result, metadata=metadata)
