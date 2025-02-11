"""
Template for implementing services running on the PlanQK platform
"""
import time
from loguru import logger
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
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
    if params and "type" in params:
        run_type = params["type"]
        logger.info(f"Run type: {run_type}")
    else:
        run_type = "default"
        logger.warning("No 'type' parameter provided. Using default run type.")
    

##############################################################################################################
    print("Starting the pipeline")
 
    start_time = time.time()
 
    results, testing = run_pipeline(finance_df, run_type=run_type)

    exec_time = time.time() - start_time

    logger.info("Printing the results")

    print(results)

    if testing:
        result = {
            "Testing Results": results # Keep as list if results is already a NumPy array or list
        }
        metadata = {
            "execution_time": exec_time,
            "tools": "Pennylane",
        }
    else:
        result ={
            "Predictions": results 
        }

        metadata = {
            "execution_time": exec_time,
            "tools": "Pennylane",
        }
    
    return ResultResponse(result=result, metadata=metadata)
