import sys
import os
import torch
import numpy as np
import logging
import yaml
import json

LOG_LEVEL = os.getenv("LOG_LEVEL") or logging.INFO
PROGRAM_NAME = os.getenv("PROGRAM_NAME") 
if PROGRAM_NAME == None: 
    raise Exception()
FAULTY = bool(int(os.getenv("FAULTY", "0"))) if PROGRAM_NAME != "rul_turbofan" else False
FAULTY_CLIENT = json.loads(os.getenv("FAULTY_CLIENT", "[]")) if FAULTY else []
NOISE_AMPLITUDE = float(os.getenv("NOISE_AMPLITUDE", "0"))
NCLIENTS = int(os.getenv("NCLIENTS", "1"))
ENGINE = int(os.getenv("ENGINE", "0"))

logging.basicConfig(
    format='%(asctime)s [%(levelname)s]:%(name)s:%(threadName)s: %(message)s',
    level=LOG_LEVEL,
)

SERVER_ADDR= 'fedadapt_server'
SERVER_PORT = 51000

home = '/usr/src/app'

runtime_config_file_path = os.path.join(home, "config.yml")
with open(runtime_config_file_path, "r") as f: 
    runtime_config = yaml.safe_load(f)

model_config_path = os.path.join(home, "models/turbofan.yml")
with open(model_config_path, "r") as f: 
    model_config = yaml.safe_load(f)

frequency = model_config["dataset"]["frequency"]
dir_frequency = f"frequency={frequency}/"
dir_faulty_client = f"faulty_client={FAULTY_CLIENT}/" if FAULTY else ""
dir_noise = f"noise_amplitude={NOISE_AMPLITUDE}/" if FAULTY else ""
dir_engine = f"engine={ENGINE}" if PROGRAM_NAME == "rul_turbofan_isolated" else ""
dir_program = f"program={PROGRAM_NAME}/"

results_dir = os.path.join(home, "results")
evaluation_directory = os.path.join(
    results_dir, runtime_config["evaluation_directory"],
    dir_frequency + dir_faulty_client + dir_noise + dir_program + dir_engine
)
if not os.path.isdir(evaluation_directory): 
    os.makedirs(evaluation_directory)

split_layer = 2
R = runtime_config["epochs"]
LR = runtime_config["learning_rate"]
B = runtime_config["batch_size"]

seed = runtime_config["seed"]
np.random.seed(seed)
torch.manual_seed(seed)
