import torch
import numpy as np
import pandas as pd
import time
import h5py
import math
import copy
import os
import logging
import sys
import yaml
import json
import functools
import statistics

from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, random_split, DataLoader
from pandas import DataFrame
from torch import nn
from copy import deepcopy
from dataclasses import dataclass

from . import FactoryModelDatasets
from distributed_learning import utils


logger_console = logging.getLogger(__name__)

with open("models/turbofan.yml") as f:
    config_turbofan = yaml.safe_load(f)

loss_function = torch.nn.MSELoss()

def read_in_data(
    filename: str,
    frequency: int,
    X_v_to_keep: List[str],
    X_s_to_keep: List [str],
    training_data: bool = True,
    keep_all_data: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """Read data from a raw data h5 file.

    Args:
        frequency: by how many samples the original dataset is to be aggregated.
        X_v_to_keep: list of virutal sensor names to be kept from the dataset.
        X_s_to_keep: list of real sensor names to be kept from the dataset.
        training_data: boolean indicating whether the training or the testing
            dataset is to be read in.
        keep_all_data: boolean indicating whether to include only the healthy
            flights from the dataset.
    """

    with h5py.File(filename, 'r') as hdf:
        # Development set
        W_dev = np.array(hdf.get('W_dev'))             # W
        X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v
        A_dev = np.array(hdf.get('A_dev'))             # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))           # W
        X_s_test = np.array(hdf.get('X_s_test'))       # X_s
        X_v_test = np.array(hdf.get('X_v_test'))       # X_v
        A_test = np.array(hdf.get('A_test'))           # Auxiliary

        # column names
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_v_var = np.array(hdf.get('X_v_var'))
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

    # from np.array to list dtype U4/U5
    W_var = list(np.array(W_var, dtype='U20'))
    X_s_var = list(np.array(X_s_var, dtype='U20'))
    X_v_var = list(np.array(X_v_var, dtype='U20'))
    T_var = list(np.array(T_var, dtype='U20'))
    A_var = list(np.array(A_var, dtype='U20'))

    if training_data == True:
        df_X_s = DataFrame(data=X_s_dev, columns=X_s_var)
        df_X_v = DataFrame(data=X_v_dev, columns=X_v_var)
        df_A = DataFrame(data=A_dev, columns=A_var)
        df_W = DataFrame(data=W_dev, columns=W_var)
    else:
        df_X_s = DataFrame(data=X_s_test, columns=X_s_var)
        df_X_v = DataFrame(data=X_v_test, columns=X_v_var)
        df_A = DataFrame(data=A_test, columns=A_var)
        df_W = DataFrame(data=W_test, columns=W_var)

    df_X_s = df_X_s[X_s_to_keep]
    df_X_v = df_X_v[X_v_to_keep]

    df_X_s["unit"] = df_A["unit"].values
    df_X_s["cycle"] = df_A["cycle"].values
    df_X_s["hs"] = df_A["hs"].values

    all_data = [df_X_s, df_X_v, df_W]
    all_data = pd.concat(all_data, axis=1)

    if keep_all_data == False:
        all_data = all_data.loc[all_data["hs"] == 1]

    if frequency > 1:
        all_data_shortened = pd.DataFrame(columns = all_data.columns)
        for unit in np.unique(all_data["unit"]) :
             data_unit = all_data.loc[all_data["unit"] == unit]
             for flight in np.unique(data_unit["cycle"]):
                data_flight = data_unit.loc[data_unit["cycle"] == flight]
                data_flight.reset_index(inplace = True)

                means = (
                    data_flight
                    .groupby(
                        np.arange(len(data_flight)) // frequency
                    )
                    .mean()
                )
                all_data_shortened = pd.concat([all_data_shortened, means], axis = 0)
        del all_data
        all_data_shortened = all_data_shortened.drop(columns=["index"]).reset_index()
        return all_data_shortened, W_var

    return all_data, W_var


def min_max_training(training_data, skip = ["cycle", "unit" , "hs"]):
    minima = {}
    maxima = {}
    for column in training_data.columns:
        if column in skip:
            continue
        minimum = training_data[[column]].min()[column]
        maximum = training_data[[column]].max()[column]
        minima[column] = minimum
        maxima[column] = maximum
    return minima, maxima

def normalization(data, minima, maxima, skip = ["cycle", "unit" , "hs"]):
    for column in data.columns:
        if column in skip:
            continue
        # normalize between -1 and 1
        u = 1
        l = -1
        minimum = minima.get(column)
        maximum = maxima.get(column)

        data.loc[:, column] = (
            (data.loc[:, column] - minimum)*(u-l)/(maximum-minimum)+l
        )
    return data


class CreatorCNNTurbofanIsolated(FactoryModelDatasets):

    def __init__(self, model_config=None): 
        if not model_config: 
            model_config_path = "models/turbofan.yml"
            with open(model_config_path, "r") as f: 
                self.model_config = yaml.safe_load(f)
        else: 
            self.model_config = model_config

    def create_model_datasets(self, neural=None):
        config_turbofan = deepcopy(self.model_config)
        config_dataset = config_turbofan["dataset"]
        config_model = config_turbofan["models"][0]
        X_v_to_keep = config_dataset["X_v_to_keep"]
        X_s_to_keep = config_dataset["X_s_to_keep"]
        stepsize_sample = config_dataset["stepsize_sample"]
        considered_length = config_dataset["considered_length"]
        frequency = config_dataset["frequency"]
        validation_size = config_dataset["validation_size"]
        ENGINE = int(os.getenv("ENGINE", "2.0"))

        df_turbofan, all_fc = read_in_data(
            "data/raw/turbofan_simulation/data_set2/N-CMAPSS_DS02-006.h5",
            frequency, X_v_to_keep, X_s_to_keep, True, True
        )
        df_turbofan = df_turbofan.drop(columns = ["hs"])
        all_variables_x = X_v_to_keep + X_s_to_keep + all_fc
        units = np.unique(df_turbofan.loc[:, "unit"])

        dict_training_flights = {}
        dict_validation_flights = {}

        for unit in units:
            last_flight = int(max(df_turbofan.loc[df_turbofan["unit"] == unit, "cycle"]))
            all_flights = list(range(1, last_flight + 1, 1))
            validation_flights = set(np.random.choice(
                np.array(all_flights),
                size=math.floor(validation_size*len(all_flights)),
                replace=False,
            ))
            training_flights =  set(all_flights) - set(validation_flights)
            dict_training_flights[unit] = training_flights
            dict_validation_flights[unit] = validation_flights
        dataset_train = EngineSimulationDataset(
            ENGINE, df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_training_flights
        )
        train_minima, train_maxima = dataset_train.minima, dataset_train.maxima
        dataset_valid = EngineSimulationDataset(
            ENGINE, df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_validation_flights, 
            train_minima, train_maxima
        )
        dataset_total_train = TurbofanSimulationDataset(
            df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_training_flights
        )
        train_total_minima = dataset_total_train.minima
        train_total_maxima = dataset_total_train.maxima
        dataset_total_valid = TurbofanSimulationDataset(
            df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_validation_flights, 
            train_total_minima, train_total_maxima
        )

        df_turbofan_test, _ = read_in_data(
            "data/raw/turbofan_simulation/data_set2/N-CMAPSS_DS02-006.h5",
            frequency, X_v_to_keep, X_s_to_keep, False, True
        )
        df_turbofan_test = df_turbofan_test.drop(columns = ["hs"])
        test_units = np.unique(df_turbofan_test.loc[:, "unit"])

        dict_test_flights = {} 
        for unit in test_units:
            last_flight = int(max(df_turbofan_test.loc[df_turbofan_test["unit"] == unit, "cycle"]))
            all_flights = list(range(1, last_flight+1, 1))
            dict_test_flights[unit] = all_flights

        dataset_test = TurbofanSimulationDataset(
            df_turbofan_test, stepsize_sample, all_variables_x,
            considered_length, dict_test_flights, train_minima, train_maxima
        )

        neural = neural if neural != None else CNNRUL(config_model, "Unit")
        return (
            neural,
            {
                "train": dataset_train, 
                "validation": dataset_valid, 
                "validation_total": dataset_total_valid,
                "test": dataset_test,
            }
        )


class CreatorCNNTurbofan(FactoryModelDatasets):

    def __init__(self, model_config=None): 
        if not model_config: 
            model_config_path = "models/turbofan.yml"
            with open(model_config_path, "r") as f: 
                self.model_config = yaml.safe_load(f)
        else: 
            self.model_config = model_config

    def create_model_datasets(self, neural=None):
        config_turbofan = deepcopy(self.model_config)
        config_dataset = config_turbofan["dataset"]
        config_model = config_turbofan["models"][0]
        X_v_to_keep = config_dataset["X_v_to_keep"]
        X_s_to_keep = config_dataset["X_s_to_keep"]
        stepsize_sample = config_dataset["stepsize_sample"]
        considered_length = config_dataset["considered_length"]
        frequency = config_dataset["frequency"]
        validation_size = config_dataset["validation_size"]

        df_turbofan, all_fc = read_in_data(
            "data/raw/turbofan_simulation/data_set2/N-CMAPSS_DS02-006.h5",
            frequency, X_v_to_keep, X_s_to_keep, True, True
        )
        df_turbofan = df_turbofan.drop(columns = ["hs"])
        all_variables_x = X_v_to_keep + X_s_to_keep + all_fc
        units = np.unique(df_turbofan.loc[:, "unit"])

        dict_training_flights = {}
        dict_validation_flights = {}

        for unit in units:
            last_flight = int(max(df_turbofan.loc[df_turbofan["unit"] == unit, "cycle"]))
            all_flights = list(range(1, last_flight + 1, 1))
            validation_flights = set(np.random.choice(
                np.array(all_flights),
                size=math.floor(validation_size*len(all_flights)),
                replace=False,
            ))
            training_flights =  set(all_flights) - set(validation_flights)
            dict_training_flights[unit] = training_flights
            dict_validation_flights[unit] = validation_flights
        dataset_train = TurbofanSimulationDataset(
            df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_training_flights
        )
        train_minima, train_maxima = dataset_train.minima, dataset_train.maxima
        dataset_valid = TurbofanSimulationDataset(
            df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_validation_flights, 
            train_minima, train_maxima
        )

        df_turbofan_test, _ = read_in_data(
            "data/raw/turbofan_simulation/data_set2/N-CMAPSS_DS02-006.h5",
            frequency, X_v_to_keep, X_s_to_keep, False, True
        )
        df_turbofan_test = df_turbofan_test.drop(columns = ["hs"])
        test_units = np.unique(df_turbofan_test.loc[:, "unit"])

        dict_test_flights = {} 
        for unit in test_units:
            last_flight = int(max(df_turbofan_test.loc[df_turbofan_test["unit"] == unit, "cycle"]))
            all_flights = list(range(1, last_flight+1, 1))
            dict_test_flights[unit] = all_flights

        dataset_test = TurbofanSimulationDataset(
            df_turbofan_test, stepsize_sample, all_variables_x,
            considered_length, dict_test_flights, train_minima, train_maxima
        )

        neural = neural if neural != None else CNNRUL(config_model, "Unit")
        return (
            neural,
            {
                "train": dataset_train, 
                "validation": dataset_valid, 
                "test": dataset_test,
            }
        )


class CreatorCNNEngine(FactoryModelDatasets):

    def __init__(self, model_config=None, neural_network=None): 
        if not model_config: 
            model_config_path = "models/turbofan.yml"
            with open(model_config_path, "r") as f: 
                self.model_config = yaml.safe_load(f)
        else: 
            self.model_config = model_config
        self.nn_unit_create(neural_network)


    def nn_unit_create(self, neural_network): 
        config_model = config_turbofan["models"][0]
        if neural_network == None: 
            neural_network = CNNRUL(config_model, "Unit")
        self.neural_network = neural_network
        return self.neural_network

    def nn_server_create(self, split_layer): 
        config_turbofan = deepcopy(self.model_config)
        config_model = config_turbofan["models"][0]
        nn_server = CNNRUL(config_model, "Server")
        nn_server.load_state_dict(
            utils.split_weights_server(self.neural_network.state_dict(), nn_server.state_dict())
        )
        return nn_server

    def create_test_dataset(self): 
        pass

    def create_model_datasets(self, split_layer):
        config_turbofan = deepcopy(self.model_config)
        config_dataset = config_turbofan["dataset"]
        config_model = config_turbofan["models"][0]
        X_v_to_keep = config_dataset["X_v_to_keep"]
        X_s_to_keep = config_dataset["X_s_to_keep"]
        stepsize_sample = config_dataset["stepsize_sample"]
        considered_length = config_dataset["considered_length"]
        frequency = config_dataset["frequency"]
        validation_size = config_dataset["validation_size"]
        ENGINE = int(os.getenv("ENGINE", "2.0"))
        FAULTY = bool(int(os.getenv("FAULTY", "0")))
        FAULTY_CLIENT = json.loads(os.getenv("FAULTY_CLIENT", "[]"))
        NOISE_AMPLITUDE = float(os.getenv("NOISE_AMPLITUDE", "0"))
        faulty = FAULTY and (ENGINE in FAULTY_CLIENT)
        logger_console.info(f"Client engine: {ENGINE}")

        df_turbofan, all_fc = read_in_data(
            "data/raw/turbofan_simulation/data_set2/N-CMAPSS_DS02-006.h5",
            frequency, X_v_to_keep, X_s_to_keep, True, True
        )
        df_turbofan = df_turbofan.drop(columns = ["hs"])
        all_variables_x = X_v_to_keep + X_s_to_keep + all_fc

        dict_training_flights = {}
        dict_validation_flights = {}

        for unit in np.unique(df_turbofan.loc[:, "unit"]):
            last_flight = int(max(df_turbofan.loc[df_turbofan["unit"] == unit, "cycle"]))
            all_flights = list(range(1, last_flight + 1, 1))
            validation_flights = np.random.choice(
                np.array(all_flights), size=math.floor(validation_size * len(all_flights)), replace=False
            )
            training_flights =  list(set(all_flights) - set(validation_flights))
            dict_training_flights[unit] = training_flights
            dict_validation_flights[unit] = validation_flights


        dataset_train = EngineSimulationDataset(
            ENGINE, df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_training_flights, faulty=faulty,
            relative_noise=NOISE_AMPLITUDE,
        )
        train_minima, train_maxima = dataset_train.minima, dataset_train.maxima
        dataset_valid = EngineSimulationDataset(
            ENGINE, df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_validation_flights, 
            train_minima, train_maxima,
        )

        neural_client = CNNRUL(config_model, "Client")
        return (
            neural_client,
            {
                "train": dataset_train, 
                "validation": dataset_valid,
            }
        )


class TurbofanSimulationDataset(Dataset):
    """Turbofan Engine Simulation dataset class.

    Due to the size of the simulation, this class reduces the sampling
    granularity, and augments the resulting data. For a single row of the
    dataset, the input data is considered to be a fixed size sample of a flight
    for a unit, and the label is the RUL (in number of flights).
    """


    def __init__(
        self,
        all_data: pd.DataFrame,
        stepsize_sample: int,
        all_variables_x: List[str],
        considered_length: int,
        considered_flights: Dict[int, List[int]],
        dict_minima: Optional[dict] = None, 
        dict_maxima: Optional[dict] = None, 
    ):
        """Initialize the TurbofanSimulation Dataset class.

        Args:
            all_data: Dataframe with the flight samples at the
                rate of 1 sample/second
            stepsize_sample: size of the step between 2 consecutive sample
                streams. E.g., if a sample stream has a length of 50 samples,
                and the stepsize is of 10 samples, the first stream would
                consider the samples 0-49, and the second stream, the samples
                10-59.
            all_variables_x: A list of sensor identifiers (virtual +
                physical) which are to be considered. Having performed a
                statistical analysis as to which sensors correlate to the RUL,
                this parameter specifies the list of these sensors.
            considered_length: The number of samples to be taken into
                consideration in a single sample stream. E.g., if this value is
                50, then the first stream of a given flight will hold the first
                50 samples (0-49).
            considered_flights: Dictionary holding for each unit, the list of
                flights which are to be taken into consideration from the whole
                DataFrame in the initialized Dataset.
            dict_minima: dictionary of the minima for each column. 
            dict_maxima: dictionary of the maxima for each column.
        """

        self.all_data = all_data.copy()
        self.all_variables_x = all_variables_x
        self.considered_length = considered_length

        self.sample_index = []
        self.unit_flight_sample_indices = {}

        self._create_samples(considered_flights, stepsize_sample, considered_length)
        self._pre_processing(dict_minima, dict_maxima)

    def _create_samples(
        self, considered_flights: dict, stepsize_sample: int, considered_length: int
    ):
        for unit in np.unique(self.all_data["unit"]):
            data_engine = self.all_data.loc[self.all_data["unit"] == unit, :]
            flights = np.unique(data_engine["cycle"])
            nr_flights = len(flights)

            for flight in flights:
                if flight not in considered_flights[unit]:
                    continue
                data_flight = data_engine.loc[data_engine["cycle"] == flight]
                nr_samples = data_flight.shape[0]
                nr_streams = (nr_samples - considered_length)//stepsize_sample + 1
                for i in range(nr_streams):
                    self.sample_index.append((
                        data_flight.index[0]+i*stepsize_sample,
                        nr_flights - flight,
                    ))
                self._add_flight_indices(
                    unit, flight,
                    [len(self.sample_index)-nr_streams, len(self.sample_index)]
                )
    def _pre_processing(self, minima, maxima):
        if (not minima) or (not maxima):
            minima, maxima = min_max_training(self.all_data)
        self.__minima, self.__maxima = minima, maxima
        self.all_data = normalization(self.all_data, self.__minima, self.__maxima)

    def _add_flight_indices(self, engine, flight, indices):
        if engine not in self.unit_flight_sample_indices:
            self.unit_flight_sample_indices[engine] = {}
        self.unit_flight_sample_indices[engine][flight] = indices

    def __len__(self):
        """Length of the initialized Dataset.

        This dunder function is called by the Dataloader class to get the size
        of the dataset.
        """

        return len(self.sample_index)

    @property
    def minima(self): 
        return self.__minima

    @property
    def maxima(self): 
        return self.__maxima

    def get_all_samples(self, sample):
        """Get all measurement streams for a (unit, flight) tuple."""

        u, f = sample
        return range(*self.unit_flight_sample_indices[u][f])

    def __getitem__(self, idx):
        """Get a single entry from the dataset.

        Given `idx`, this function should return the entry at the position
        specified by `idx`.
        """

        start, RUL = self.sample_index[idx]
        sample_x = self.all_data.loc[
            (start):(start + self.considered_length-1),
            self.all_variables_x
        ]
        sample_x = sample_x.to_numpy()
        sample_x = np.float32(sample_x)
        return (
            torch.from_numpy(sample_x).unsqueeze(0),
            torch.from_numpy(np.array(np.float32(RUL))).unsqueeze(0)
        )


class EngineSimulationDataset(TurbofanSimulationDataset):


    def __init__(self, engine, all_data, *args, faulty=False, relative_noise=0.5, **kwargs):
        self.engine = engine
        all_data = all_data.loc[all_data["unit"] == engine, :].copy()
        if faulty == True:
            all_data = self.add_noise(all_data, relative_noise)
        super().__init__(all_data, *args, **kwargs)

    def add_noise(self, df, relative_noise, ignore=set(["unit", "cycle"])): 
        logger_console.info(
            f"Adding noise to engine data. std: `{relative_noise}`"
        )
        col_std = df.std()
        assert isinstance(col_std, pd.Series)
        for idx, std_dev in col_std.items(): 
            if idx in ignore: 
                continue
            nrows = len(df.loc[:, idx])
            df.loc[:, idx] += np.random.normal(
                0, std_dev*relative_noise, nrows
            )
        return df


def validate(neural, dataloader_validation): 
    import tqdm
    loss_sum = 0
    len_dataset = len(dataloader_validation.dataset)
    with torch.no_grad(): 
        for inputs, targets in tqdm.tqdm(
            dataloader_validation,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'
        ): 
            outputs = neural(inputs)
            loss_sum += torch.sum((targets-outputs)**2).item()
    MSE = loss_sum/len_dataset
    RMSE = math.sqrt(MSE)
    logger_console.info(f"Validate RMSE: {RMSE}\tMSE: {MSE}")
    return RMSE

def train_one_epoch(
    neural, dataloader_train, optimizer, loss_criterion
): 
    import tqdm
    for (inputs, targets) in tqdm.tqdm(
        dataloader_train, 
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}'
    ):
        optimizer.zero_grad()
        outputs = neural(inputs)
        loss = loss_criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def sum_squared_error(error): 
    return torch.sum(error**2).item()

def sum_absolute_error(error):
    return torch.sum(torch.abs(error)).item()

def propagate_flight_samples(neural, dataset_test, indices): 
    outputs, targets = torch.tensor([]), torch.tensor([])
    neural.eval()
    with torch.no_grad():
        for idx in indices: 
            entry, target = dataset_test[idx]
            output = neural(entry)
            outputs = torch.cat((output, outputs), 0)
            targets = torch.cat((target, targets), 0)
        return outputs, targets

def test(neural, dataset_test):
    sum_se = sum_ae = 0 
    len_dataset = len(dataset_test)
    for unit, flights in dataset_test.unit_flight_sample_indices.items():
        for flight in flights: 
            indices = dataset_test.get_all_samples((unit, flight))
            outputs, targets = propagate_flight_samples(neural, dataset_test, indices)
            err = targets - outputs
            sum_ae += sum_absolute_error(err)
            sum_se += sum_squared_error(err)
    mae = sum_ae/len_dataset
    mse = sum_se/len_dataset
    rmse = math.sqrt(mse)
    logger_console.info(f"Test RMSE: {rmse}\tMAE: {mae}")

def test_per_flight(neural, dataset_test, filepath): 
    import tqdm
    sum_se = sum_ae = 0 
    len_dataset = 0
    dict_engine_ruls = {}
    for unit, flights in dataset_test.unit_flight_sample_indices.items():
        logger_console.info(f"Unit {unit}")
        dict_engine_ruls[unit] = []
        for flight in tqdm.tqdm(
            flights,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}',
        ): 
            indices = dataset_test.get_all_samples((unit, flight))
            outputs, targets = propagate_flight_samples(neural, dataset_test, indices)
            if not torch.numel(targets): 
                continue
            target = targets[0]
            output = torch.median(outputs)
            dict_engine_ruls[unit].append({
                "RUL": target.item(),
                "predicted": output.item(),
                "average": torch.mean(outputs).item(), 
                "std_dev": torch.std(outputs).item(),
            })
            err = target - output
            sum_ae += sum_absolute_error(err)
            sum_se += sum_squared_error(err)
            len_dataset += 1
    with open(filepath, "w") as f: 
        json.dump(dict_engine_ruls, f)
    mae = sum_ae/len_dataset
    mse = sum_se/len_dataset
    rmse = math.sqrt(mse)
    logger_console.info(f"Test RMSE: {rmse}\tMAE: {mae}")
    return rmse, mae

def compute_rmse_mae(outputs, targets): 
    rmses = []
    maes = []
    for thread_outputs, thread_targets in zip(outputs, targets):
        err = thread_outputs - thread_targets
        sum_se = sum_squared_error(err)
        sum_ae = sum_absolute_error(err)
        rmses.append(math.sqrt(sum_se/thread_outputs.size()[0]))
        maes.append(sum_ae/thread_outputs.size()[0])

    rmse_max, rmse_min = max(rmses), min(rmses)
    rmses_filtered = [rmse for rmse in rmses if (rmse != rmse_max) and (rmse != rmse_min)]
    rmse, mae = sum(rmses_filtered)/len(rmses_filtered), sum(maes)/len(maes)
    return rmse, mae


class CNNRUL(nn.Module):


    def __init__(self, cfg, location):
        super(CNNRUL, self).__init__()

        kernel_size = cfg["kernel_size"]
        self.kernel_size = (kernel_size.get("height"), kernel_size.get("width"))
        self.stride = 1

        self.split_layer = cfg["split_layer"]
        self.location = location

        self.features, self.denses = self._make_layers(cfg["layers"])
        self._initialize_weights()


    def forward(self, sample_x):
        out = self.features(sample_x) if len(self.features) > 0 else sample_x
        out = self.denses(out) if len(self.denses) > 0 else out
        return out

    def _make_layers(self, cfg):
        features = []
        denses = []

        if self.location == "Server":
            cfg = cfg[(self.split_layer+1):]
        elif self.location == "Client":
            cfg = cfg[0:(self.split_layer+1)]
        elif self.location == "Unit":
            pass

        for x in cfg: #For all considered tuples
            if x[0] == "C":
                features.extend([
                    nn.Conv2d(
                        in_channels=x[1], out_channels=x[2],
                        kernel_size=self.kernel_size, stride=self.stride,
                        padding="same"
                    ),
                    nn.ReLU(inplace = True)])
            elif x[0] == "L":
                if x[3] == True: #Do we need the ReLU activation function? (True = yes, False = no)
                    denses.extend([
                        nn.Linear(
                            in_features=x[1], out_features=x[2], bias=True
                        ),
                        nn.ReLU(inplace = True),
                        nn.Dropout(0.5)
                    ])
                else:
                    denses.extend([
                        nn.Linear(in_features=x[1], out_features=x[2], bias=True)
                    ])
            elif x[0] == "F":
                denses.extend([nn.Flatten()])
            else:
                logger_console.error(
                    f"Error: X[0] does not equal F, L or C, but equals {x[0]}"
                )
                raise ValueError("Wrong value for the layer type x[0]")
        return nn.Sequential(*features), nn.Sequential(*denses)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity = "relu")
                assert m.bias != None # mypy
                nn.init.constant_(m.bias, 0)


class UncomparableModels(Exception): 
    pass

class NotCNNRULInstance(Exception): 
    pass


@dataclass
class FileCNNRULStruct: 
    """Struct to serialize and deserialize a CNNRUL model.

    Attributes: 
        model_state_dict: pytorch state dict to be used to load the CNNRUL
            model.
        model_config_context: provides the configuration context used to create
            the model and the dataset. Used to compare between 2 different
            CNNRUL models.
        model_config_runtime: informative attribute, that indicates the runtime
            context in which the model was created. This enables
            reproducibility.
        validation_results: provides the validation results for the model using
            the this instance's `model_state_dict`. This attribute is used to
            compare validation results with other comparable models.
    """


    model_state_dict: dict
    model_config_context: dict
    model_config_runtime: dict
    validation_results: float


def improved_validation_cnnrul(
        model: FileCNNRULStruct, other: FileCNNRULStruct
) -> bool: 
    """Return whether `other` has better validation results that `model`."""

    if model.validation_results > other.validation_results: 
        return True
    return False
    
def equivalent_config_cnnrul(model_config, other_config): 
    """Indicates whether 2 rul cnn models have equivalent configurations."""
    if model_config == other_config: 
        return True
    return False

def model_recreate_cnnrul(
    file_model_struct: FileCNNRULStruct, config_file_dict: dict
) -> CNNRUL: 
    if not isinstance(file_model_struct, FileCNNRULStruct): 
        raise NotCNNRULInstance()
    neural = CNNRUL(config_file_dict["models"][0], "Unit")
    neural.load_state_dict(file_model_struct.model_state_dict)
    return neural
