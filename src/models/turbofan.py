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

from typing import List, Dict, Tuple
from torch.utils.data import Dataset, random_split, DataLoader
from pandas import DataFrame
from torch import nn

import config

from . import FactoryModelDatasets

print("imported packages") 

logger_console = logging.getLogger(__name__)
logger_console.propagate = False

logger_loss = logging.getLogger(f"{__name__}.loss")
logger_loss.propagate = False

handler_stream_console = logging.StreamHandler(sys.stdout)
handler_file_console = logging.FileHandler('logs/console.log', mode='a')
handler_file_loss = logging.FileHandler('logs/loss.log', mode='a')

handler_stream_console.setLevel(logging.INFO)
handler_file_console.setLevel(logging.INFO)
handler_file_loss.setLevel(logging.INFO)

format_console = logging.Formatter('[%(levelname)s]: %(name)s : %(message)s')
format_file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler_stream_console.setFormatter(format_console)
handler_file_console.setFormatter(format_file)
handler_file_loss.setFormatter(format_file)

logger_console.addHandler(handler_stream_console)
logger_console.addHandler(handler_file_console)
logger_loss.addHandler(handler_stream_console)
logger_loss.addHandler(handler_file_loss)

with open("models/turbofan.yml") as f:
    config_turbofan = yaml.safe_load(f)

loss_function = torch.nn.MSELoss()

print("Finished all import stuff")

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
        # Development setma
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
        all_data_shortened = all_data_shortened.drop(columns = ["index"])
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


class CreatorCNNTurbofan(FactoryModelDatasets):


    @staticmethod
    def create_model_datasets():
        config_dataset = config_turbofan["dataset"]
        config_model = config_turbofan["models"][0]
        X_v_to_keep = config_dataset["X_v_to_keep"]
        X_s_to_keep = config_dataset["X_s_to_keep"]
        stepsize_sample = config_dataset["stepsize_sample"]
        considered_length = config_dataset["considered_length"]
        frequency = config_dataset["frequency"]
        seed = config_turbofan["seed"]
        validation_size = config_turbofan["validation_size"]
        np.random.seed(seed)
        torch.manual_seed(seed)

        df_turbofan, all_fc = read_in_data("data/raw/turbofan_simulation/data_set2/N-CMAPSS_DS02-006.h5", frequency, X_v_to_keep, X_s_to_keep, True, True)
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

        #FInd the minimum and maximum sensor measurement of the training set 
        #Make a dataframe with all training data 
        all_training_data = pd.DataFrame(columns = df_turbofan.columns)
       
        for unit in np.unique(df_turbofan.loc[:, "unit"]):
            data_engine = df_turbofan.loc[df_turbofan["unit"] == unit, :]
            flights = np.unique(data_engine["cycle"])
            nr_flights = len(flights)

            for flight in flights:
                if flight not in dict_training_flights[unit]:
                    continue
                data_flight = data_engine.loc[data_engine["cycle"] == flight]
                all_training_data = pd.concat([all_training_data, data_flight], ignore_index = True)
                        
        minimum, maximum = min_max_training(all_training_data) 
        del all_training_data 

        dataset_train = TurbofanSimulationDataset(
            df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_training_flights, minimum, maximum
        )
        dataset_valid = TurbofanSimulationDataset(
            df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_validation_flights, minimum, maximum
        )

        #Make the test data set 
        #Step 1: read in all test data (first False: we get the test data )
        df_turbofan_test, all_fc_test = read_in_data(
            "data/raw/turbofan_simulation/data_set2/N-CMAPSS_DS02-006.h5",
            frequency, X_v_to_keep, X_s_to_keep, False, True
        )
        df_turbofan_test = df_turbofan_test.drop(columns = ["hs"])
        test_units = np.unique(df_turbofan_test.loc[:, "unit"])

        #Step 2. Make a dictionary with all flights of all testing units 
        dict_test_flights = {} 
        for unit in test_units:
            last_flight = int(max(df_turbofan_test.loc[df_turbofan_test["unit"] == unit, "cycle"]))
            all_flights = list(range(1, last_flight + 1, 1))
            dict_test_flights[unit] = all_flights

        #Step 3. Make the test dataset  
        dataset_test = TurbofanSimulationDataset(
            df_turbofan_test, stepsize_sample, all_variables_x,
            considered_length, dict_test_flights, minimum, maximum
        )

        
        neural_client = CNNRUL(config_model, "Client")
        return (
            neural_client,
            {"train": dataset_train, "validation": dataset_valid, "test": dataset_test, "test_units" : dict_test_flights}
        )


class CreatorCNNEngine(FactoryModelDatasets):

    @staticmethod
    def nn_unit_create(): 
        config_model = config_turbofan["models"][0]
        nn_unit = CNNRUL(config_model, "Unit")
        return nn_unit

    @staticmethod
    def nn_server_create(split_layer): 
        config_model = config_turbofan["models"][0]
        config_model["split_layer"] = split_layer
        nn_server = CNNRUL(config_model, "Server")
        return nn_server

    @staticmethod
    def create_model_datasets(split_layer):
        config_dataset = config_turbofan["dataset"]
        config_model = config_turbofan["models"][0]
        config_model["split_layer"] = split_layer
        X_v_to_keep = config_dataset["X_v_to_keep"]
        X_s_to_keep = config_dataset["X_s_to_keep"]
        stepsize_sample = config_dataset["stepsize_sample"]
        considered_length = config_dataset["considered_length"]
        frequency = config_dataset["frequency"]
        seed = config_turbofan["seed"]
        validation_size = config_turbofan["validation_size"]
        ENGINE = int(os.getenv("ENGINE", "2.0"))
        np.random.seed(seed)
        torch.manual_seed(seed)

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

        #FInd the minimum and maximum sensor measurement of the training set 
        #Make a dataframe with all training data 
        all_training_data = pd.DataFrame(columns = df_turbofan.columns)
       
        for unit in np.unique(df_turbofan.loc[:, "unit"]):
            data_engine = df_turbofan.loc[df_turbofan["unit"] == unit, :]
            flights = np.unique(data_engine["cycle"])
            nr_flights = len(flights)

            for flight in flights:
                if flight not in dict_training_flights[unit]:
                    continue
                data_flight = data_engine.loc[data_engine["cycle"] == flight]
                all_training_data.append(data_flight)
                        
        minimum, maximum = min_max_training(all_training_data) 
        del all_training_data 

        dataset_train = EngineSimulationDataset(
            ENGINE, df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_training_flights, minimum, maximum
        )
        dataset_valid = EngineSimulationDataset(
            ENGINE, df_turbofan, stepsize_sample, all_variables_x,
            considered_length, dict_validation_flights, minimum, maximum 
        )


        neural_client = CNNRUL(config_model, "Client")
        return (
            neural_client,
            {"train": dataset_train, "validation": dataset_valid}
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
        minimum: float,
        maximum: float
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
        """

        self.all_data = all_data
        self.all_variables_x = all_variables_x
        self.considered_length = considered_length

        self.minimum = minimum 
        self.maximum = maximum 

        self.sample_index = []
        self.unit_flight_sample_indices = {}

        self._create_samples(considered_flights, stepsize_sample, considered_length)
        self._pre_processing()

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

    def _pre_processing(self):
        #minimum, maximum = min_max_training(self.all_data)
        self.all_data = normalization(self.all_data, self.minimum, self.maximum)

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


    def __init__(self, engine, all_data, *args, **kwargs):
        self.engine = engine
        self.flight_indices = {}
        all_data = all_data.loc[all_data["unit"] == engine, :].copy()
        super().__init__(all_data, *args, **kwargs)
        del self.unit_flight_sample_indices

    def _add_flight_indices(self, engine, flight, indices):
        self.flight_indices[flight] = indices

    def get_all_samples(self, flight):
        return range(*self.flight_indices[flight])


def train_one_epoch(
    neural_client, neural_server, loss_function,
    optimizer, data_loader, in_training
):
    running_loss = 0.
    num_batches = len(data_loader)
    for i, (sample_x, RUL) in enumerate(data_loader):
        logger_console.info(f"Batch {i}/{num_batches}")
        if in_training == False:
            with torch.no_grad():
                predicted = neural_client(sample_x)
                predicted = neural_server(predicted)
                predicted = predicted.squeeze(1)
                RUL = RUL.squeeze(1) #same shape in loss function
                loss = loss_function(RUL, predicted)
                running_loss = running_loss + loss.item()
        else:
            predicted = neural_client(sample_x)
            predicted = neural_server(predicted)
            predicted = predicted.squeeze(1)
            RUL = RUL.squeeze(1) #same shape in loss function
            loss = loss_function(RUL, predicted)
            running_loss = running_loss + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return running_loss


class CNNRUL(nn.Module):


    def __init__(
        self, cfg, location
    ):
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
                    ])
                else:
                    denses.extend([
                        nn.Linear(in_features=x[1], out_features=x[2], bias=True)
                    ])
            elif x[0] == "F":
                denses.extend([nn.Flatten()] )
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


def main_function(name_nn, name_loss_file, name_loss_graph, name_console):
    batch_size = config_turbofan["batch_size"]
    learning_rate = config_turbofan["learning_rate"]
    num_epochs = config_turbofan["num_epochs"]
    config_model = config_turbofan["models"][0]

    neural_client, training_partitions = CreatorCNNTurbofan.create_model_datasets()
    dataset_train, dataset_valid, dataset_test, test_units = training_partitions["train"], training_partitions["validation"], training_partitions["test"], training_partitions["test_units"] 
    del dataset_test, test_units
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_validation = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False)

    neural_server = CNNRUL(config_model, "Server")
    nn_path = "trained/" + name_nn +  ".pth"
    params = list(neural_server.parameters())
    params.extend(neural_client.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    best_validation_loss = None
    all_train_losses = []
    all_validation_losses = []

    for epoch in range(0, num_epochs, 1):
        start_time_epochs = time.time()
        logger_console.info(f"Epoch {epoch}/{num_epochs}")
        neural_client.train(True)
        neural_server.train(True)
        loss_train = train_one_epoch(
            neural_client, neural_server, loss_function, optimizer, dataloader_train, in_training=True
        )
        all_train_losses.append(loss_train)

        neural_client.train(False)
        neural_server.train(False)
        loss_validation = train_one_epoch(
            neural_client, neural_server, loss_function, optimizer, dataloader_validation, in_training=False
        )
        all_validation_losses.append(loss_validation)

        if (best_validation_loss == None) or (loss_validation < best_validation_loss):
            best_validation_loss = loss_validation
            logger_console.info(
                f"Update parameters. "
                f"Best validation loss: {best_validation_loss}"
            )
            torch.save(copy.deepcopy(neural_client.state_dict()), nn_path)
            #maybe we can also save the server model here? For the testing later on?
            #Or should we do that where wwe average the weights of the different clients? (I haven't been able to locate that spot in our code yet) 


        time_it_took =  time.time() - start_time_epochs
        logger_console.info(f"Epoch duration: {time_it_took}")

    logger_loss.info('Training loss')
    logger_loss.info(all_train_losses)
    logger_loss.info('Validation loss')
    logger_loss.info(all_validation_losses)


# def test_nn():
#     #something very similar as above: We get the test data, make the neural network, load the weigths, and do one partition
#     config_model = config_turbofan["models"][0]
    
#     #Get the test data 
#     neural_client, training_partitions = CreatorCNNTurbofan.create_model_datasets()
#     dataset_train, dataset_valid, dataset_test, test_units = training_partitions["train"], training_partitions["validation"], training_partitions["test"], training_partitions["test_untis"] 
#     del dataset_train, dataset_valid 

#     #make one neural network that is on the server (so without any clients, just all layers from input to output) (Not reaally sure ho =w to do this)
#     #Later on, I call this neural network "neural" 
#     #before: neural = CNNRUL(num layers etc.) 
#     #We can also make two neural networks, and then below, pass the input through both (under "with torch.no_grad()")
    

#     #Load the weights on this neural network 
#     #Before: neural.load_state_dict(torch.load(weights))
    
#     #Test the neural network :) 
#     MSE = 0 
#     MAE = 0 
#     points = 0 

#     for unit in test_units.keys():
#         print("\n the test unit is " , unit) 

#         MSE_unit = 0 
#         MAE_unit = 0 
#         last_flight = int(max(test_units.get(unit))) 

#         for flight in range(1, last_flight + 1,1):
#             print("the flight is " ,flight) 

#             mean_RUL_prediction = 0 

#             #Get alls amples belonging to this unit and flight
#             #Great that you kept this function :) 
#             all_samples = dataset_test.get_all_samples((unit, flight))
#             num_samples = len(all_samples)

#             #Get the RUL prediciton for each samples
#             for s in all_samples:
#                 data_sample = dataset_test.__getitem__(s)
#                 true_RUL = data_sample[1] 
#                 inp = data_sample[0] 
#                 inp = torch.tensor(inp) 
#                 with torch.no_grad():
#                     predicted = neural_network(inp)
#                     #we can also first pass it through the aggregated client neural network, and then through the server 
                                    
#                 mean_RUL_prediction = mean_RUL_prediction + predicted[0][0] #Let's see if it still takes this form!
            
#             #Get the prediction and the metrics
#             mean_RUL_prediction = mean_RUL_prediction / num_samples
#             error = mean_RUL_prediction - true_RUL 
#             MAE_unit = MAE_unit + abs(error)
#             MSE_unit = MSE_unit + error**2 

#         #Update the grand total
#         MAE = MAE + MAE_unit
#         MSE = MSE + MSE_unit
#         points = points + last_flight 

#         #Get the total MSE/MAE for the unit 
#         MAE_unit = MAE_unit / last_flight
#         RMSE_unit = math.sqrt(MSE_unit / last_flight)
#         print("For unit " , unit, " the MAE is " , MAE_unit, " and the RMSE is " , RMSE_unit)

#     #Get the grand total
#     MAE = MAE / points
#     RMSE = math.sqrt(MSE/points)
#     print("In total, the MAE is " , MAE, " and the RMSE is " , RMSE)


#main_function("cnn_weights", "Losses", "Graph", "console")