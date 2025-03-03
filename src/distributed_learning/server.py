import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import threading
import tqdm
import functools
import numpy as np
import socket
import time
import random
import logging
import statistics

from typing import (
    List, Dict, Type, Iterable, Dict, Any, OrderedDict, Optional, Union
)
from functools import partial
from dataclasses import dataclass

from .communicator import Communicator
from . import utils


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
tqdm.tqdm = partial(tqdm.tqdm, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')

np.random.seed(0)
torch.manual_seed(0)


class InitSplitFedServerException(Exception): 
    pass


class ThreadValidationException(Exception): 
    pass


class ValidateModelState:


    unit_state_dict: OrderedDict
    _validation_result: Optional[float]

    def __init__(self, unit_state_dict: OrderedDict): 
        self.unit_state_dict = unit_state_dict
        self._validation_result = None

    @property
    def validation_result(self): 
        if self._validation_result == None:
            raise Exception()
        return self._validation_result

    @validation_result.setter
    def validation_result(self, value): 
        self._validation_result = value


class ValidatedModel: 


    unit_state_dict: OrderedDict 
    validation_result: float

    def __init__(self, unit_state_dict, validation_result): 
        self.unit_state_dict = unit_state_dict
        self.validation_result = validation_result

    def is_better(self, other): 
        if not isinstance(other, ValidatedModel): 
            raise Exception()
        return self.validation_result <= other.validation_result


class CollectionValidateModelState:


    _validate_models: Dict[int, ValidateModelState]

    def __init__(self):
        self._validate_models = {}
        self._populated = False
        self._cached_results = None

    def add_model(self, model_state_dict: OrderedDict, client_index): 
        self._validate_models[client_index] = ValidateModelState(model_state_dict)

    def models_to_validate(self) -> Dict[int, OrderedDict]: 
        return {
            client_index: validate_model.unit_state_dict 
            for client_index, validate_model in self._validate_models.items()
        }
    
    @property
    def validation_result(self): 
        if self._populated == False: 
            raise Exception()
        return self._validate_models

    @validation_result.setter
    def validation_result(self, value: Dict[int, float]): 
        if self._populated == True: 
            raise Exception()
        self._populated = True
        for client_index, result in value.items(): 
            self._validate_models[client_index].validation_result = result


class BestModelStateValidation: 
    
    validate_model_state: Optional[ValidateModelState]
    validating_client: Optional[int]
    original_client: Optional[int]

    def __init__(self):
        self.validate_model_state = None
        self.validating_client = None
        self.original_client = None
        self._populated = False

    def new_best(
        self, validate_model_state: ValidateModelState, validating_client: int, 
        original_client: int,
    ): 
        self._populated = True
        self.validate_model_state = validate_model_state
        self.validating_client = validating_client
        self.original_client = original_client

    @property
    def populated(self): 
        return self._populated

    @property
    def validation_result(self): 
        if self.validate_model_state == None:
            raise Exception()
        return self.validate_model_state.validation_result

    @property
    def unit_state_dict(self): 
        if self.validate_model_state == None:
            raise Exception()
        return self.validate_model_state.unit_state_dict

    def compare(self, other: ValidateModelState): 
        """Indicates whether `other` is better than the current stored model.

        If self has not yet been populated, this function will always return
        `True`. 
        """

        if ((self._populated == False) or 
            (other.validation_result < self.validation_result)
        ):
            return True
        return False


class ValidationSoftmax: 

    _validation_results: torch.Tensor
    _softmax: Optional[List]
    _unit_state_dicts: List

    def __init__(self): 
        self._validation_results = torch.Tensor(size=(0, 1))
        self._unit_state_dicts = []
        self._softmax = None
    
    def add_validation_result(
        self, 
        validate_model_state: Union[ValidateModelState, ValidatedModel],
    ): 
        self._validation_results = torch.cat(
            (
                self._validation_results, 
                torch.Tensor([[validate_model_state.validation_result]])
            ), 
            0
        ) 
        self._unit_state_dicts.append(validate_model_state.unit_state_dict)

    def compute_softmax(self): 
        inverse_validations = torch.pow(self._validation_results, -1)
        mean_inverse = torch.mean(inverse_validations)
        std_inverse = torch.std(inverse_validations)
        normalized_inverse = inverse_validations
        if std_inverse > 1e-6: 
            normalized_inverse = (inverse_validations-mean_inverse)/std_inverse
        softmax = torch.nn.Softmax(dim=0)
        self._softmax = softmax(normalized_inverse).squeeze(1).tolist()
        return self._softmax

    @property
    def softmax(self) -> List: 
        if self._softmax == None: 
            raise Exception()
        return self._softmax

    def zip_state_dict_softmax(self): 
        return zip(self._unit_state_dicts, self.softmax)
        


@dataclass 
class StructOptimizerConstructor: 
    cls_optimizer: Type[torch.optim.Optimizer]
    args: Iterable
    kwargs: Dict[str, Any]


class SplitFedServerThread:


    comm: Communicator
    _optimizer: torch.optim.Optimizer
    _validate_model_state: Optional[ValidateModelState]
    _unit_state_dict: Optional[OrderedDict]

    def __init__(
        self, comm, neural_network, cls_optimizer, criterion,
    ):
        self.comm = comm
        self.criterion = criterion
        self.cls_optimizer = cls_optimizer
        self.neural_network = neural_network
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._loss_validation = None
        self._validate_model_state = None
        self._unit_state_dict = None
         
    def optimizer(self, *args, **kwargs): 
        self._optimizer = self.cls_optimizer(
            self.neural_network.parameters(), *args, **kwargs
        )

    @property
    def validate_model_state(self): 
        if self._validate_model_state == None: 
            raise Exception()
        return self._validate_model_state
    
    @validate_model_state.setter
    def validate_model_state(self, value: Optional[ValidateModelState]): 
        self._validate_model_state = value

    @property
    def loss_validation(self): 
        if self._loss_validation == None:
            raise ThreadValidationException(
                "Validation has not been executed after training"
            )
        return self._loss_validation

    def train_offloading(self):
        self._loss_validation = None
        self._unit_state_dict = None
        self.neural_network.train()
        _, iterations_number = self.comm.recv_msg(
            expect_msg_type='CLIENT_TRAINING_ITERATIONS_NUMBER'
        )
        logger.debug(f"Number training iterations: {iterations_number}")
        self.inputs_total = 0
        for i in tqdm.tqdm(range(iterations_number)):
            msg = self.comm.recv_msg('MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
            smashed_layers = msg[1]
            labels = msg[2]

            inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
            self.inputs_total += inputs.size()[0]
            self._optimizer.zero_grad()
            outputs = self.neural_network(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self._optimizer.step()

            msg = ['MSG_SERVER_GRADIENTS_SERVER_TO_CLIENT', inputs.grad]
            self.comm.send_msg(msg)
        training_time = self.comm.recv_msg(
            expect_msg_type='MSG_TRAINING_TIME_PER_ITERATION'
        )

    @property
    def unit_state_dict(self) -> OrderedDict: 
        if self._unit_state_dict == None: 
            raise Exception()
        return self._unit_state_dict

    def neural_network_unit_compose(self, neural_network_unit): 
        _, weights_client = self.comm.recv_msg(
            expect_msg_type='MSG_LOCAL_WEIGHTS_CLIENT_TO_SERVER'
        )
        self._unit_state_dict = utils.concat_weights(
            neural_network_unit.state_dict(),
            weights_client,
            self.neural_network.state_dict(),
        )
        return self._unit_state_dict

    def neural_network_load_server(self, nn_unit): 
        server_weights = utils.split_weights_server(
            nn_unit.state_dict(), self.neural_network.state_dict()
        )
        self.neural_network.load_state_dict(server_weights)

    def neural_network_load_client(self, nn_unit):
        msg = [
            'MSG_INITIAL_GLOBAL_WEIGHTS_SERVER_TO_CLIENT', nn_unit
        ]
        self.comm.send_msg(msg)

    def validate_model(self, validate_model_state): 
        self.comm.send_msg([
            "MODEL_TO_VALIDATE", validate_model_state.unit_state_dict
        ])
        _, batch_num = self.comm.recv_msg(
            expect_msg_type="MODEL_VALIDATION_ITERATIONS_NUMBER"
        )
        for i in tqdm.tqdm(range(batch_num)):
            self.comm.recv_msg(expect_msg_type="MODEL_VALIDATION_ITERATION")
        _, validate_model_state.validation_result = self.comm.recv_msg(
            expect_msg_type='MODEL_VALIDATION_RESULT'
        )

    def validate_models(self, validate_models: CollectionValidateModelState): 
        self.comm.send_msg([
            "MODELS_TO_VALIDATE", validate_models.models_to_validate()
        ])
        _, batch_num = self.comm.recv_msg(
            expect_msg_type="MODELS_VALIDATION_ITERATIONS_NUMBER"
        )
        for i in tqdm.tqdm(range(batch_num)):
            self.comm.recv_msg(expect_msg_type="MODELS_VALIDATION_ITERATION")
        _, validation_results = self.comm.recv_msg(
            expect_msg_type='MODELS_VALIDATION_RESULT'
        )
        validate_models.validation_result = validation_results


    def validate(self): 
        _, iterations_number = self.comm.recv_msg(
            expect_msg_type='CLIENT_VALIDATION_ITERATIONS_NUMBER'
        )
        logger.debug(f"Number validation iterations: {iterations_number}")
        self.neural_network.eval()
        self.outputs_validate = torch.tensor([])
        self.targets_validate = torch.tensor([])
        with torch.no_grad(): 
            for i in tqdm.tqdm(range(iterations_number)):
                msg = self.comm.recv_msg('MSG_LOCAL_ACTIVATIONS_CLIENT_TO_SERVER')
                smashed_layers = msg[1]
                labels = msg[2]
                inputs, targets = smashed_layers.to(self.device), labels.to(self.device)
                outputs = self.neural_network(inputs)
                self.outputs_validate = torch.cat((self.outputs_validate, outputs), 0)
                self.targets_validate = torch.cat((self.targets_validate, targets), 0)


class SplitFedServer: 


    sock: socket.socket
    threads: List[SplitFedServerThread]
    pending_clients: List[SplitFedServerThread]
    pending_lock: threading.Lock
    cls_optimizer: Type[torch.optim.Optimizer]
    struct_optimizer_constructor: StructOptimizerConstructor
    thread_listen: threading.Thread
    thread_train: threading.Thread

    def __init__(
        self, ip_address, server_port, neural_network_unit, 
        cls_optimizer: Type[torch.optim.Optimizer], criterion,
        nn_server_creator, split_layer
    ): 
        self.sock = socket.socket()
        self.sock.bind((ip_address, server_port))
        self.sock.settimeout(5)
        self.neural_network_unit = neural_network_unit
        self.cls_optimizer = cls_optimizer
        self.threads = []
        self.pending_clients = []
        self.pending_lock = threading.Lock()
        self._stop_server = False
        self._stop_server_lock = threading.Lock()
        self.criterion = criterion
        self.nn_server_creator = nn_server_creator
        self.split_layer = split_layer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def optimizer(self, *args, **kwargs): 
        self.struct_optimizer_constructor = StructOptimizerConstructor(
            self.cls_optimizer, [*args], {**kwargs}
        )

    def create_thread(self, comm): 
        thread_sf = SplitFedServerThread(
            comm, self.nn_server_creator(self.split_layer),
            self.cls_optimizer, self.criterion
        )
        thread_sf.optimizer(
            *self.struct_optimizer_constructor.args,
            **self.struct_optimizer_constructor.kwargs,
        )
        with self.pending_lock:
            self.pending_clients.append(thread_sf)

    def _listen(self): 
        logger.info("Ready to connect")

        while True:
            if self.stop_server:
                return
            try: 
                self.sock.listen(5)
                (sock, (ip, _)) = self.sock.accept()
                logger.info(f'Client connected: {ip}')
                self.create_thread(Communicator(sock=sock))
            except socket.timeout: 
                continue


    @property
    def stop_server(self): 
        with self._stop_server_lock:
            return self._stop_server

    @stop_server.setter
    def stop_server(self, value: bool): 
        with self._stop_server_lock: 
            self._stop_server = value

    def listen(self): 
        if not hasattr(self, "struct_optimizer_constructor"): 
            raise InitSplitFedServerException(
                "Optimizer was not initialized."
            )
        self.thread_listen = threading.Thread(
            target=self._listen, name="thread_listen"
        )
        self.thread_listen.start()

    def _train(self): 
        threads_training = [
            threading.Thread(
                target=t.train_offloading, 
                name=f"thread_training_{i}"
            ) 
            for i, t in enumerate(self.threads)
        ]
        logger.debug("Start threads training")
        for t in threads_training: 
            t.start()
        for t in threads_training: 
            t.join()
        logger.debug("End threads training")

    def train(self, min_clients=1):
        self._add_pending_clients()
        while (len(self.threads) < min_clients) and (not self.stop_server): 
            logger.info("Not enough clients connected")
            self._add_pending_clients()
            time.sleep(2)
        return self._train()
    
    def aggregate(self, method): 
        if method == "fed_avg": 
            self.fed_avg()
        elif method == "best_validation_model":
            self.best_validation_model()
        elif method == "validation_softmax": 
            self.validation_softmax()
        elif method == "full_best_validation": 
            self.full_best_validation()
        elif method == "full_softmax": 
            self.full_softmax()
        else: 
            raise NotImplementedError(method)
        self._nn_threads_update()
        self._weights_nn_unit_send(self.threads)

    def _nn_threads_update(self): 
        for thread in self.threads: 
            thread.neural_network_load_server(self.neural_network_unit)

    def _add_pending_clients(self): 
        with self.pending_lock: 
            self.threads.extend(self.pending_clients)
            list_clients_init = self.pending_clients
            self.pending_clients = []
        self._weights_nn_unit_send(list_clients_init)

    def _weights_nn_unit_send(self, list_client_threads): 
        for client_thread in list_client_threads: 
            client_thread.neural_network_load_client(self.neural_network_unit.state_dict())

    def compose_unit_neural_networks(self): 
        for client_thread in self.threads: 
            client_thread.neural_network_unit_compose(self.neural_network_unit)

    def validate_models(self) -> List[ValidatedModel]: 
        collection_threads = []
        for client_thread in self.threads:
            model_collection = CollectionValidateModelState()
            for client_index_, client_thread_ in enumerate(self.threads): 
                model_collection.add_model(
                    client_thread_.unit_state_dict, client_index_
                )
            thread_execution = threading.Thread(
                target=client_thread.validate_models, args=(model_collection, )
            )
            collection_threads.append(
                CollectionThreadContext(thread_execution, model_collection)
            )
        for thread_context in collection_threads:
            thread_context.start_thread()
        collection_combined = CollectionCombinedValidations()
        for thread_context in collection_threads: 
            thread_context.join_thread()
            client_validations = thread_context.model_collection
            collection_combined.add_validation_results(client_validations) 
        return collection_combined\
            .compute_models_validation_result()

    def full_softmax(self): 
        self.compose_unit_neural_networks()
        validated_models = self.validate_models()
        unit_state_dict = self.global_softmax(validated_models)
        self.neural_network_unit.load_state_dict(unit_state_dict)

    def full_best_validation(self): 
        self.compose_unit_neural_networks()
        validated_models = self.validate_models()
        unit_state_dict = self.select_best_model(validated_models)
        self.neural_network_unit.load_state_dict(unit_state_dict)

    def global_softmax(self, validated_models: List[ValidatedModel]): 
        validation_results = [model.validation_result for model in validated_models]
        logger.info(validation_results)
        validation_softmax = ValidationSoftmax()
        for model in validated_models: 
            validation_softmax.add_validation_result(model)
        validation_softmax.compute_softmax()
        logger.info(validation_softmax.softmax)
        zero_model = utils.zero_init(self.neural_network_unit).state_dict()
        return utils.fed_avg(
            zero_model, list(validation_softmax.zip_state_dict_softmax())
        )

    def select_best_model(self, validated_models: List[ValidatedModel]): 
        best_model = None
        validation_results = [model.validation_result for model in validated_models]
        logger.info(validation_results)
        for model in validated_models: 
            if (best_model == None) or (not best_model.is_better(model)): 
                best_model = model
        if best_model == None: 
            raise Exception()
        logger.info("Selected model with validation result:"
                    f"{best_model.validation_result}")
        return best_model.unit_state_dict

    def random_validation(self): 
        pass

    def fed_avg(self): 
        list_weights_concat = []
        distributed_inputs_total = functools.reduce(
            lambda acc, x: x.inputs_total + acc, self.threads, 0
        )
        for thread in self.threads: 
            weights_concat = thread.neural_network_unit_compose(self.neural_network_unit)
            list_weights_concat.append((
                weights_concat, thread.inputs_total/distributed_inputs_total
            ))
        zero_model = utils.zero_init(self.neural_network_unit).state_dict()
        aggregated_model = utils.fed_avg(
            zero_model, list_weights_concat
        )
        self.neural_network_unit.load_state_dict(aggregated_model)

    def best_validation_model(self): 
        validation_threads = []
        num_threads = len(self.threads)
        for client_idx, assigned_idx in enumerate(random.sample(range(num_threads), num_threads)): 
            original_client = self.threads[client_idx]
            unit_state = original_client.neural_network_unit_compose(self.neural_network_unit)
            assigned_client = self.threads[assigned_idx]
            validate_model_state = ValidateModelState(unit_state)
            thread_execution = threading.Thread(
                target=assigned_client.validate_model,
                args=(validate_model_state,)
            )
            thread_context = ModelStateValidationThreadContext(
                thread_execution, assigned_client, assigned_idx,
                original_client, client_idx, validate_model_state
            )
            validation_threads.append(thread_context)
        for thread_context in validation_threads: 
            thread_context.start_thread()

        best_result = BestModelStateValidation()
        for thread_context in validation_threads: 
            thread_context.join_thread()
            validate_model_state = thread_context.validate_model_state
            validation_result = validate_model_state.validation_result
            logger.info(
                f"{thread_context.original_client_idx} -> "
                f"{thread_context.assigned_client_idx}: {validation_result}"
            )
            if best_result.compare(validate_model_state):
                best_result.new_best(
                    validate_model_state, thread_context.assigned_client_idx,
                    thread_context.original_client_idx,
                )

        aggregated_model = best_result.unit_state_dict
        logger.info(f"Selected model from client {best_result.original_client}")
        self.neural_network_unit.load_state_dict(aggregated_model)


    def validation_softmax(self): 
        validation_threads = []
        num_threads = len(self.threads)
        for client_idx, assigned_idx in enumerate(random.sample(range(num_threads), num_threads)): 
            original_client = self.threads[client_idx]
            unit_state = original_client.neural_network_unit_compose(self.neural_network_unit)
            assigned_client = self.threads[assigned_idx]
            validate_model_state = ValidateModelState(unit_state)
            thread_execution = threading.Thread(
                target=assigned_client.validate_model,
                args=(validate_model_state,)
            )
            thread_context = ModelStateValidationThreadContext(
                thread_execution, assigned_client, assigned_idx,
                original_client, client_idx, validate_model_state
            )
            validation_threads.append(thread_context)
        for thread_context in validation_threads: 
            thread_context.start_thread()

        validation_softmax = ValidationSoftmax()
        for thread_context in validation_threads: 
            thread_context.join_thread()
            validation_softmax.add_validation_result(thread_context.validate_model_state)
        validation_softmax.compute_softmax()
        zero_model = utils.zero_init(self.neural_network_unit).state_dict()
        aggregated_model = utils.fed_avg(
            zero_model, list(validation_softmax.zip_state_dict_softmax())
        )
        logger.info(f"Validation results: {validation_softmax._validation_results.squeeze(1).tolist()}")
        logger.info(f"Model weights: {validation_softmax.softmax}")
        self.neural_network_unit.load_state_dict(aggregated_model)


    def validate(self): 
        threads_training = [
            threading.Thread(
                target=t.validate, 
                name=f"thread_validate_{i}"
            ) 
            for i, t in enumerate(self.threads)
        ]
        logger.debug("Start threads validation")
        for t in threads_training: 
            t.start()
        for t in threads_training: 
            t.join()
        outputs = []
        targets = []
        for t in self.threads:
            outputs.append(t.outputs_validate)
            targets.append(t.targets_validate)
        return outputs, targets

    def test(self, testloader): 
        self.neural_network_unit.eval()
        with torch.no_grad(): 
            test_loss = correct = total = 0
            for inputs, targets in tqdm.tqdm(testloader): 
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.neural_network_unit(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
        logger.info(f"Test Accuracy: {acc}")


class ModelStateValidationThreadContext:

    
    thread_execution: threading.Thread
    assigned_client: SplitFedServerThread
    assigned_client_idx: int
    original_client: SplitFedServerThread
    original_client_idx: int
    validate_model_state: ValidateModelState

    def __init__(
        self, thread_execution, assigned_client, assigned_client_idx,
        original_client, original_client_idx, validate_model_state
    ):
        self.thread_execution = thread_execution
        self.assigned_client = assigned_client
        self.assigned_client_idx = assigned_client_idx
        self.original_client = original_client
        self.original_client_idx = original_client_idx
        self.validate_model_state = validate_model_state

    def start_thread(self): 
        self.thread_execution.start()

    def join_thread(self): 
        self.thread_execution.join()

    @property
    def validation_result(self): 
        return self.validate_model_state.validation_result


class CollectionThreadContext: 


    thread_execution: threading.Thread
    model_collection: CollectionValidateModelState

    def __init__(self, thread_execution, model_collection): 
        self.thread_execution = thread_execution
        self.model_collection = model_collection

    def start_thread(self): 
        self.thread_execution.start()

    def join_thread(self): 
        self.thread_execution.join()



class ModelCombinedValidations: 


    unit_state_dict: OrderedDict
    validation_results: List[float]
    _cached_computation: Optional[ValidatedModel]

    def __init__(self, unit_state_dict): 
        self.unit_state_dict = unit_state_dict
        self.validation_results = []

    def add_validation_result(self, result: ValidateModelState):
        self._cached_computation = None
        self.validation_results.append(result.validation_result)

    def compute_summarised_validation_result(self) -> ValidatedModel: 
        if self._cached_computation != None: 
            return self._cached_computation
        summarized_validation = statistics.median(self.validation_results)
        self._cached_computation = ValidatedModel(
            self.unit_state_dict, summarized_validation
        )
        return self._cached_computation




class CollectionCombinedValidations: 
    

    collection_models: Dict[int, ModelCombinedValidations]
    _cached_result: Optional[List[ValidatedModel]]

    def __init__(self): 
        self.collection_models = {}
        self._cached_result = None

    def add_validation_results(
        self, client_validations: CollectionValidateModelState,
    ): 
        for entry in client_validations.validation_result.items(): 
            client_index, validation_result = entry
            self._add_validation_result(client_index, validation_result)

    def _add_validation_result(self, client_index, validation_result): 
        if self.collection_models.get(client_index) == None: 
            unit_state_dict = validation_result.unit_state_dict
            model_validations = ModelCombinedValidations(unit_state_dict)
            self.collection_models[client_index] = model_validations
        self.collection_models[client_index].add_validation_result(validation_result)

    def compute_models_validation_result(
        self
    ) -> List[ValidatedModel]:
        if self._cached_result != None: 
            return self._cached_result
        self._cached_result = [
            model.compute_summarised_validation_result()
            for model in self.collection_models.values()
        ]
        return self._cached_result
