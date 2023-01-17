# ModularFed - A Federated Learning Framework

ModularFed is a federated learning framework that allows researchers to easily build
their theories and run them in a federated learning context. <br>

Different from others, ModularFed considers federated learning as a composition of different components that should be
replaced on the spot without many changes.

An example of running a federated learning task

```
federated = FederatedLearning(params)
federated.add_subscriber(logger)
federated.add_subscriber(plotter)
federated.start()
```

This code will start a federated learning application with the given parameters and subscribers.

For a quick overview, a running example is available on codeocean following this link: https://codeocean.com/capsule/2481633/tree
## Federated Parameters

FederatedLearning class requires different parameters to control its behaviour. Of these parameters, we can list

### trainer_manager:

An instance of the TrainerManager interface defines how trainers are running. TrainerManager is followed by
TrainerParams, which define the clients' training parameters, such as epochs, loss, and so on. It exists, for now, three
builtin trainer managers:

- SeqTrainerManager that runs trainers sequentially on a single thread
- MPITrainerManager runs trainers on a different instance, so it would be possible to run multiple trainers
  simultaneously. Using this manager would also allow us to control the trainers' behaviour regarding allocated
  resources for their runs since they run on different processes (or different devices). Such control is given because
  we are using MPI to complete this task. Refer to MPI Section to see examples.

```python
trainer_manager = SeqTrainerManager()
trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=25, optimizer='sgd',
                               criterion='cel', lr=0.1)
```

### aggregator:

An instance of an Aggregator interface defines how the collected models are merged into one global model. AVGAggregator
is the
widely used aggregator that takes the average of the models' weights to generate the global model

```python
aggregator = aggregators.AVGAggregator()
```

### client_selector

An instance of a ClientSelector interface controls the selected clients to train in each round. Available client
selectors:

- Random(nb): select [nb] a number of clients randomly to train in each round
- All(): select all the clients to train in each round

```python
# select 40% of the clients to train a model each round
client_selector = client_selectors.Random(0.4)

# select 10 of the clients to train a model each round
client_selector = client_selectors.Random(10)

# select all clients
client_selector = client_selectors.All()
```

### metrics

An instance of ModelInfer is used to test the model accuracy on test data after each round. Available metrics:

- AccLoss(batch_size,criterion): test the model and returns accuracy and loss

```python
acc_loss_metric = metrics.AccLoss(batch_size=8, criterion=nn.CrossEntropyLoss())
```

### trainers_data_dict

A dictionary of <b>[client_id:int,DataContainer]</b> that defines each client what data they have. DataContainer is a
class that holds (x,y), the features and labels. Example:

```python
from src.data.data_container import DataContainer

# clients in this dataset have 3 records, each having three features. 
# A record is labelled 1 when all the features have the same value and 0 otherwise
# A sample of data
client_data = {
    0: DataContainer(x=[[1, 1, 1],
                        [2, 2, 3],
                        [2, 1, 2]],
                     y=[1, 0, 0]),
    1: DataContainer(x=[[1, 2, 1],
                        [2, 2, 2],
                        [2, 2, 2]],
                     y=[0, 1, 1])
}
```

Usually, we only test the model on manually created data. This example is only to know the structure of the input.
DataContainer contains some valuable methods used inside federated learning classes. However,
you can refer to the data loader section to create meaningful data.

### initial_model

A function definition that the execution should return an initialized model. Example:

```python
initial_model = lambda: LogisticRegression(28 * 28, 10) 
```

or

```python
def create_model():
    return LogisticRegression(28 * 28, 10)


initial_model = create_model
```

### num_rounds

For how many rounds the federated learning task should run? 0 used for unlimited

### desired_accuracy

Desired accuracy where federated learning should stop when it is reached

### train_ratio

FederatedLearning instance splits the data into train and test when it initializes. train_ratio value decides where we
should split the data. For example, for a train_ratio=0.8, that means train data should be 80% and test data should 20%
for each client data.

### test_data

An optional parameter used for cases when the dataset have already specific test data to test the model accuracy.

```python
test_data = DataContainer(...)
```

## Federated Learning Example

```python
from torch import nn

client_data = preload('mnist', LabelDistributor(num_clients=100, label_per_client=5, min_size=600, max_size=600))
trainer_params = TrainerParams(trainer_class=trainers.TorchTrainer, batch_size=50, epochs=25, optimizer='sgd',
                               criterion='cel', lr=0.1)
federated = FederatedLearning(
    trainer_manager=SeqTrainerManager(),
    trainer_config=trainer_params,
    aggregator=aggregators.AVGAggregator(),
    metrics=metrics.AccLoss(batch_size=50, criterion=nn.CrossEntropyLoss()),
    client_selector=client_selectors.Random(0.4),
    trainers_data_dict=client_data,
    initial_model=lambda: LogisticRegression(28 * 28, 10),
    num_rounds=50,
    desired_accuracy=0.99,
)
federated.start()
```

## Data Loader

Federated Learning tasks should include experiments of different kinds of data set that are usually non identically
distributed and compare to data identically distributed and so on. That would cause researcher to preprocess the same
data differently before even starting with federated learning. To get the idea, suppose that we are working on mnist
dataset. Using federated learning, we should test the model creation under these scenarios:

1. mnist data distributed into x number of clients with the same simple size
2. mnist data distributed into x number of clients of big different in sample size
3. mnist data distributed into x number of clients where each have x number of labels (shards), by google
4. mnist data distributed into x number of clients where each have at least 80% of the same label

Different scenario could be tested, and generating these scenarios can take a lot of work. And this only for mnist
dataset without considering working with other datasets.

To solve this issue we create a data managers that can help to generate data based on the listed scenarios. It is also
capable of saving the distributed data and load it for the next run avoiding the long loading time due to distributing
data to clients.

### example

```python
# label distributor distribute data to clients based on how many labels is client should have. 
# Example, distribute the data such as each client have 5 labels and 600 records. 
distributor = LabelDistributor(num_clients=100, label_per_client=5, min_size=600, max_size=600)
client_data = preload('mnist', distributor)
# preload will take care of downloading the file from our cloud, distribute it based on the passed distributor 
# and finally save it into a pickle file.
```

Available Distributors:

```angular2html
soon...
```

## Federated Plugins

Many additional task might be required when running a federated learning application. For Example:

- plot the accuracy after each round
- plot the local accuracy of each client
- log what is happening on each step
- measure the accuracy on a benchmarking tools like wandb or tensorboard
- measure the time needed to complete each federated learning step
- save results to database
- add support for blockchain or different framework
- create a new tools that requires changes in the core framework

All of these issue related to the scalability of the application. As such, we have introduced federated plugins. This
implementation works by requesting from FederatedLearning to register an event subscriber. A subscriber will receive a
broadcasts from the federated learning class in each step allowing each subscriber to further enhance the application
with additional features.

Example of federated event subscribers:

```python
federated = ...
# display log only when federated task is selecting trainers and when the round is finished
federated.add_subscriber(FederatedLogger())
# compute the time needed to all trainers to finished training
federated.add_subscriber(Timer())
# show plots each round
federated.add_subscriber(RoundAccuracy(plot_ratio=1))
federated.add_subscriber(RoundLoss(plot_ratio=1))
```

## Federated Example

For more example, refer to apps/experiments

<u><b>Important Examples:</b></u>

<b>Simple example:</b> apps/experiments/federated_averaging.py<br>
<b>Description:</b> FederatedLearning using MNIST dataset distributed to 100 clients with 600 records each.

<b>Distributed example:</b> apps/experiments/distributed_averaging.py<br>
<b>Description:</b> same as a sample example but using MPI for parallel training. Using MPI requires additional software
on the host. Please refer to MPI documentation for additional information. You may find the command required to run the
script at the top of the script.

```bash
# run 11 instances of the script. The first one will be considered the server, while the rest ten will be considered 
# as clients. Make sure the client selector selects ten clients each round to benefit from all instances
mpiexec -n 11 python distributed_averaging.py
```

## Docker Containers

Enable parallel distributed training through docker containers.

### Build

```bash
docker build -t arafeh94/ModularFed .
```

```bash
docker pull arafeh94/ModularFed
```

### Create one container for the server and two for the clients

```bash
docker run -d --name head -p 20:20 arafeh94/ModularFed
docker run -d --name node1 arafeh94/ModularFed
docker run -d --name node2 arafeh94/ModularFed
```

### Check containers' IP

```bash
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' head
```

### Connect to a node

```bash
docker exec -it -u mpirun head /bin/bash
```

### Check if all is working

```bash
# check if SSH connection is established and working correctly
ssh 172.17.0.3 hostname
# should return the docker container hostname
cd ~
# check if MPI can initialize without issues
mpirun -np 3 --host localhost,172.17.0.3 python3 ${HOME}/mpi4py_benchmarks/check.py
# nothing appear => everything is working

# check if containers can send and receives messages without any issues 
mpirun -np 3 --host localhost,172.17.0.3 python3 ${HOME}/mpi4py_benchmarks/com.py
# nothing appear => everything is working
```

### Run distributed federated learning

```bash
docker exec -it -u mpirun head /bin/bash
cd ${HOME}/ModularFed/apps/experiments/
mpirun -np 3 --host localhost,172.17.0.3,172.17.0.4 python3 distributed_averaging.py
```
