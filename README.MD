# ModularFed - A Federated Learning Framework

## Preample

ModularFed is a federated learning framework that allows researchers and Federated Learning enthusiasts to build their theories easily and run them in a Federated Learning context withount the need to reinvent the wheel.

Different from others, ModularFed considers Federated Learning as a composition of different components that can be easily modified or replaced with as minimum changes as possible. The goal is to provide ML enthusiasts and researchers with a robust tool to start with Federated Learning in no time. You can start a Federated Learning process as easy as calling couple of lines of code as ModularFed performs the heavy lifting for you behind the scenes.

```
federated = FederatedLearning(params)
federated.add_subscriber(logger)
federated.start()
```

In this Readme file, we try to summarize the way ModularFed works and provide some code line examples so that you can start working with your ideas right away.

**NB2:** In its backend, ModularFed uses several industry standadrd machine and deep learning libraries such as Mathplotlib and Numpy and relies on PyTorch in its core to provide the deep learning components such as models, loss functions and optimizers.

**NB2:** It is expected from you, at least knowing what Federated Learning is and have some experience with Python, so that you can pick up and connect the dots easily when you read about the different components of ModularFed.

Let us start with a generic question: What components do we need to start with a Federated Learning task in general?

Mainly, a lot of things, but the main ones can be summarized as the below:
1. Data (for example, MNIST, CIFAR10, CIFAR100, Customized data, etc.)
2. A model to train the data on (for example, Logistic Regression, CNN, etc.)
3. An optimizer (for example, SGD, Adam, etc.)
4. A loss function (for example, Cross Entropy Loss)
5. Number of Epochs (for the local training rounds)
6. Number of Rounds (for the global model training rounds)
7. A Federated Averaging method (for example, FedAvg)

Each one of these components is a domain by itself and active areas of research, but ModularFed abstracts those and provides you direct access to these components with a simple call. Furthermore, inside the project, each part of these components is stored or functions as part of a bigger group. As we proceed, we will highlight each of these groups and later show how things all come together.

## How It Works

Mainly, you need 3 major components following the steps below to run a Federated Learning tasks with ModularFed:
1.	**preload** to prepare the data
```
client_data = preload('mnist', ShardDistributor(300, 2)).select(range(100))
```
2.	**TrainerParams** to set the training parameters
```
trainer_params = TrainerParams(
                trainer_class=trainers.TorchTrainer,
                batch_size=50,
                epochs=3,
                optimizer='sgd',
                criterion='cel',
                lr=0.1)
```
3.	**Federated** to start your federated learning process
```
federated = FederatedLearning(
            num_rounds=25,
            initial_model=lambda: CNN_OriginalFedAvg(only_digits=True),
            trainer_manager=SeqTrainerManager(),
            aggregator=aggregators.AVGAggregator(),
            metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
            client_selector=client_selectors.Random(2),
            client_scanner=client_scanners.DefaultScanner(client_data),
            trainer_config=trainer_params,
            trainers_data_dict=client_data,
            desired_accuracy=0.99)
            
federated.start()
```

### Step 1: Prearing the data

A big chunk of time in ML goes to preprocessing data and making it suit our framework and needs.

How to preprocess our data so that it works with the ModularFed framework?

First, we need to bring the data to the framework.
This can be done using the preload function as follows:
```
client_data = preload('mnist', ShardDistributor(300, 2)).select(range(100))
```

The above code lone returns clients' data of type `Dict[int, DataContainer]` where the output is a dictionary having the client number as an `integer` and the data per each client in the form of a `DataContainer`.
`Datacontainer` is a custom class in ModularFed containing the features and the labels of the data that can be accessed via:

```
features = datacontainer.x
labels = datacontainer.y
```

The preload function takes the `name` of the dataset as a string for the first parameter and a `Distributer` as the other parameter.

ModularFed comes equipped with some of the most used datasets in the research community such as: 
```javascript
- "mnist10k
- "mnist"
- "femnist"
- "kdd"
- "kdd_train"
- "kdd_test"
- "fekdd_test"
- "fekdd_train"
- "signs"
- "cifar10"
- "fall_by_client"
- "fall_ar_by_client"
- "mnist10k_mr1"
- "cifar100_train"
- "cifar100_test"
- "cifar100"
```

The second parameter is a `Distributer`.
In ModularFed we have several of distributers, all of which control how the data will be distributed across the framework. 
Usually, the name of the Distributer is self explanatory. For example if you are calling the 'ShardDistributor' class, it would mean that you would like to partition your data into shards. Each distributer, depending on how it distributes the data has parameters. For example, `DirichletDistributor` can have a parameter such as `skewness` that will control how the data distribution is skewed, while the `SizeDistributer` might have a `min` and `max` parameter to control the distribution. 

ModularFed comes equipped with the following distributors. (we also showcase a pprint output of the output data after calling each distributor):

<ul>
  <li>DirichletDistributor(num_clients, num_labels, skewness=0.5)</li>
  <li>PercentageDistributor(num_clients, min_size, max_size, percentage)</li>
  <li>LabelDistributor(num_clients, min_size, max_size, percentage)</li>
  <li>SizeDistributor(num_clients, min_size, max_size)</li>
  <li>UniqueDistributor(num_clients, min_size, max_size)</li>
  <li>ShardDistributor(shard_size, shards_per_client)</li>
  <li>ManualDistributor(size_dict: Dict[int, int])</li>
  <li>PipeDistributor</li>
</ul>

Examples:
```
#DirichletDistributor
client_data = preload('mnist', DirichletDistributor(num_clients=5, num_labels=10, skewness=0.5))

{0: Size:12217, Unique:[1 2 3 4 5 6 7 8 9], Features:torch.Size([784]),
 1: Size:15265, Unique:[0 1 2 3 4], Features:torch.Size([784]),
 2: Size:14305, Unique:[0 1 2 3 5 6], Features:torch.Size([784]),
 3: Size:12214, Unique:[0 1 2 3 4 5 6 7 8 9], Features:torch.Size([784]),
 4: Size:15134, Unique:[0 1 2 3 4 5 6 7], Features:torch.Size([784])}

#PercentageDistributor
client_data = preload('mnist', PercentageDistributor(num_clients=5, min_size=90, max_size=100, percentage=100))

{0: Size:18, Unique:[6], Features:torch.Size([784]),
 1: Size:12, Unique:[1], Features:torch.Size([784]),
 2: Size:20, Unique:[6], Features:torch.Size([784]),
 3: Size:14, Unique:[6], Features:torch.Size([784]),
 4: Size:16, Unique:[0], Features:torch.Size([784])}


#LabelDistributor
client_data = preload('mnist', LabelDistributor(num_clients=5, min_size=1, max_size=10, label_per_client=3))

{0: Size:6, Unique:[1 2 3], Features:torch.Size([784]),
 1: Size:3, Unique:[4 5 6], Features:torch.Size([784]),
 2: Size:3, Unique:[7 8 9], Features:torch.Size([784]),
 3: Size:3, Unique:[0 1 2], Features:torch.Size([784]),
 4: Size:3, Unique:[3 4 6], Features:torch.Size([784])}


#SizeDistributor
client_data = preload('mnist', SizeDistributor(num_clients=5, min_size=1, max_size=10))

{0: Size:8, Unique:[1 2], Features:torch.Size([784]),
 1: Size:6, Unique:[2 3], Features:torch.Size([784]),
 2: Size:7, Unique:[4 5 6 7], Features:torch.Size([784]),
 3: Size:7, Unique:[0 8 9], Features:torch.Size([784]),
 4: Size:8, Unique:[0 9], Features:torch.Size([784])}


#UniqueDistributor
client_data = preload('mnist', UniqueDistributor(num_clients=5, min_size=1, max_size=10))

{0: Size:8, Unique:[0], Features:torch.Size([784]),
 1: Size:4, Unique:[1], Features:torch.Size([784]),
 2: Size:4, Unique:[2], Features:torch.Size([784]),
 3: Size:2, Unique:[3], Features:torch.Size([784]),
 4: Size:4, Unique:[4], Features:torch.Size([784])}


#ShardDistributor
client_data = preload('mnist', ShardDistributor(300, 2)).select(range(5))

{0: Size:600, Unique:[3. 4.], Features:(784,),
 1: Size:600, Unique:[4. 5.], Features:(784,),
 2: Size:600, Unique:[2. 4.], Features:(784,),
 3: Size:600, Unique:[3. 5.], Features:(784,),
 4: Size:600, Unique:[0. 9.], Features:(784,)}

```

For example, let us try to create a set of clients to partake in the Federated Learning rounds and populate them with data from the MNIST dataset.
We want to distribute the data as shards such that, in our example each client will have `2 shards` each one having `300 images`.
This means clients will have `600 images` distributed equally between `2 classes`. That is,` 300 images` from each class. 
```
client_data = preload('mnist', ShardDistributor(300, 2)).select(range(100))
```
When you select a dataset, such as `'mnist'` and it has not been downloaded before, ModularFed fetches it from the internet and creates a `.pkl` file with the distribution you provided as the second parameter. You can also provide name for the `pkl` file if you want by passing a string with the tag parameter such as `tag='mypklfile`. The `tag` parameter is optionl. If you do not pass it, ModularFed will save the file with an automated generated name with the information of its creation and save it in the `datasets/pickles` folder in the project. By doing this, (1) in the future, if you try to use the same dataset with the same distribution, there would be no need to download it and it will be used directly from the already saved `pkl` file and (2) you can move the `pkl` file easily between projects or even share it with other researchers if they want to work on similar dataset.

Now that we have the data, let's keep it aside and build the trainer parameters needed which will be important part of our learning. 
 
### Step 2: Preparing the Training Parameters

Once we have the data ready, we start preparing the training parameters using the class `TrainerParams`. <br>
The class `TrainerParams` accepts several parameters as the main ingredient of a Federated Learning task. For example, the optimizer and the criterion (i.e., the loss function) and others are provided here.

- **Trainer Class:** The trainer class passes the training method from the class <italic>TorchTrainer</italic> which contains the <italic>train</italic> function internally that carries on the actual training on the local devices.
  
- **Batch Size:** Characterized by batch_size in the learning parameter, this hyperparameter sets the number of batches the data will be fed to the neural network. Depending on your hardware, you can select the appropriate size of the batch. Larger batch sizes usually need better hardware resources. Again, you need to experiment with this value to reach a value that suits your needs
  
- **Epochs:** In the context of deep learning in general might have a bit different meaning that it has in federated learning. In the context of Federated Learning, epoch symbolizes the amount of rounds a model will train locally on each client. While rounds in Federated Learning usually symbolizes the amount of times the entire clients will finalize their epochs and send their local model to the global model for aggregation.
  
- **Optimizer:** At the moment, ModularFed provides the SGD and Adam optimizers. You can call ‘sgd’ or ‘adam’ in the parameter. In case you want to use a different optimizer, you can add it yourself in the src/federated/components/params.py file. This is where the modular advantages of ModularFed comes in where adding or changing something is very fast.
  
- **Criterion:** Or interchangeable used with the Loss Function. At the moment, ModularFed provides the Cross Entropy Loss function by calling ‘cel’ (the first letter of each of the words). In case you want to use a different criterion, you can add it yourself in the src/federated/components/params.py file. This is where the modular advantages of ModularFed comes in where adding or changing something is very fast.
  
- **Learning Rate:** The learning rate is a hyperparameter used with the optimizer. In ModularFed it is characterized by lr in the parameters where it takes a float as the learning rate. Common learning rates are 0.1, 0.01, 0.001, etc depending on the learning task at hand. Coming up with the correct lr needs a bit testing and experimenting until you get a good lr for your task at hand.

```
trainer_params = TrainerParams(
                 trainer_class=trainers.TorchTrainer,
                 batch_size=32,
                 epochs=3,
                 optimizer='sgd',
                 criterion='cel',
                 lr=0.1)
```

### Step 3: Preparing and Launching the Federated Learning Process

The main component for the Federated Learning process is the class `FederatedLearning`. <br>
`FederatedLearning` takes up to 11 parameters of which some are mandatory and some optional:

- **initial_model:** This is the model on which the local training per client will take place. ModularFed comes with several well known models. (More information on models in the models section below) 

- **trainers_data_dict:** Which is of type `Dict[int, DataContainer]` that the `preload` method returns in step 1.

- **trainer_manager:** Which is of type `TrainerManager` and is responsible for managing the training data in the learning process. ModularFed provides 2 types of TrainManagers:
  - `SeqTrainerManager` (If you are not planning on using parallel or cluster computing, using `SeqTrainerManager` is an easier option.)
  - `MPITrainerManager`
  
- **trainer_config:** Which is of type `TrainerParams` which were created in step 2.

- **aggregator:** Which is of type `Aggregator` and controls how the central server will aggregate the local models into the global one. ModularFed provides the `AVGAggregator` aggregator approach.
  
- **client_selector:** Which is of type `ClientSelector`. This is where you control how many clients will be selected by the framework for each round. ModularFed has 4 built in options for client selection
  - `All`: which returns all the clients created in preload in step 1.
  - `random`: which selects random clients based on the number you specify as its parameter
  -  `ClusterSelector`:
  -  `Specific`: 

- **metrics:** Which takes `ModelInfer` that is handles the inference of the model using the function `AccLoss` that calculates the accuracy and the loss of the batched data.
  
- **num_rounds**: Which is an integer representing the number of rounds the central server will gather the local models and aggregate them into a global one. 
(If this parameter is not provided, a default value of 10 rounds is used)

- **desired_accuracy**: Which is a float number that highlights the accuracy at which the convergence would be considered acceptable and the Federated Learning would stop. 0.90 implies 90%. (If this parameter is not provided, a default value of 0.99 (99%) is used)

- **train_ratio**: This parameter is important if no testing data is provided to the framework against which ModularFed can perform the testing. Accordingly, this value is the ratio with which the training data `trainers_data_dict` will be further into training and test sets.  For example, a value of 0.7 would mean the data will be divided into 70% training set and 30% testing set. (If this parameter is not provided, a default value of 0.8 (80%) is used)

- **test_data**:(optional) Accepts a Datacontainer similar to `trainers_data_dict` but containing the test data against which we want our model to get tested. Usually this is data that is not used at all during the training and the model would be seeing the data for the first time.

Once the above parameters are ready, we can create an instance of FederatedLearning class and give to it all the information:

```
federated = FederatedLearning(
    initial_model=lambda: CNN_OriginalFedAvg(only_digits=True),
    trainers_data_dict=client_data,
    trainer_manager=SeqTrainerManager(),
    aggregator=aggregators.AVGAggregator(),
    num_rounds=25,
    metrics=metrics.AccLoss(batch_size=50, criterion='cel'),
    client_selector=client_selectors.Random(2),
    client_scanner=client_scanners.DefaultScanner(client_data),
    trainer_config=trainer_params,
    desired_accuracy=0.99)
```

Once done, we can call the `start` method on the instantiated FederatedLearning class, which will start the Federated Learning process:

```
federated.start()
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

Desired accuracy defines the accuracy of which federated learning should stop when it is reached

### train_ratio

FederatedLearning instance splits the data into train and test when it initializes. train_ratio value decides where we
should split the data. For example, for a train_ratio=0.8, that means train data should be 80% and test data should 20%
for each client data.

### test_data

An optional parameter used for cases when the dataset have already specific test data to test the model accuracy.<br>
Otherwise, federated learning class will internally split the given data to train, test sets

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
Using a clone docker file (have always the latest updates)
```bash
docker build -t arafeh94/localfed .
```

Using a preloaded docker hub image (updated occasionally or after major changes)
```bash
docker pull arafeh94/localfed
```

### Create one container for the server and two for the clients

```bash
docker run -d --name head -p 20:20 arafeh94/localfed
docker run -d --name node1 arafeh94/localfed
docker run -d --name node2 arafeh94/localfed
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
cd ${HOME}/localfed/apps/experiments/
mpirun -np 3 --host localhost,172.17.0.3,172.17.0.4 python3 distributed_averaging.py
```
