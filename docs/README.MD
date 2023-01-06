Our federated framework is composed of multiple components. Each one is a driver
with multiple implementations that can be used and changed. The second layer is the driver layer, where we define the
protocols that need to be implemented. The federated will delegate the work to these drivers considering that they
implemented all the required methods. Each driver is needed in the federated learning to complete a specific work. We
start with the trainer manager. This component will handle the trainers. How they are created, and how they receive the
updates. We have two built-in implementations of this driver. Sequential Trainer Manager and MPITrainerManager. The
sequential trainer manager runs the trainers in sequence. For example, suppose that the server selects ten trainers to
train a model in a specific round. These trainers will run in sequence one after each other. since the trainers are
running in sequence on the same instance as the server, the training process will take considerable time. Another
implementation of this driver is the MPI trainer. Besides its capabilities of running multiple trainers in parallel,
this manager can also represent a tangible distributed federated learning architecture since each trainer now has its
own process. Also, now we can use the properties of MPI, for example, running such a process on a different host or in a
containerized environment like docker and limiting its capability to simulate an actual device. The second driver on the
list is the trainer. This component handles the model training. We now have the torch trainer implemented; however, it's
possible also to have a Tensorflow trainer or any other.s this driver will allows our framework to support different
types of training frameworks. The following driver on the list is the client selector. This component will handle the
client selection in each round. Besides the random and all selector, it's possible to have a meta selector where clients
are selected based on an algorithm, for example, clustering clients and selecting only clients of the same cluster.
Aggregator drivers take care of aggregating the model received from the clients. For now, we have the implementation of
the avg aggregator the final metric is the methods that infer the model accuracy and loss.

Moving to the process of execution. Federated learning start with the initialize methods in which the initial model will
be created. Next, trainers are selected based on the provided client selector and send to the training method. The "
train" method, in this case, will use the trainer manager that will create trainers instances and send train requests
and wait for them to finish. When the trainer manager finishes, the collected models will be sent to the aggregator to
aggregate them to create the new initial model. We will check if the requirement or achieved in case they are federated
will end; otherwise, a new round will start.

This is the regular federated learning; however, many applications could be included between the steps. For example, we
need to plot the accuracy of each round, or we need to log what is happening in the background or measure each method's
time. Many applications can be made to improve our implementation; however, adding all of these in a usual way will be a
drawback against future scalability since lots of modifications to the same structure will make it more complex. To
solve this issue, we've added the support of subscribing component to our federated learning. After the completion of
any process, we broadcast the results to all the subscribers. A subscriber will receive these results and do the
required modifications or representations like plotting, logging, saving information for later. This is how the new
architecture would look like after adding the broadcasts. When federated start, we broadcast the start of the task and
send the configuration to the subscribers. At each step, we send a similar broadcast. Each has its own parameters. An
example of these subscribers is Timer, which logs the time needed for each process to complete, FedSave, which saves the
federated learning results like accuracy and loss after each round. Fedplot shows a plot at the end of the federated
learning or each round resumable, making our run support stop and resume even on other devices used for the runs that
require a lot of time. Finally, wandblogger, which sends the results to wandb benchmarking website for a better
representation of caching of the results. 

