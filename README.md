Simple implementations of some machine-learning algorithms
============

In neurons pacakage we have:

Perceptron
------------(https://github.com/piotrjaromin/machinelearning/imgs/perceptron_schematic.png "Perceptron")

Updates weights after processing each line.
Calculated error based on quantizer function result.

Adaline
------------
Adaptive linear neuron
(https://github.com/piotrjaromin/machinelearning/imgs/adaline_schematic.png "Adaline")

Processes whole data set, and after that updateds weights.
Uses activations function to find current error between expected result and current one.

Testing
======

to run tests:
```bash
go test ./..
```

Tests are based on [Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)