# mynet

**mynet** is a personal deep learning research project focused on implementing and training neural networks from scratch (with minimal libraries) on classic datasets such as **MNIST** and **ESC-50**.  
The goal is to explore the behavior of different architectures, activation functions, and optimization strategies in small to medium-scale deep learning experiments.

---
# Usage

Feel free to clone the repository and experiment with the models yourself. All the datasets are already downloaded on this repository.
One prerequisit is you need to have open-blas installed on your system, to do this on ubuntu its

```
sudo apt update
sudo apt install libopenblas-dev
```

---

## Current Results

Note: My computer is using an AMD Ryzen 5 7600 results are likely to vary. 

**Dataset:** MNIST  

**Model:** Neural Network with 2 hidden layers with 32 neurons.

**Batch Size:** 256

Visualization of training progress:  

Learning rate - 0.1

![MNIST Training Curve](images/Figure_1.png)

Learning rate - 5.0

![MNIST Training Curve](images/Figure_2.png)

Learning rate - 10.0

![MNIST Training Curve](images/Figure_3.png)

Learning rate - 100.0

![MNIST Training Curve](images/Figure_4.png)

Learning rate - 500.0

![MNIST Training Curve](images/Figure_5.png)

---


