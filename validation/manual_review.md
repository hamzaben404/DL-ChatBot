# Manual Review of Retrieval Results

Date: 2025-05-29

Below are the retrieval results for each sample query. For each passage, mark ✓ if on-topic or ✗ if off-topic, with brief notes.

| Query ID | Question                                                | Passage # | Passage ID                    | Section      | ✓/✗ | Notes                                             |
| -------- | ------------------------------------------------------- | --------- | ----------------------------- | ------------ | --- | ------------------------------------------------- |
| **Q1**   | What is backpropagation in a multilayer perceptron?     | 1         | DeepLearning1-0039b4d5-f865   | unknown      | ✗   | Focuses on activation, not backprop               |
|          |                                                         | 2         | DeepLearning1-8e368b78-495b   | unknown      | ✗   | General MLP plan, not algorithm detail            |
|          |                                                         | 3         | DeepLearning1-2f9c56bc-cb2d   | unknown      | ✗   | Energy consumption, off-topic                     |
|          |                                                         | 4         | DeepLearning1-aa77bc5e-8361   | unknown      | ✓   | Introduction to MLP sections                      |
|          |                                                         | 5         | DeepLearning1-85986ea1-0364   | Architecture | ✓   | Mentions propagation roles                        |
| **Q2**   | How do you initialize weights and biases in an MLP?     | 1         | TD1-3f90ca49-0e6e             | Architecture | ✓   | Likely shows weight diagrams                      |
|          |                                                         | 2         | DeepLearning1-85766d89-6e22   | Architecture | ✓   | Symbols for θ and b explained                     |
|          |                                                         | 3         | DeepLearning1-608ab7a5-8540   | Architecture | ✗   | Describes gradient descent, not init              |
|          |                                                         | 4         | DeepLearning1-0039b4d5-f865   | unknown      | ✗   | Activation function description                   |
|          |                                                         | 5         | DeepLearning1-8e368b78-495b   | unknown      | ✗   | General MLP plan                                  |
| **Q3**   | Explain the forward pass in a neural network.           | 1         | DeepLearning1-0039b4d5-f865   | unknown      | ✗   | Activation-focused, not forward pass specifically |
|          |                                                         | 2         | DeepLearning1-46141ec0-621d   | Architecture | ✓   | Describes output and layer formulas               |
|          |                                                         | 3         | DeepLearning1-2f9c56bc-cb2d   | unknown      | ✗   | Off-topic energy consumption                      |
|          |                                                         | 4         | DeepLearning1-339936eb-e923   | Architecture | ✓   | Explicitly mentions forward propagation           |
|          |                                                         | 5         | DeepLearning1-7534a7e9-109e   | Architecture | ✓   | Layer formula with Z and activation               |
| **Q4**   | What is the role of the sigmoid activation function?    | 1         | DeepLearning1-aabe9056-547c   | Architecture | ✓   | Numeric example applying sigmoid                  |
|          |                                                         | 2         | DeepLearning1-1d9f1428-a295   | Architecture | ✓   | Lists benefits of sigmoid                         |
|          |                                                         | 3         | DeepLearning1-0039b4d5-f865   | unknown      | ✗   | General activation definition                     |
|          |                                                         | 4         | DeepLearning1-0cfdf65e-29db   | Architecture | ✗   | Example uses tanh, not sigmoid                    |
|          |                                                         | 5         | DeepLearning1-46141ec0-621d   | Architecture | ✗   | Describes output layer, not role                  |
| **Q5**   | How is Mean Squared Error computed?                     | 1         | DeepLearning1-7f378b36-e887   | Architecture | ✗   | Mentions loss types, not specific MSE formula     |
|          |                                                         | 2         | DeepLearning1-1fadbadf-0f82   | unknown      | ✗   | Perceptron update step, off-topic                 |
|          |                                                         | 3         | Td1\_correction-06fcbc77-6835 | unknown      | ✗   | Numeric backprop example, not MSE                 |
|          |                                                         | 4         | DeepLearning1-2468c67c-c3ab   | Objectif     | ✗   | Mentions MAE and R2, not MSE                      |
|          |                                                         | 5         | DeepLearning1-608ab7a5-8540   | Architecture | ✗   | Gradient descent list, off-topic                  |
| **Q6**   | Describe gradient descent for training neural networks. | 1         | DeepLearning1-608ab7a5-8540   | Architecture | ✓   | Describes batch, SGD, mini-batch                  |
|          |                                                         | 2         | DeepLearning1-85766d89-6e22   | Architecture | ✓   | Mentions learning rate and gradient symbol        |
|          |                                                         | 3         | DeepLearning1-339936eb-e923   | Architecture | ✓   | Connects forward/backward to gradient updates     |
|          |                                                         | 4         | DeepLearning1-1e65dac6-abf4   | Objectif     | ✓   | Lists hyperparameters for optimization            |
|          |                                                         | 5         | DeepLearning1-2702d639-eb1d   | Architecture | ✓   | Describes Momentum, RMSProp, Adam                 |
