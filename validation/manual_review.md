# Manual Review of Retrieval Results

Date: 2025-06-01 (Updated based on new `retrieval_results.json`)

Below are the retrieval results for each sample query. For each passage, mark ✓ if on-topic or ✗ if off-topic, with brief notes.

| Query ID | Question                                                | Passage # | Passage ID                    | Section      | ✓/✗ | Notes                                                                 |
| -------- | ------------------------------------------------------- | --------- | ----------------------------- | ------------ | --- | --------------------------------------------------------------------- |
| **Q1** | What is backpropagation in a multilayer perceptron?     | 1         | DeepLearning1-0039b4d5-f865   | unknown      | ✗   | About activation functions, not specifically backprop.                  |
|          |                                                         | 2         | DeepLearning1-8e368b78-495b   | unknown      | ✗   | General intro/plan for MLP, not the algorithm.                        |
|          |                                                         | 3         | DeepLearning1-2f9c56bc-cb2d   | unknown      | ✗   | About energy consumption, perceptron simple; off-topic.                 |
|          |                                                         | 4         | DeepLearning1-aa77bc5e-8361   | unknown      | ?   | Mentions MLP & Apprentissage/Optimisation in plan; indirect.          |
|          |                                                         | 5         | DeepLearning1-85986ea1-0364   | Architecture | ?   | Roles of activation functions in MLPs; indirect to backprop.          |
| **Q2** | How do you initialize weights and biases in an MLP?     | 1         | TD1-3f90ca49-0e6e             | Architecture | ✗   | "Architecture du MLP" header; too general, no init details.           |
|          |                                                         | 2         | DeepLearning1-85766d89-6e22   | Architecture | ?   | Mentions Poids (θ) and biais, but in context of gradient/optimization.  |
|          |                                                         | 3         | DeepLearning1-608ab7a5-8540   | Architecture | ✗   | About gradient descent variants, not initialization.                    |
|          |                                                         | 4         | DeepLearning1-0039b4d5-f865   | unknown      | ✗   | About activation functions, not weight initialization.                |
|          |                                                         | 5         | DeepLearning1-8e368b78-495b   | unknown      | ✗   | General intro/plan for MLP, not initialization.                       |
| **Q3** | Explain the forward pass in a neural network.           | 1         | DeepLearning1-0039b4d5-f865   | unknown      | ✗   | About activation functions, not specifically forward pass.              |
|          |                                                         | 2         | DeepLearning1-46141ec0-621d   | Architecture | ✓   | Describes final prediction, neuron types for outputs; related to pass.  |
|          |                                                         | 3         | DeepLearning1-2f9c56bc-cb2d   | unknown      | ✗   | About energy consumption, perceptron simple; off-topic.                 |
|          |                                                         | 4         | DeepLearning1-339936eb-e923   | Architecture | ✓   | "La propagation avant calcule la sortie du réseau." - Directly relevant. |
|          |                                                         | 5         | DeepLearning1-7534a7e9-109e   | Architecture | ✓   | Formulas Z(l) and A(l) = f(zl) are core to forward pass.              |
| **Q4** | What is the role of the sigmoid activation function?    | 1         | DeepLearning1-aabe9056-547c   | Architecture | ✓   | Example of applying sigmoid for probability.                          |
|          |                                                         | 2         | DeepLearning1-1d9f1428-a295   | Architecture | ✓   | Mentions non-linearity, adapting output, specifically names Sigmoid.    |
|          |                                                         | 3         | DeepLearning1-0039b4d5-f865   | unknown      | ?   | General activation definition; sigmoid is an example.                 |
|          |                                                         | 4         | DeepLearning1-0cfdf65e-29db   | Architecture | ✗   | Example provided is for Tanh, not Sigmoid.                            |
|          |                                                         | 5         | DeepLearning1-46141ec0-621d   | Architecture | ✓   | Mentions "Classification binaire → 1 neurone (Sigmoïde)".              |
| **Q5** | How is Mean Squared Error computed?                     | 1         | DeepLearning1-7f378b36-e887   | Architecture | ?   | Mentions "Types de Fonctions de Perte" and regression; MSE is a type.   |
|          |                                                         | 2         | DeepLearning1-1fadbadf-0f82   | unknown      | ✗   | Perceptron weight update rule, not MSE.                               |
|          |                                                         | 3         | Td1\_correction-06fcbc77-6835 | unknown      | ✗   | Calculation of loss (binary cross-entropy), not MSE.                  |
|          |                                                         | 4         | DeepLearning1-2468c67c-c3ab   | Objectif     | ✗   | Mentions MAE and R2 for regression, not MSE computation details.        |
|          |                                                         | 5         | DeepLearning1-608ab7a5-8540   | Architecture | ✗   | Gradient descent variants, not MSE.                                   |
| **Q6** | Describe gradient descent for training neural networks. | 1         | DeepLearning1-608ab7a5-8540   | Architecture | ✓   | Describes Batch, Stochastic, and Mini-Batch Gradient Descent.         |
|          |                                                         | 2         | DeepLearning1-85766d89-6e22   | Architecture | ✓   | Defines terms for gradient descent: θ, η, ∇θLθ.                       |
|          |                                                         | 3         | DeepLearning1-339936eb-e923   | Architecture | ✓   | "La descente de gradient met à jour les paramètres..."                  |
|          |                                                         | 4         | DeepLearning1-1e65dac6-abf4   | Objectif     | ?   | Lists hyperparameters including learning rate; related to optimization. |
|          |                                                         | 5         | DeepLearning1-2702d639-eb1d   | Architecture | ✓   | Describes advanced optimizers like Momentum, RMSProp, Adam.           |