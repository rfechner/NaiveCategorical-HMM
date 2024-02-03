# HMM + Naı̈ve Bayes: HMM with multiple independent observations

A simple python implementation of Hidden Markov Models, capable of dealing with **multiple independent categorically distributed** emission signals $Y^{(k)} \sim \text{Cat}(p^{(k)})$ per timestep (not to be confused with multiple observation sequences, although this implementation supports these aswell).

## Example usage
```python
B_1 = np.array([ # emission probabilities for first emission
[0.5, 0.2, 0.3],
[0.0, 0.2, 0.8]
])

B_2 = np.array([ # emission probabilities for second emission
    [0.6, 0.4],
    [0.1, 0.9]
])

Bs = np.concatenate([B_1, B_2], axis=1)

hmm = MultiCatHMM(
    init_A = np.array([
        [0.7, 0.3],
        [0.2, 0.8]
    ]),

    init_Bs = Bs,

    init_pi = np.array(
        [0.5, 0.5]
    ),

    num_emission_symbols = np.array(
        [3, 2]
    )
)
observations = [np.array([[0, 1], [1, 0], [2, 1], [1, 0]]),
                np.array([[0, 1], [1, 0], [0, 0], [1, 1], [1, 0]])]

hmm.fit(observations) # EM - Algorithm

```

### Limitations
Currently, due to numerical instability in the implementation, the model is limited to rather short (<= 30 timesteps) observation sequences.

### Derivations
Please refer to the jupyter-notebook `hmmstudy.ipynb` for an in depth coverage and derivations
