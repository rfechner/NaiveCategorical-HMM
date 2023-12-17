"""
Author: Richard Fechner, 2023

Simple Python implementation of a Hidden Markov Model 
with multiple categorical Emissionsignals.
"""

import numpy as np

from tqdm import tqdm
from typing import List, Sequence
class MultiCatEmissionHMM():
    """
    Hidden Markov Model with Multiple Categorical Emissions.
    This class represents a Hidden Markov Model (HMM) where the emissions are modeled
    as distinct Categorical distributions. The HMM consists of a set of states, transition
    probabilities between states, initial state probabilities, and multiple emission matrices.
    """
    
    def __init__(self, init_pi : np.ndarray, init_A : np.ndarray, init_Bs : np.ndarray, num_emission_symbols : List[int]):        
        """
        Initializes an instance of MultiCatEmissionHMM.

        :param init_pi: The initial state probabilities. A 1-D array of shape (num_states,).
        :type init_pi: np.ndarray

        :param init_A: The state transition probability matrix. A 2-D array of shape (num_states, num_states).
        :type init_A: np.ndarray

        :param init_Bs: The emission matrices for each state. A 2-D array of shape (num_states, sum(num_emission_symbols)) of concatenated
                    emission matrices.
        :type init_Bs: np.ndarray

        :param num_emission_symbols: A list containing the num_emission_symbols of individual sequences in the dataset.
        :type num_emission_symbols: List[int]
        """
        self.pi : np.ndarray = init_pi
        self.A : np.ndarray = init_A
        self.Bs : np.ndarray = init_Bs
        self.num_emission_symbols : List[int] = num_emission_symbols
        self.num_states : int = len(init_pi)

    def predict(self, Ys : np.ndarray) -> np.ndarray:
        """
        Predict the posterior distribution :math:`p(X|Ys, \\theta)` over the hidden
        states, given some observations.

        :param Ys: The list of observations. Each observation comes from an individual
            Categorical distribution. Shape should be (D, sum(self.num_emission_symbols)).
        :type Ys: np.ndarray
            

        :return: Posterior probabilities over hidden states. Entry [t, i] denotes the
            probability of being in state i at timestep t, given all the data
            :math:`p(X_t = i|Ys, \\theta)`.
        :rtype: np.ndarray
        """

        # start filtering and updating routine
        timesteps = Ys.shape[0]
        x_tm1 : np.ndarray = self.pi
        predictions : List[np.ndarray] = [x_tm1]
        updates : List[np.ndarray] = []

        for t in range(timesteps):
            x_updated = self.update_mv(x_tm1, Ys[t, :])
            x_tm1 = self._predict(x_updated)

            updates.append(x_updated)
            predictions.append(x_tm1)

        smoothed = [updates[-1]]
        x_tp1 = updates[-1]

        for t in range(len(Ys) - 2, -1, -1):
            # print(f"t: {t}, x_tp1: {x_tp1}")
            x_tp1 = self.smooth(t, x_tp1, updates, predictions)
            smoothed.append(x_tp1)

        smoothed = np.array(list(reversed(smoothed)))
        return smoothed

    def smooth(self, t : int, x_tp1 : np.ndarray, updates, predictions) -> np.ndarray:
        """
        Perform smoothing to estimate the hidden state probabilities at time t.

        :param t: The timestep for which smoothing is performed.
        :type t: int

        :param x_tp1: The posterior probability distribution over hidden states at time t+1.
        :type x_tp1: np.ndarray

        :param updates: List containing the filtered posterior probability distributions for each timestep.
        :type updates: List[np.ndarray]

        :param predictions: List containing the predicted posterior probability distributions for each timestep.
        :type predictions: List[np.ndarray]

        :return: The smoothed posterior probability distribution over hidden states at time t.
        :rtype: np.ndarray
        """
        filtered = updates[t]
        predicted = predictions[t + 1]
        u = self.A @ (x_tp1 / predicted)
        ret = filtered * u

        #print(f"Filtered: {filtered}, Predicted : {predicted}, A@(x_tp1/pred) : {u}, Ret: {ret}")
        return ret
    
    def likelihood(self, y_i : np.ndarray) -> np.ndarray:
        """
        Calculate the conditional likelihood p(Y_t|X_t) of an observation sequence for each hidden state.

        Calculate the conditional likelihood p(Y_t|X_t) of an observation sequence for each hidden state.

        :param y_i: Observation sequence for the emission matrices.
        :type y_i: np.ndarray

        :return: The product of likelihoods for the observation sequence for each hidden state.
        :rtype: np.ndarray

        :notes: 
            This method calculates the likelihood of an observation sequence for each hidden state.
            The observation sequence corresponds to the emission matrices.

            The likelihood is computed by taking the product of the probabilities of each observation
            given the emission matrix for the corresponding hidden state.

        """
        sections = np.insert(np.cumsum(self.num_emission_symbols)[:-1], 0, 0)
        indeces = y_i + sections
        
        arrs = self.Bs[:, indeces]

        return np.prod(arrs, axis=1).squeeze()

    def update_mv(self, x_tm1 : np.ndarray, y_t : Sequence[int]) -> np.ndarray:

        """
        Update the hidden state distribution based on the current observation.

        :param x_tm1: Vector of hidden state distribution from timestep t-1.
        :type x_tm1: np.ndarray

        :param y_t: Sequence of integers, where each integer at index i corresponds to the observation
            for the emission matrix at index i.
        :type y_t: Sequence[int]

        :return: The updated posterior probability distribution over hidden states at time t.
        :rtype: np.ndarray

        :notes:
            This method updates the hidden state distribution at time t based on the current observation.

            The update is performed by multiplying the likelihood of the observation given each hidden state
            (calculated using the emission matrices) with the prior probability distribution over hidden states
            from the previous timestep. The result is the unnormalized posterior probability distribution over
            hidden states at time t.

            In this example, 'result' would contain the updated posterior probability distribution over
            hidden states at the current timestep based on the input observation sequence.
        """

        # likelihood p(y_t | x_t) factorizes into p(y^1_t|x_t) * p(y^2_t|x_t) * ... *  p(y^K_t|x_t)
        likelihoods = self.likelihood(y_t)
        prior = x_tm1
        
        posterior_unnormalized = likelihoods * prior

        # possible underflow -> we'll fix it later on
        return posterior_unnormalized / posterior_unnormalized.sum()

    def _predict(self, x_tm1 : np.ndarray) -> np.ndarray:
        """
        Predict the next hidden state probabilities based on the current state.

        :param x_tm1: The probability distribution over hidden states at time t-1.
        :type x_tm1: np.ndarray

        :return: The predicted probability distribution over hidden states at time t, given all prior observations.
        :rtype: np.ndarray

        :raises AssertionError: If the input x_tm1 is not a valid probability distribution (sum is not close to 1).

        :notes:
            This method calculates the predicted probability distribution over hidden states
            at the next timestep (t) based on the probability distribution at the current timestep (t-1).

            The prediction is performed by applying the state transition probability matrix (A) to the
            probability distribution at time t-1. The result is the predicted probability distribution
            over hidden states at time t.
        """
        #check if x_tm1 is a valid probability distribution
        assert np.isclose(x_tm1.sum(), 1), "x_tm1 isn't stochastic, sum != 1"

        return self.A.T @ np.atleast_1d(x_tm1)

    def viterbi(self, Ys : np.ndarray) -> Sequence[int]:
        """
        Viterbi algorithm for multivariate discrete emissions.

        :param Ys: nd array containing observations. Shape: (N, D)
        :type Ys: np.ndarray

        :notes: Observations are of shape (N, D) where D is the number of emission signals.
            The ordering of these signals MUST correspond to the ordering in the list of emission matrices.
        """

        # shape: (num_states, num_observations)
        timesteps : int = Ys.shape[0]
        
        delta : np.ndarray = np.zeros(shape=(self.num_states, timesteps))
        psi : np.ndarray = np.zeros_like(delta)

        # initialization
        delta[:, 0] = self.pi * self.likelihood(Ys[0, :])

        # recursion/iteration
        for t in range(1, timesteps):
            likelihood = self.likelihood(Ys[t, :])
            transitions = np.diag(delta[:, t-1]) @ self.A
            max_i = np.argmax(transitions, axis=0)
            max_v = np.max(transitions, axis=0)
            psi[:, t] = max_i
            delta[:, t] = max_v * likelihood

        # termination
        q_T : int = np.argmax(delta[:, -1], keepdims=False)
        state_sequence_reversed : Sequence[int] = [q_T]

        q_prime : int = q_T
        for t in range(timesteps - 2, -1, -1):
            q_prime = int(psi[q_prime, t + 1])
            state_sequence_reversed.append(q_prime)

        return list(reversed(state_sequence_reversed))

    def fit(self, YYs : List[np.ndarray]):
        
        """
            Implementation of the Baum-Welch or EM-Algorithm for 1) multiple observations of varying lengths
            and 2) multiple categorically distributed emission signals. Fit the parameters pi, A and B_1, ..., B_K of a HMM.

            See https://stephentu.github.io/writeups/hmm-baum-welch-derivation.pdf or the provided
            jupyter notebook `hmmstudy.ipynb` for derivation.
        """
        NUM_ITER = 10
        for cur_i in tqdm(range(NUM_ITER)):

            # calculate forward/backward/gamma variables
            fl : List[np.ndarray] = self.forward_lattice(YYs) # List of np.ndarrays of shape (num_states, num_timesteps)
            bl : np.ndarray = self.backward_lattice(YYs) # List of np.ndarrays of shape (num_states, num_timesteps)
            gl : np.ndarray = self.gamma_lattice(YYs) # List of np.ndarrays of shape (num_states, num_timesteps)

            #print(f'fl: {fl}\n\nbl: {bl}\n\ngl: {gl}\n\n')
            # ---------------- pi update
            pi_hat = np.concatenate([gl_i[0, :][:, None] for gl_i in gl], axis=1).sum(axis=1)


            # ---------------- A update
            # keep track of different sequences
            Xis = np.empty(shape=(len(YYs), self.num_states, self.num_states))

            # loop over different observation sequences
            for ys_i, Ys in enumerate(YYs):
                
                xi = np.empty(shape=(len(Ys), self.num_states, self.num_states))
                for t in range(len(Ys) - 1):
                    alpha_t = fl[ys_i][:, t]
                    beta_t = bl[ys_i][:, t+1] * self.likelihood(Ys[t])
                    xi_t = np.outer(alpha_t, beta_t) * self.A # shape (num_states, num_states)

                    # normalize
                    xi_t /= xi_t.sum(axis=(0, 1))
                    xi[t, :, :] = xi_t
                
                # sum over all timesteps
                Xis[ys_i, :, :] = xi.sum(axis=0)
            #print(f'Xis: {Xis}')            
            # sum out all observations, then normalize
            A_hat = Xis.sum(axis=0)
            #print(f'A_hat unnormalized: {A_hat}')
            A_hat /= A_hat.sum(axis=1)[:, None]

            # ---------------- Bs update
            indices = np.cumsum(self.num_emission_symbols)[:-1]
            Bs = np.split(self.Bs, indices_or_sections=indices, axis=1)
            #print(f'Bs : {Bs}')
            Bs_buffer = []

            for i_emission, B in enumerate(Bs):
                num_symbols = B.shape[1]
                buffer_B = np.empty(shape=(len(YYs), self.num_states, num_symbols))
                

                # Ys is observation of shape (num_obs, num_emissions) where each emission is categorically distributed.
                for ys_i, Ys in enumerate(YYs):

                    # collect the gamma lattice corresponding to the current observation Ys
                    gamma_ys_i = gl[ys_i].T # shape (num_states, timesteps)
                    obs = np.array(Ys[:, i_emission]) # shape (timesteps,)

                    # loop over all individual emission symbols of emission i_emission
                    for symbol in range(num_symbols):

                        # binary mask of where the observation is equal to the symbol
                        mask = np.array(symbol == obs).astype(np.int64)[None, :]
                        _2dmask = np.repeat(mask, self.num_states, axis=0)

                        # mask out gamma_bi where the symbol was observed. Sum out timesteps
                        update_symbol = (gamma_ys_i * _2dmask).sum(axis=1) # shape (num_states,)

                        # For symbol=0, the update_symbol carries b_{i, 0},
                        # which is a column vector of the corresponding B matrix
                        
                        # normalize 
                        update_symbol /= gamma_ys_i.sum(axis=1)
                        buffer_B[ys_i, :, symbol] = update_symbol.flatten()

                B_update = buffer_B.sum(axis=0)
                #print(f'bupdate : {B_update}')
                B_update /= B_update.sum(axis=1)[:, None]
                #print(f'nomralized bupdate : {B_update}') 
                Bs_buffer.append(B_update)
                #print(f'Bs_buffer : {Bs_buffer}')

            Bs_hat = np.concatenate(Bs_buffer, axis=1)

            # TODO: check convergence in parameter space and on likelihood
            #print(f'pi_hat : {pi_hat}\n\nA_hat: {A_hat}\n\nBs_hat: {Bs_hat}\n\n')
            self.pi = pi_hat
            self.A = A_hat
            self.Bs = Bs_hat


    def forward_lattice(self, YYs : np.ndarray):
        """
            forward computation of :math:`\\alpha_{r}` where :math:`r` runs over all observed sequences.
            :math:`\\alpha_{ri}(t) = b_{i}(y_{t})\\sum_j\\alpha_{rj}(t-1) a_{ji}`. In bayesian terms, we're computing
            :math:`\\alpha_{ri}(t) = p(X_t = i| Y^r_1, \\dots, Y^r_t ; \\theta)`

        """

        alphas = []

        for Ys in YYs:
            Ys = np.array(Ys)
            num_timesteps = len(Ys)
            
            lattice = np.empty(shape=(self.num_states, num_timesteps))
            lattice[:, 0] = self.pi * self.likelihood(Ys[0, :])

            for t in range(1, len(Ys)):
                lattice[:, t] = self.A.T @ np.atleast_1d(lattice[:, t-1]) * self.likelihood(Ys[t, 0])
            
            alphas.append(lattice)

        return alphas


    def backward_lattice(self, YYs : np.ndarray):
        
        betas = []

        for Ys in YYs:
            Ys = np.array(Ys)
            num_timesteps = len(Ys)
            
            lattice = np.empty(shape=(self.num_states, num_timesteps))
            lattice[:, -1] = np.ones(shape=(self.num_states,))

            # loop backwards
            for t in range(len(Ys) - 2, -1, -1):
                lattice[:, t] = self.A @ (lattice[:, t+1] * self.likelihood(Ys[t+1, 0]))
            
            betas.append(lattice)

        return betas

    def gamma_lattice(self, YYs : np.ndarray):
        
        gammas = []
        for Ys in YYs:
            gammas.append(self.predict(Ys))
        return gammas








