from src.hmm import MultiCatEmissionHMM
from scipy.special import softmax
from scipy.optimize import minimize
from abc import ABC, abstractmethod

import numpy as np

class BaseOptimizer(ABC):

    def __init__(self, hmm : MultiCatEmissionHMM) -> None:
        self.params = hmm.get_params()
    
    @abstractmethod
    def optimize(Ys : np.ndarray):
        pass

class ExpectationMaximization(BaseOptimizer):

    def __init__(self, hmm: MultiCatEmissionHMM) -> None:
        super().__init__(hmm)

class NumericalOptimizer(BaseOptimizer):

    def __init__(self,  hmm : MultiCatEmissionHMM) -> None:
        super().__init__(hmm)