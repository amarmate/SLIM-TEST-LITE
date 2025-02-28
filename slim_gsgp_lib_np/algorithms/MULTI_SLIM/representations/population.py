# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Population class implementation for evaluating genetic programming trees.
"""
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree_utils import _execute_tree
import numpy as np

class Population:
    def __init__(self, pop):
        """
        Initializes a population of Trees.

        This constructor sets up the population with a list of Tree objects,
        calculating the size of the population and the total node count.

        Parameters
        ----------
        pop : List
            The list of tree Tree objects that make up the population.

        Returns
        -------
        None
        """
        self.population = pop
        self.size = len(pop)
        self.nodes_count = sum(ind.nodes_count for ind in pop)
        self.fit = None

    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for each tree in the population.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for calculating semantics.
        testing : bool, optional
            Boolean indicating if the calculation is for testing semantics.

        Returns
        -------
        None
        """
        # computing the semantics for all the trees in the population
        [
            tree.calculate_semantics(inputs, testing)
            for tree in self.population
        ]

        # computing testing semantics, if applicable
        if testing:
            # setting the population semantics to be a list with all the semantics of all trees
            self.test_semantics = [
                tree.test_semantics for tree in self.population
            ]

        else:
            # setting the population semantics to be a list with all the semantics of all trees
            self.train_semantics = [
                tree.train_semantics for tree in self.population
            ]
