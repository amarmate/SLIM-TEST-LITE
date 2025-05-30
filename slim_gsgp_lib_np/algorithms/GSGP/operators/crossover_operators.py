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
Geometric crossover implementation for genetic programming trees.
"""
import numpy as np

def geometric_crossover(tree1, tree2, random_tree, testing, new_data=False):
    """
    Performs geometric crossover between two trees using a random tree.

    Parameters
    ----------
    tree1 : Tree or np.ndarray
        The first parent tree. If geometric_crossover is called with new_data=True, it means the final tree is being
        evaluated on testing data and tree1 is a np.ndarray. Otherwise, during training, the individuals
        are Tree instances.
    tree2 : Tree or np.ndarray
        The second parent tree. If geometric_crossover is called with new_data=True, it means the final tree is being
        evaluated on testing data and tree2 is a np.ndarray. Otherwise, during training, the individuals
        are Tree instances.
    random_tree : Tree or np.ndarray
        The random tree used for crossover. If geometric_crossover is called with new_data=True, it means the
        final tree is being evaluated on testing data and random_tree is a np.ndarray. Otherwise, during training,
        random_tree is a Tree instance.
    testing : bool
        Flag indicating whether to use test semantics or train semantics.
    new_data : bool
        Flag indicating whether the trees are exposed to new data, outside the evolution process. In this case,
        operations are performed on the inputs rather than semantics.

    Returns
    -------
    np.ndarray
        The semantics of the individual resulting from geometric crossover.
    """
    # If new (testing) data is used (for the testing of the final tree), return semantics resulting from crossover
    if new_data:
        return np.add(
            np.multiply(tree1, random_tree),
            np.multiply(np.subtract(1, random_tree), tree2),
        )
    # If new_data is False, geometric_crossover is being called during GSGP's training phase
    else:
        if testing:
            return np.add(
                np.multiply(tree1.test_semantics, random_tree.test_semantics),
                np.multiply(np.subtract(1, random_tree.test_semantics), tree2.test_semantics),
            )
        else:
            return np.add(
                np.multiply(tree1.train_semantics, random_tree.train_semantics),
                np.multiply(np.subtract(1, random_tree.train_semantics), tree2.train_semantics),
            )
