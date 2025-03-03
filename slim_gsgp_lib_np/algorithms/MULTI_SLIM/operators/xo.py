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
Crossover operator implementation.
"""

from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree import Tree
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree_utils import (get_all_branches,get_subtree,replace_subtree)
import random


def crossover(ind1, ind2):
    """
    Perform subtree crossover between two individuals.
    
    The operator uses the following rules:
      - If both individuals are trees (tuples), randomly select a branch
        in each and swap them.
      - If one individual is a terminal (non-tuple) and the other is a tree,
        randomly select a branch from the tree, graft the terminal into that spot,
        and return the removed branch as the other offspring.
      - If both individuals are terminals, no crossover is performed.
    
    Parameters
    ----------
    ind1 : any
        The first individual (tree or terminal).
    ind2 : any
        The second individual (tree or terminal).
    
    Returns
    -------
    tuple
        A pair of offspring resulting from the crossover.
    
    Example
    -------
    >>> ind_1 = 'S_2'
    >>> ind_2 = (cond1, (cond2, 'S_1', 'S_4'), (cond3, 'S_4', 'S_1'))
    >>> offs1, offs2 = crossover(ind_1, ind_2)
    # offs1 becomes (cond2, 'S_1', 'S_4')
    # offs2 becomes (cond1, 'S_2', (cond3, 'S_4', 'S_1'))
    """
    # Both individuals are trees.
    if isinstance(ind1, tuple) and isinstance(ind2, tuple):
        branches1 = get_all_branches(ind1)
        branches2 = get_all_branches(ind2)
        if branches1 and branches2:
            path1 = random.choice(branches1)
            path2 = random.choice(branches2)
            subtree1 = get_subtree(ind1, path1)
            subtree2 = get_subtree(ind2, path2)
            offspring1 = replace_subtree(ind1, path1, subtree2)
            offspring2 = replace_subtree(ind2, path2, subtree1)
        else:
            offspring1, offspring2 = ind1, ind2

    # First individual is a tree and second is terminal.
    elif isinstance(ind1, tuple) and not isinstance(ind2, tuple):
        branches1 = get_all_branches(ind1)
        if branches1:
            path = random.choice(branches1)
            subtree = get_subtree(ind1, path)
            offspring1 = subtree
            offspring2 = replace_subtree(ind1, path, ind2)
        else:
            offspring1, offspring2 = ind1, ind2

    # First individual is terminal and second is a tree.
    elif not isinstance(ind1, tuple) and isinstance(ind2, tuple):
        branches2 = get_all_branches(ind2)
        if branches2:
            path = random.choice(branches2)
            subtree = get_subtree(ind2, path)
            offspring1 = subtree
            offspring2 = replace_subtree(ind2, path, ind1)
        else:
            offspring1, offspring2 = ind1, ind2

    # Both are terminals; no crossover is possible.
    else:
        offspring1, offspring2 = ind1, ind2

    return Tree(offspring1), Tree(offspring2)

























































# ----------------------------------------------------------------------------------------------------------------------------

def crossover_trees(FUNCTIONS):
    """
    Returns a function that performs crossover between two tree representations.

    To avoid passing the FUNCTIONS parameter unnecessarily, a new function is created utilizing it. This function is
    returned and passed as a parameter to the GP algorithm, where it is then called when crossover is performed.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of allowed functions in the trees.

    Returns
    -------
    Callable
        A function (`inner_xo`) that performs crossover between two tree representations.
        Inner function to perform crossover between two trees.

        Parameters
        ----------
        tree1 : tuple
            The first tree representation.
        tree2 : tuple
            The second tree representation.
        tree1_n_nodes : int
            Number of nodes in the first tree representation.
        tree2_n_nodes : int
            Number of nodes in the second tree representation.

        Returns
        -------
        tuple
            Two new tree representations after performing crossover.
        Notes
        -----
        This function selects random crossover points from both `tree1` and `tree2` and swaps
        their subtrees at those points. If either tree is a terminal node, it returns the tree
        representations unchanged.

    Notes
    -----
    The returned function (`inner_xo`) takes two tree representations and their node counts,
    selects random subtrees, and swaps them to create the representations of the new offspring trees.
    """
    # getting the function to substitute a subtree in a tree
    subtree_substitution = substitute_subtree(FUNCTIONS=FUNCTIONS)
    # getting the random subtree selection function
    random_subtree_picker = random_subtree(FUNCTIONS=FUNCTIONS)

    def inner_xo(tree1, tree2, tree1_n_nodes, tree2_n_nodes):
        """
        Performs crossover between two tree representations.
        Inner function to perform crossover between two trees.

        Parameters
        ----------
        tree1 : tuple
            The first tree representation.
        tree2 : tuple
            The second tree representation.
        tree1_n_nodes : int
            Number of nodes in the first tree representation.
        tree2_n_nodes : int
            Number of nodes in the second tree representation.

        Returns
        -------
        tuple
            Two new tree representations after performing crossover.
        Notes
        -----
        This function selects random crossover points from both `tree1` and `tree2` and swaps
        their subtrees at those points. If either tree is a terminal node, it returns the tree
        representations unchanged.
        """
        if isinstance(tree1, tuple) and isinstance(tree2, tuple):
            # Randomly select crossover points in both trees
            crossover_point_tree1 = random_subtree_picker(
                tree1, num_of_nodes=tree1_n_nodes
            )
            crossover_point_tree2 = random_subtree_picker(
                tree2, num_of_nodes=tree2_n_nodes
            )

            # Swap subtrees at the crossover points
            new_tree1 = subtree_substitution(
                tree1, crossover_point_tree1, crossover_point_tree2
            )
            new_tree2 = subtree_substitution(
                tree2, crossover_point_tree2, crossover_point_tree1
            )

            return new_tree1, new_tree2
        else:
            # If either tree1 or tree2 is a terminal node, return them as they are (no crossover)
            return tree1, tree2

    return inner_xo
