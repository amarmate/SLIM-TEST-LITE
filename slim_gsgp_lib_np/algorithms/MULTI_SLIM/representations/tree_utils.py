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
Utility functions and tree operations for genetic programming.
"""

import random
import numpy as np


def bound_value(vector, min_val, max_val):
    """
    Constrains the values within a specific range.

    Parameters
    ----------
    vector : torch.Tensor
        Input tensor to be bounded.
    min_val : float
        Minimum value for bounding.
    max_val : float
        Maximum value for bounding.

    Returns
    -------
    torch.Tensor
        A Tensor with values bounded between min_val and max_val.
    """
    return np.clip(vector, min_val, max_val)


def flatten(data):
    """
    Flattens a nested tuple structure.

    Parameters
    ----------
    data : tuple
        Input nested tuple data structure.

    Yields
    ------
    object
        Flattened data element by element. If data is not a tuple, returns the original data itself.
    """
    if isinstance(data, tuple):
        for x in data:
            yield from flatten(x)
    else:
        yield data


def create_grow_random_tree(depth, 
                            FUNCTIONS, 
                            TERMINALS, 
                            CONSTANTS, 
                            p_c=0.3,
                            p_t=0.5, 
                            first_call=True):
    """
    Generates a random tree representation using the Grow method with a maximum specified depth.

    Parameters
    ----------
    depth : int
        Maximum depth of the tree to be created.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    TERMINALS_KEYS : list
        Precomputed list of terminal keys.
    CONSTANTS_KEYS : list
        Precomputed list of constant keys.
    FUNCTIONS_KEYS : list
        Precomputed list of function keys.
    p_c : float, optional
        Probability of choosing a constant node. Default is 0.3.
    p_t : float, optional
        Probability of choosing a terminal node. Default is 0.5.
    first_call : bool, optional
        Variable that controls whether the function is being called for the first time. Default is True.

    Returns
    -------
    tuple or str
        The generated tree representation according to the specified parameters.
    """
    
    # If depth is 1 or a terminal is selected and it's not the first call
    if (depth <= 1 or random.random() < p_t) and not first_call:
        if random.random() > p_c:
            return random.choice(list(TERMINALS.keys()))
        else:
            return random.choice(list(CONSTANTS.keys()))
    
    # If a function is selected
    else:
        node = random.choice(list(FUNCTIONS.keys()))
        if FUNCTIONS[node]["arity"] == 2:
            left_subtree = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS,
                                                   p_c, p_t, False)
            right_subtree = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS,
                                                    p_c, p_t, False)
            return (node, left_subtree, right_subtree)
        else:
            left_subtree = create_grow_random_tree(depth - 1, FUNCTIONS, TERMINALS, CONSTANTS,
                                                   p_c, p_t, False)
            return (node, left_subtree)


def create_random_tree(depth_condition, depth_tree, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
                       p_specialist=0.5, p_t=0.5, p_c=0.3):
    """
    Generates a random ensemble tree representing an individual in the GP ensemble.
    
    The tree is structured as a conditional operator:
        (condition_tree, true_branch, false_branch)
    
    - condition_tree: generated using your existing create_grow_random_tree function.
    - true_branch / false_branch: either a further nested conditional tree or, with some probability,
      a specialist selected from SPECIALISTS.
    
    Parameters
    ----------
    depth_condition : int
        Maximum depth for the condition trees.
    depth_tree : int
        Maximum depth for the ensemble trees.
    FUNCTIONS : dict
        Dictionary of function nodes (for condition trees).
    TERMINALS : dict
        Dictionary of terminal nodes (for condition trees).
    CONSTANTS : dict
        Dictionary of constant nodes (for condition trees).
    SPECIALISTS : dict
        Dictionary (or list) of specialist solutions.
    p_specialist : float, optional
        Probability of terminating a branch with a specialist rather than another conditional node.
    p_t : float, optional
        Terminal probability passed to create_grow_random_tree.
    p_c : float, optional
        Constant probability passed to create_grow_random_tree.
    
    Returns
    -------
    tuple or str
        A conditional tree or a specialist (when the branch is terminated).
    """
    # Base case: if depth is 0 or by chance we decide to return a specialist
    if depth_tree <= 0 or random.random() < p_specialist:
        return random.choice(list(SPECIALISTS.keys()))
    
    # Generate a condition tree.
    # Here we choose a random depth for the condition tree (at most the current depth)
    condition_tree = create_grow_random_tree(depth_condition, FUNCTIONS, TERMINALS, CONSTANTS,
                                             p_c=p_c, p_t=p_t, first_call=True)
    
    # Recursively build the true and false branches.
    true_branch = create_random_tree(depth_condition, depth_tree - 1, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
                                     p_specialist=p_specialist, p_t=p_t, p_c=p_c)
    false_branch = create_random_tree(depth_condition, depth_tree - 1, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
                                      p_specialist=p_specialist, p_t=p_t, p_c=p_c)
    
    return (condition_tree, true_branch, false_branch)


def initializer(init_pop_size, 
                depth_condition, 
                depth_tree, 
                FUNCTIONS, 
                TERMINALS, 
                CONSTANTS,
                SPECIALISTS,
                p_c=0.3, 
                p_t=0.5, 
                p_specialist=0.5,
                **kwargs):
    """
    Generates a list of individuals with random ensemble trees for a GP population.
    
    The individuals are binned along two axes:
      - Condition tree depth: from 0 to depth_condition (inclusive).
      - Ensemble tree depth: from 1 to depth_tree.
    
    For each (condition, ensemble) bin, four modes are created by adjusting the probabilities:
      1. grow–grow: condition: p_t, p_c as given; ensemble: p_specialist as given.
      2. full (condition)–grow: condition: p_t = 0, p_c = 0; ensemble: p_specialist as given.
      3. grow–full: condition: p_t, p_c as given; ensemble: p_specialist = 0.
      4. full–full: condition: p_t = 0, p_c = 0; ensemble: p_specialist = 0.
    
    Parameters
    ----------
    init_pop_size : int
        Total number of individuals to generate.
    depth_condition : int
        Maximum depth for condition trees. (A minimum of 2 is enforced.)
        For example, if 5 is passed then condition depths 0,1,...,5 will be used.
    depth_tree : int
        Maximum ensemble tree depth. If depth_tree <= 0, only specialists will be chosen.
    FUNCTIONS : dict
        Dictionary of allowed function nodes.
    TERMINALS : dict
        Dictionary of allowed terminal symbols.
    CONSTANTS : dict
        Dictionary of allowed constant values.
    SPECIALISTS : dict
        Dictionary of specialist individuals.
    p_c : float, optional
        Constant probability for tree creation (default: 0.3).
    p_t : float, optional
        Terminal probability for tree creation (default: 0.5).
    p_specialist : float, optional
        Specialist termination probability for ensemble tree creation (default: 0.5).
    
    Returns
    -------
    list
        A list of tree representations (each either a conditional tree or a specialist)
        forming the initial population.
    """
    # Enforce a minimum condition depth of 2.
    if depth_condition < 2:
        depth_condition = 2

    # If depth_tree <= 0, then no ensemble tree can be built: return specialists.
    if depth_tree <= 0:
        return [random.choice(list(SPECIALISTS.keys())) for _ in range(init_pop_size)]
    
    population = []
    num_condition_bins = depth_condition - 1  # condition depths: 2 ... depth_condition
    num_ensemble_bins = depth_tree            # ensemble depths: 1 ... depth_tree

    # Define four modes by fixing the probabilities:
    # Each mode is a tuple: (condition_p_t, condition_p_c, ensemble_p_specialist)
    modes = [
         (p_t,      p_c,      p_specialist),  # grow–grow
         (0,        p_c,        p_specialist),  # full (condition)–grow
         (p_t,      p_c,      0),             # grow–full
         (0,        p_c,        0)              # full–full
    ]
    total_bins = num_condition_bins * num_ensemble_bins * len(modes)
    inds_per_bin = init_pop_size // total_bins

    # Loop over all bins.
    for cond_depth in range(2, depth_condition + 1):
        for ens_depth in range(1, depth_tree + 1):
            for mode in modes:
                count = 0 
                cond_p_t, cond_p_c, ens_p_specialist = mode
                for _ in range(inds_per_bin):
                    tree = create_random_tree(
                        cond_depth, ens_depth, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
                        p_specialist=ens_p_specialist, p_t=cond_p_t, p_c=cond_p_c
                    )
                    population.append(tree)
                    count += 1
                print(count)

    
    # If there are still fewer individuals than desired, fill with trees at maximum depths using default probabilities.
    while len(population) < init_pop_size:
        tree = create_random_tree(
            depth_condition, depth_tree, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS,
            p_specialist=p_specialist, p_t=p_t, p_c=p_c
        )
        population.append(tree)
    
    return population[:init_pop_size]


def tree_depth_and_nodes(FUNCTIONS, SPECIALISTS):
    """
    Returns a function that calculates three measures for a tree:
      1. Depth: length of the longest path from the root to a leaf.
      2. Node count: the total number of nodes in the tree (for function nodes, each is counted;
         ensemble nodes themselves are not counted as extra nodes).
      3. Ensemble nodes total: the sum of the node counts from the specialist trees used as terminals.
         For a specialist terminal, its individual node count is used (accessed via SPECIALISTS[key].nodes_count);
         non-specialist terminals contribute 0.

    The tree may be:
      - A function node (created by your grow method), represented as a tuple whose first element is a key in FUNCTIONS.
      - An ensemble (conditional) node, represented as a 3-tuple (condition, true_branch, false_branch)
        where the condition is a function tree and the branches are either ensemble nodes or terminals.
      - A terminal (string). If the string is a key in SPECIALISTS, its ensemble total is taken from the individual.
    
    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of function nodes allowed in the tree. Each function has an "arity" entry.
    SPECIALISTS : dict
        Dictionary of specialist individuals. For a specialist terminal key (e.g. "S_0"), its ensemble
        node count is retrieved by SPECIALISTS[key].nodes_count.
    
    Returns
    -------
    Callable
        A function that, given a tree, returns a triple (depth, node_count, ensemble_nodes_total).
    """
    def depth_and_nodes(tree, count_ensemble):
        # Base case: terminal node.
        if not isinstance(tree, tuple):
            if tree in SPECIALISTS:
                # Depth and node count are 1 for the terminal,
                # but the ensemble total comes from the specialist's own nodes_count.
                return 1, 1, SPECIALISTS[tree].nodes_count
            else:
                # Regular terminal (variable or constant): count as 1 node and depth 1, but no ensemble total.
                return 1, 1, 0

        # If the node is a function node (its first element is a key in FUNCTIONS)
        if isinstance(tree[0], str) and tree[0] in FUNCTIONS:
            arity = FUNCTIONS[tree[0]]["arity"]
            if arity == 2:
                d_left, n_left, ens_left = depth_and_nodes(tree[1], True)
                d_right, n_right, ens_right = depth_and_nodes(tree[2], True)
                depth_val = 1 + max(d_left, d_right)
                nodes_val = 1 + n_left + n_right
                ens_total = ens_left + ens_right
                return depth_val, nodes_val, ens_total
            elif arity == 1:
                d_child, n_child, ens_child = depth_and_nodes(tree[1], True)
                return 1 + d_child, 1 + n_child, ens_child
            else:
                # If the function node is of arity 0, treat it as a leaf.
                return 1, 1, 0

        else:
            # Otherwise, assume it's an ensemble (conditional) node: (condition, true_branch, false_branch)
            if len(tree) != 3:
                raise ValueError("Invalid ensemble tree structure. Expected a tuple of length 3.")
            d_cond, n_cond, ens_cond = depth_and_nodes(tree[0], False)
            d_true, n_true, ens_true = depth_and_nodes(tree[1], False)
            d_false, n_false, ens_false = depth_and_nodes(tree[2], False)
            # For depth, take the max depth among children; if this ensemble operator is counted, add 1.
            d = max(d_cond, d_true, d_false)
            if count_ensemble:
                d += 1
            # For node count, simply sum the node counts of the children (ensemble operator not counted as an extra node).
            n = n_cond + n_true + n_false
            # For ensemble total, sum the ensemble totals of the children.
            ens_total = ens_cond + ens_true + ens_false
            return d, n, ens_total

    return lambda tree: depth_and_nodes(tree, True)

def _execute_gp_tree(repr_, X, FUNCTIONS, TERMINALS, CONSTANTS):
    if isinstance(repr_, tuple):  # If it's a function node
        function_name = repr_[0]
        if FUNCTIONS[function_name]["arity"] == 2:
            left_subtree, right_subtree = repr_[1], repr_[2]
            left_result = _execute_gp_tree(left_subtree, X, FUNCTIONS, TERMINALS,
                                        CONSTANTS)  # equivalent to Tree(left_subtree).apply_tree(inputs) if no parallelization were used
            right_result = _execute_gp_tree(right_subtree, X, FUNCTIONS, TERMINALS,
                                         CONSTANTS)  # equivalent to Tree(right_subtree).apply_tree(inputs) if no parallelization were used
            output = FUNCTIONS[function_name]["function"](
                left_result, right_result
            )
        else:
            left_subtree = repr_[1]
            left_result = _execute_gp_tree(left_subtree, X, FUNCTIONS, TERMINALS,
                                        CONSTANTS)  # equivalent to Tree(left_subtree).apply_tree(inputs) if no parallelization were used
            output = FUNCTIONS[function_name]["function"](left_result)

        return bound_value(output, -1e12, 1e12)

    else:  # If it's a terminal node
        if repr_ in TERMINALS:
            return X[:, TERMINALS[repr_]]
        elif repr_ in CONSTANTS:
            return np.full((X.shape[0],), CONSTANTS[repr_](None))


def _execute_tree(repr_, X, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS, testing=False, predict=False):
    """
    Evaluates a tree representation that may include ensemble (conditional) nodes.
    
    Ensemble nodes are represented as a 3-tuple:
         (condition, branch_if_true, branch_if_false)
    For the condition part, we use _execute_gp_tree to compute its semantics and then 
    create a mask (condition > 0). The mask is used to select, for each sample in X, which 
    branch to follow.
    
    Parameters
    ----------
    repr_ : tuple or str
        The tree representation. This may be:
          - A function node (tuple with the first element a key in FUNCTIONS),
          - An ensemble node (tuple with first element NOT in FUNCTIONS), or
          - A terminal (string).
    X : np.ndarray
        Input data samples (rows correspond to samples).
    FUNCTIONS : dict
        Dictionary of allowed functions for GP trees (each with an "arity" and "function").
    TERMINALS : dict
        Dictionary mapping terminal symbols to their corresponding column indices in X.
    CONSTANTS : dict
        Dictionary mapping constant symbols to constant-producing functions.
    SPECIALISTS : dict
        Dictionary mapping specialist keys to specialist individuals. Each specialist must have:
           - The precomputed attributes `train_semantics` and `test_semnatics`, and
           - A method `predict(X)` to compute semantics on new data.
    testing : bool, optional
        If True, use the specialist's precomputed test semantics. Else, use the train semantics.
    predict : bool, optional
        If True, use the specialist's predict method to compute semantics on new data.
    
    Returns
    -------
    np.ndarray
        A NumPy array of semantics for each sample in X.
    """
    # Check for ensemble (conditional) node:
    # We assume an ensemble node is a tuple whose first element is not a key in FUNCTIONS.
    if isinstance(repr_, tuple) and not (isinstance(repr_[0], str) and repr_[0] in FUNCTIONS):
        # Evaluate the condition using _execute_gp_tree.
        condition_semantics = _execute_gp_tree(repr_[0], X, FUNCTIONS, TERMINALS, CONSTANTS)
        # Create a Boolean mask: for each sample, True if condition > 0.
        mask = condition_semantics > 0
        
        # Evaluate the true and false branches recursively.
        true_branch = _execute_tree(repr_[1], X, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS, testing)
        false_branch = _execute_tree(repr_[2], X, FUNCTIONS, TERMINALS, CONSTANTS, SPECIALISTS, testing)
        # Combine the results: for each sample, pick the branch based on the mask.
        return np.where(mask, true_branch, false_branch)
    
    # Otherwise, if it is a standard function node, use _execute_gp_tree.
    if isinstance(repr_, tuple) and (isinstance(repr_[0], str) and repr_[0] in FUNCTIONS):
        return _execute_gp_tree(repr_, X, FUNCTIONS, TERMINALS, CONSTANTS)
    
    # Terminal node: check terminals, constants, and specialists.
    if not isinstance(repr_, tuple):
        if repr_ in TERMINALS:
            return X[:, TERMINALS[repr_]]
        elif repr_ in CONSTANTS:
            return np.full((X.shape[0],), CONSTANTS[repr_](None))
        elif repr_ in SPECIALISTS:
            if testing:
                return SPECIALISTS[repr_].test_semantics
            elif not predict:
                return SPECIALISTS[repr_].train_semantics
            else: 
                return SPECIALISTS[repr_].predict(X)
        else:
            raise ValueError("Unknown terminal symbol: " + str(repr_))
        








# --------------------------------------------- NOT IMPLEMENTED ---------------------------------------------
def random_subtree(FUNCTIONS):
    """
    Creates a function that selects a random subtree from a given tree representation.

    This function generates another function that traverses a tree representation to randomly
    select a subtree based on the arity of the functions within the tree.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    Callable
        A function ('random_subtree_picker') that selects a random subtree from the given tree representation.

        This function navigates the tree representation recursively, choosing a subtree based on
        probabilities determined by the overall representation of the tree.

        Parameters
        ----------
        tree : tuple
            The tree representation from which to select a subtree.
        first_call : bool, optional
            Indicates whether this is the initial call to the function. Defaults to True.
        num_of_nodes : int, optional
            The total number of nodes in the tree. Used to calculate probabilities.

        Returns
        -------
        tuple
            The randomly selected subtree (or the original node if not applicable).

    Notes
    -----
    The returned function traverses the tree representation recursively, selecting subtrees based on random
    probabilities influenced by the representation of the tree.
    """
    def random_subtree_picker(tree, first_call=True, num_of_nodes=None):
        """
        Selects a random subtree from the given tree representation.

        This function navigates the tree representation recursively, choosing a subtree based on
        probabilities determined by the overall representation of the tree.

        Parameters
        ----------
        tree : tuple
            The tree representation from which to select a subtree.
        first_call : bool, optional
            Indicates whether this is the initial call to the function. Defaults to True.
        num_of_nodes : int, optional
            The total number of nodes in the tree. Used to calculate probabilities.

        Returns
        -------
        tuple
            The randomly selected subtree (or the original node if not applicable).
        """
        if isinstance(tree, tuple):
            current_number_of_nodes = (
                num_of_nodes if first_call else len(list(flatten(tree)))
            )
            if FUNCTIONS[tree[0]]["arity"] == 2:
                if first_call:
                    subtree_exploration = (
                        1
                        if random.random()
                        < len(list(flatten(tree[1]))) / (current_number_of_nodes - 1)
                        else 2
                    )
                else:
                    p = random.random()
                    subtree_exploration = (
                        0
                        if p < 1 / current_number_of_nodes
                        else (
                            1
                            if p < len(list(flatten(tree[1]))) / current_number_of_nodes
                            else 2
                        )
                    )
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                subtree_exploration = (
                    1
                    if first_call
                    else (0 if random.random() < 1 / current_number_of_nodes else 1)
                )

            if subtree_exploration == 0:
                return tree
            elif subtree_exploration == 1:
                return (
                    random_subtree_picker(tree[1], False)
                    if isinstance(tree[1], tuple)
                    else tree[1]
                )
            elif subtree_exploration == 2:
                return (
                    random_subtree_picker(tree[2], False)
                    if isinstance(tree[2], tuple)
                    else tree[2]
                )
        else:
            return tree

    return random_subtree_picker


def substitute_subtree(FUNCTIONS):
    """
    Generates a function that substitutes a specific subtree in a tree representation with a new subtree.

    This function returns another function that can recursively traverse a tree representation to replace
    occurrences of a specified subtree with a new one, maintaining the representation and
    validity of the original tree.

    Parameters
    ----------
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.

    Returns
    -------
    Callable
        A function ('substitute') that substitutes a specified subtree within the given tree representation with a new subtree.

        This function recursively searches for occurrences of the target subtree within the tree
        representation and replaces it with the new subtree when found. If the original tree
        representation is a terminal or equal to the new one, return it.

        Parameters
        ----------
        tree : tuple or str
            The tree representation in which to perform the substitution. Can be a terminal.
        target_subtree : tuple or str
            The subtree to be replaced.
        new_subtree : tuple or str
            The subtree to insert in place of the target subtree.

        Returns
        -------
        tuple
            The modified tree representation with the target subtree replaced by the new subtree.
        str
            The new tree leaf node if the original is a leaf.

    Notes
    -----
    The returned function performs replacements while preserving the tree structure based on
    the arity of the function nodes.
    """

    def substitute(tree, target_subtree, new_subtree):
        """
        Substitutes a specified subtree within the given tree representation with a new subtree.

        This function recursively searches for occurrences of the target subtree within the tree
        representation and replaces it with the new subtree when found. If the original tree
        representation is a terminal or equal to the new one, return it.

        Parameters
        ----------
        tree : tuple or str
            The tree representation in which to perform the substitution. Can be a terminal.
        target_subtree : tuple or str
            The subtree to be replaced.
        new_subtree : tuple or str
            The subtree to insert in place of the target subtree.

        Returns
        -------
        tuple
            The modified tree representation with the target subtree replaced by the new subtree.
        str
            The new tree leaf node if the original is a leaf.
        """
        if tree == target_subtree:
            return new_subtree
        elif isinstance(tree, tuple):
            if FUNCTIONS[tree[0]]["arity"] == 2:
                return (
                    tree[0],
                    substitute(tree[1], target_subtree, new_subtree),
                    substitute(tree[2], target_subtree, new_subtree),
                )
            elif FUNCTIONS[tree[0]]["arity"] == 1:
                return tree[0], substitute(tree[1], target_subtree, new_subtree)
        else:
            return tree

    return substitute


def tree_pruning(TERMINALS, CONSTANTS, FUNCTIONS, p_c=0.3):
    """
    Generates a function that reduces both sides of a tree representation to a specific depth.

    This function returns another function that can prune a given tree representation to a
    specified depth by replacing nodes with terminals or constants based on a defined probability.

    Parameters
    ----------
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree.
    FUNCTIONS : dict
        Dictionary of functions allowed in the tree.
    p_c : float, optional
        Probability of choosing a constant node. Default is 0.3.

    Returns
    -------
    Callable
        A function ('pruning') that prunes the given tree representation to the specified depth.

        This function replaces nodes in the tree representation with terminals or constants
        if the target depth is reached, ensuring the tree representation does not exceed the
        specified depth.

        Parameters
        ----------
        tree : tuple or str
            The tree representation to be pruned.
        target_depth : int
            The depth to which the tree representation should be pruned.

        Returns
        -------
        tuple
            The pruned tree representation, which may consist of terminals, constants, or
            a modified subtree.
        str
            The pruned tree if it is a leaf.
    """
    def pruning(tree, target_depth):
        """
        Prunes the given tree representation to the specified depth.

        This function replaces nodes in the tree representation with terminals or constants
        if the target depth is reached, ensuring the tree representation does not exceed the
        specified depth.

        Parameters
        ----------
        tree : tuple or str
            The tree representation to be pruned.
        target_depth : int
            The depth to which the tree representation should be pruned.

        Returns
        -------
        tuple
            The pruned tree representation, which may consist of terminals, constants, or
            a modified subtree.
        str
            The pruned tree if it is a leaf.
        """
        if target_depth <= 1 and tree not in TERMINALS:
            return (
                np.random.choice(list(TERMINALS.keys()))
                if random.random() > p_c
                else np.random.choice(list(CONSTANTS.keys()))
            )
        elif not isinstance(tree, tuple):
            return tree
        if FUNCTIONS[tree[0]]["arity"] == 2:
            new_left_subtree = pruning(tree[1], target_depth - 1)
            new_right_subtree = pruning(tree[2], target_depth - 1)
            return tree[0], new_left_subtree, new_right_subtree
        elif FUNCTIONS[tree[0]]["arity"] == 1:
            new_left_subtree = pruning(tree[1], target_depth - 1)
            return tree[0], new_left_subtree

    return pruning
