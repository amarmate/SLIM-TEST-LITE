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
Mutation Functions for SLIM GSGP.
"""

import random
import numpy as np
import torch
from slim_gsgp_lib.algorithms.GSGP.representations.tree import Tree
from slim_gsgp_lib.algorithms.SLIM_GSGP.representations.individual import Individual
from slim_gsgp_lib.utils.utils import get_random_tree, swap_sub_tree, get_indices, get_indices_with_levels
from functools import lru_cache

# two tree function
def two_trees_delta(operator="sum"):
    """
    Generate a function for the two-tree delta mutation.

    Parameters
    ----------
    operator : str
        The operator to be used in the mutation ("sum" or other).

    Returns
    -------
    Callable
        A mutation function (`tt_delta`) for two Individuals that returns the mutated semantics.

        Parameters
        ----------
        tr1 : Individual
            The first tree individual.
        tr2 : Individual
            The second tree individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train Individual semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.

    Notes
    -----
    The returned function ('tt_delta_{operator}') takes as input two individuals, the mutation step, a boolean
    representing whether to use the train or test semantics, and returns the calculated semantics of the new individual.
    """

    def tt_delta(tr1, tr2, ms, testing):
        """
        Performs delta mutation between two trees based on their semantics.

        Parameters
        ----------
        tr1 : Individual
            The first tree Individual.
        tr2 : Individual
            The second tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train Individual semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.
        """
        if testing:
            return (
                torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics))
                if operator == "sum"
                else torch.add(
                    1, torch.mul(ms, torch.sub(tr1.test_semantics, tr2.test_semantics))
                )
            )

        else:
            return (
                torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics))
                if operator == "sum"
                else torch.add(
                    1,
                    torch.mul(ms, torch.sub(tr1.train_semantics, tr2.train_semantics)),
                )
            )

    tt_delta.__name__ += "_" + operator

    return tt_delta


def one_tree_delta(operator="sum", sig=False):
    """
    Generate a function for the one-tree delta mutation.

    Parameters
    ----------
    operator : str
        The operator to be used in the mutation ("sum" or other).
    sig : bool
        Boolean indicating if sigmoid should be applied.

    Returns
    -------
    Callable
        A mutation function (`ot_delta`) for one-tree mutation.

        Parameters
        ----------
        tr1 : Individual
            The tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.
    Notes
    -----
    The returned function ('ot_delta_{operator}_{sig}') takes as input one individual, the mutation step,
    a boolean representing whether to use the train or test semantics, and returns the mutated semantics.
    """
    def ot_delta(tr1, ms, testing):
        """
        Performs delta mutation on one tree based on its semantics.

        Parameters
        ----------
        tr1 : Individual
            The tree Individual.
        ms : float
            Mutation step.
        testing : bool
            Flag to indicate whether to use test or train semantics.

        Returns
        -------
        torch.Tensor
            The mutated semantics.
        """
        if sig:
            if testing:
                return (
                    torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1))
                    if operator == "sum"
                    else torch.add(
                        1, torch.mul(ms, torch.sub(torch.mul(2, tr1.test_semantics), 1))
                    )
                )
            else:
                return (
                    torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1))
                    if operator == "sum"
                    else torch.add(
                        1,
                        torch.mul(ms, torch.sub(torch.mul(2, tr1.train_semantics), 1)),
                    )
                )
        else:
            if testing:
                return (
                    torch.mul(
                        ms,
                        torch.sub(
                            1, torch.div(2, torch.add(1, torch.abs(tr1.test_semantics)))
                        ),
                    )
                    if operator == "sum"
                    else torch.add(
                        1,
                        torch.mul(
                            ms,
                            torch.sub(
                                1,
                                torch.div(
                                    2, torch.add(1, torch.abs(tr1.test_semantics))
                                ),
                            ),
                        ),
                    )
                )
            else:
                return (
                    torch.mul(
                        ms,
                        torch.sub(
                            1,
                            torch.div(2, torch.add(1, torch.abs(tr1.train_semantics))),
                        ),
                    )
                    if operator == "sum"
                    else torch.add(
                        1,
                        torch.mul(
                            ms,
                            torch.sub(
                                1,
                                torch.div(
                                    2, torch.add(1, torch.abs(tr1.train_semantics))
                                ),
                            ),
                        ),
                    )
                )

    ot_delta.__name__ += "_" + operator + "_" + str(sig)
    return ot_delta


def inflate_mutation(FUNCTIONS, TERMINALS, CONSTANTS, two_trees=True, operator="sum", single_tree_sigmoid=False, sig=False):
    """
    Generate an inflate mutation function.

    Parameters
    ----------
    FUNCTIONS : dict
        The dictionary of functions used in the mutation.
    TERMINALS : dict
        The dictionary of terminals used in the mutation.
    CONSTANTS : dict
        The dictionary of constants used in the mutation.
    two_trees : bool
        Boolean indicating if two trees should be used.
    operator : str
        The operator to be used in the mutation.
    single_tree_sigmoid : bool
        Boolean indicating if sigmoid should be applied to a single tree.
    sig : bool
        Boolean indicating if sigmoid should be applied.

    Returns
    -------
    Callable
        An inflate mutation function (`inflate`).

        Parameters
        ----------
        individual : Individual
            The tree Individual to mutate.
        ms : float
            Mutation step.
        X : torch.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after mutation (default: True).

        Returns
        -------
        Individual
            The mutated tree Individual.

    Notes
    -----
    The returned function performs inflate mutation on Individuals, using either one or two randomly generated trees
    and applying either delta mutation or sigmoid mutation based on the parameters.
    """
    
    def inflate(
        individual,
        ms,
        X,
        max_depth=8,
        p_c=0.1,
        X_test=None,
        grow_probability=1,
        reconstruct=True,
        ):
        """
        Perform inflate mutation on the given Individual.

        Parameters
        ----------
        individual : Individual
            The tree Individual to mutate.
        ms : float
            Mutation step.
        X : torch.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1).
        reconstruct : bool, optional
            Whether to reconstruct the Individual's collection after mutation (default: True).

        Returns
        -------
        Individual
            The mutated tree Individual.
        """
        if two_trees:
            # getting two random trees
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=True,
            )
            random_tree2 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=True,
            )
            
            # adding the random trees to a list, to be used in the creation of a new block
            random_trees = [random_tree1, random_tree2]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(X_test, testing=True, logistic=True)
                    for rt in random_trees
                ]

        if not two_trees:
            # getting one random tree
            random_tree1 = get_random_tree(
                max_depth,
                FUNCTIONS,
                TERMINALS,
                CONSTANTS,
                inputs=X,
                p_c=p_c,
                grow_probability=grow_probability,
                logistic=single_tree_sigmoid or sig,
            )
            # adding the random tree to a list, to be used in the creation of a new block
            random_trees = [random_tree1]

            # calculating the semantics of the random trees on testing, if applicable
            if X_test is not None:
                [
                    rt.calculate_semantics(
                        X_test, testing=True, logistic=single_tree_sigmoid or sig
                    )
                    for rt in random_trees
                ]

        # getting the correct mutation operator, based on the number of random trees used
        variator = (
            two_trees_delta(operator=operator)
            if two_trees
            else one_tree_delta(operator=operator, sig=sig)
        )        

        # creating the new block for the individual, based on the random trees and operators
        new_block = Tree(
            structure=[variator, *random_trees, ms],
            train_semantics=variator(*random_trees, ms, testing=False),
            test_semantics=(
                variator(*random_trees, ms, testing=True)
                if X_test is not None
                else None
            ),
            reconstruct=True,
        )
        # creating the offspring individual, by adding the new block to it
        offs = Individual(
            collection=[*individual.collection, new_block] if reconstruct else None,
            train_semantics=torch.stack(
                [
                    *individual.train_semantics,
                    (
                        new_block.train_semantics
                        if new_block.train_semantics.shape != torch.Size([])
                        else new_block.train_semantics.repeat(len(X))
                    ),
                ]
            ),
            test_semantics=(
                (
                    torch.stack(
                        [
                            *individual.test_semantics,
                            (
                                new_block.test_semantics
                                if new_block.test_semantics.shape != torch.Size([])
                                else new_block.test_semantics.repeat(len(X_test))
                            ),
                        ]
                    )
                )
                if individual.test_semantics is not None
                else None
            ),
            reconstruct=reconstruct,
        )
        # computing offspring attributes
        offs.size = individual.size + 1
        offs.nodes_collection = [*individual.nodes_collection, new_block.nodes]
        offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

        offs.depth_collection = [*individual.depth_collection, new_block.depth]
        offs.depth = max(
            [
                depth - (i - 1) if i != 0 else depth
                for i, depth in enumerate(offs.depth_collection)
            ]
        ) + (offs.size - 1)
        
        offs.age = individual.age + 1

        return offs

    return inflate


def deflate_mutation(individual, reconstruct):
    """
    Perform deflate mutation on a given Individual by removing a random 'block'.

    Parameters
    ----------
    individual : Individual
        The Individual to be mutated.
    reconstruct : bool
        Whether to store the Individual's structure after mutation.

    Returns
    -------
    Individual
        The mutated individual
    """
    # choosing the block that will be removed
    mut_point = random.randint(1, individual.size - 1)

    # removing the block from the individual and creating a new Individual
    offs = Individual(
        collection=(
            [
                *individual.collection[:mut_point],
                *individual.collection[mut_point + 1 :],
            ]
            if reconstruct
            else None
        ),
        train_semantics=torch.stack(
            [
                *individual.train_semantics[:mut_point],
                *individual.train_semantics[mut_point + 1 :],
            ]
        ),
        test_semantics=(
            torch.stack(
                [
                    *individual.test_semantics[:mut_point],
                    *individual.test_semantics[mut_point + 1 :],
                ]
            )
            if individual.test_semantics is not None
            else None
        ),
        reconstruct=reconstruct,
    )

    # computing offspring attributes
    offs.size = individual.size - 1
    offs.nodes_collection = [
        *individual.nodes_collection[:mut_point],
        *individual.nodes_collection[mut_point + 1 :],
    ]
    offs.nodes_count = sum(offs.nodes_collection) + (offs.size - 1)

    offs.depth_collection = [
        *individual.depth_collection[:mut_point],
        *individual.depth_collection[mut_point + 1 :],
    ]
    offs.depth = max(
        [
            depth - (i - 1) if i != 0 else depth
            for i, depth in enumerate(offs.depth_collection)
        ]
    ) + (offs.size - 1)
    
    offs.age = individual.age + 1

    return offs


# ----------------------------- ADDED ----------------------------- #
@lru_cache(maxsize=128)
def exp_decay_prob(n, decay_rate=0.1):
    """
    Generate an exponential decay probability distribution.
    
    Parameters
    ----------
    n : int
        Number of elements in the distribution.
    decay_rate : float, optional
        Decay rate for the exponential distribution (default: 0.1).

    Returns
    -------
    np.ndarray
        The exponential decay probability distribution.
    """

    prob = np.exp(-decay_rate * np.arange(n))
    prob = prob[::-1]  # Reverse the array
    return prob / np.sum(prob)

@lru_cache(maxsize=128) 
def choose_depth_norm(max_depth, random_index, mean=None, std_dev=None):
    """
    Choose a depth for the structure mutation.
    
    Parameters
    ----------
    max_depth : int
        Maximum depth for generated trees.
    random_index : list
        List of random indices.
    mean : float, optional
        Mean of the normal distribution (default: None).
    std_dev : float, optional
        Standard deviation of the normal distribution (default: None).
        
    Returns
    -------
    int
        The chosen depth.
    """
    depth = max_depth - len(random_index)
    depths = np.arange(1, depth + 1) if len(random_index) > 1 else np.arange(2, depth + 1)
    
    # Set mean and standard deviation
    if mean is None:
        mean = depths.mean()  # Default mean: middle of the range
    if std_dev is None:
        std_dev = (depths[-1] - depths[0]) / 4 
    
    # Generate probabilities using the normal distribution formula
    probabilities = np.exp(-((depths - mean) ** 2) / (2 * std_dev ** 2))
    probabilities /= probabilities.sum()  # Normalize
    
    # Choose a depth using the probabilities
    chosen_depth = random.choices(depths, weights=probabilities, k=1)[0]
    
    return chosen_depth


def structure_mutation(FUNCTIONS, TERMINALS, CONSTANTS, type="old", depth_dist="norm"):
    """
    Generate a function for the structure mutation.

    Parameters
    ----------
    FUNCTIONS : dict
        The dictionary of functions used in the mutation.
    TERMINALS : dict
        The dictionary of terminals used in the mutation.
    CONSTANTS : dict
        The dictionary of constants used in the mutation.
    type : str
        The type of structure mutation to be used.
    depth_dist : str, optional
        Distribution to choose the depth of the new tree (default: "norm"), options: "norm", "exp", "uniform", "max".

    
    Returns
    -------
    
    Callable
        A structure mutation function (`structure`).
        
    """
    
    # def structure_old(individual,
    #                     X,
    #                     max_depth=8,
    #                     p_c=0.1,
    #                     p_prune=0.4,
    #                     X_test=None,
    #                     grow_probability=1,
    #                     replace_probability=0.1,
    #                     reconstruct=True, 
    #                     exp_decay=False,
    #                     **args,
    # ):
                        
    #     """
    #     Perform a mutation on a given Individual by changing the main structure of the tree.

    #     Parameters
    #     ----------
    #     individual : Individual
    #         The Individual to be mutated.
    #     X : torch.Tensor
    #         Input data for calculating semantics.
    #     max_depth : int, optional
    #         Maximum depth for generated trees (default: 8).
    #     p_c : float, optional
    #         Probability of choosing constants (default: 0.1).
    #     p_prune : float, optional
    #         Probability of pruning the tree (default: 0.5).
    #     X_test : torch.Tensor, optional
    #         Test data for calculating test semantics (default: None).
    #     grow_probability : float, optional
    #         Probability of growing trees during mutation (default: 1). 
    #         If changed, trees will be completely replaced during mutation more often.
    #     replace_probability : float, optional
    #         Probability of replacing the main tree during mutation (default: 0.1).
    #     X_test : torch.Tensor, optional
    #         Test data for calculating test semantics (default: None).
    #     exp_decay : bool, optional
    #         Flag to indicate whether exponential decay should be used to soften the mutation (default: False).
    #     reconstruct : bool
    #         Whether to store the Individuals structure after mutation.

    #     Returns
    #     -------
    #     Individual
    #         The mutated individual
    #     """

    #     # Replace the tree
    #     if random.random() < replace_probability:
    #         # In this case, the GP tree will be completely replaced by a new one, larger or smaller
    #         # The intuition behind is that maybe the tree needs to be expanded or diminished
    #         new_block = get_random_tree(
    #             max_depth,
    #             FUNCTIONS,
    #             TERMINALS,
    #             CONSTANTS,
    #             inputs=X,
    #             p_c=p_c,
    #             grow_probability=grow_probability,
    #             logistic=False,
    #         )

    #     # Prune the tree
    #     elif random.random() < p_prune:
    #         # Prune the tree - equivalent to deflate mutation 
    #         indices_list = get_indices(individual.structure[0])

    #         if exp_decay:
    #             probs = exp_decay_prob(len(indices_list), decay_rate=0.1) 
    #             random_index = random.choices(indices_list, weights=probs)[0]

    #         else:
    #             random_index = random.choice(indices_list)

    #         # Generate a terminal or costant
    #         if random.random() < p_c:
    #             new_block = random.choice(list(CONSTANTS.keys()))
    #         else:
    #             new_block = random.choice(list(TERMINALS.keys()))
            
    #         # Swap the subtree in the main tree
    #         new_structure = swap_sub_tree(individual.structure[0], new_block, list(random_index))
    #         new_block = Tree(structure=new_structure,
    #                             train_semantics=None,
    #                             test_semantics=None,
    #                             reconstruct=True)
            
    #         new_block.calculate_semantics(X)

    #     # Replace a block in the tree
    #     else:
    #         indices_list = get_indices(individual.structure[0])
            
    #         # Pre-filter the indices to ensure they meet the condition
    #         valid_indices = [index for index in indices_list if max_depth - len(index) >= 2]

    #         if not valid_indices:
    #             raise ValueError("No valid indices satisfy the condition max_depth - len(index) >= 2")

    #         # Choose from the filtered list
    #         if exp_decay:
    #             probs = exp_decay_prob(len(valid_indices), decay_rate=0.1)
    #             random_index = random.choices(valid_indices, weights=probs)[0]
    #         else:
    #             random_index = random.choice(valid_indices)
                
    #         depth = max_depth - len(random_index)

    #         # get a random tree to replace a block in the main tree
    #         rt = get_random_tree(
    #             depth,
    #             FUNCTIONS,
    #             TERMINALS,
    #             CONSTANTS,
    #             inputs=X,
    #             p_c=p_c,
    #             grow_probability=grow_probability,
    #             logistic=False,
    #         ) 

    #         # Swap the subtree in the main tree
    #         new_structure = swap_sub_tree(individual.structure[0], rt.structure, list(random_index))
        
    #         # Create the new block
    #         new_block = Tree(structure=new_structure,
    #                             train_semantics=None,
    #                             test_semantics=None,
    #                             reconstruct=True)
            
    #         new_block.calculate_semantics(X)            
        
    #     # Create the offspring individual
    #     if X_test is not None:
    #         new_block.calculate_semantics(X_test, testing=True, logistic=False)

    #     offs = Individual(
    #         collection=[new_block, *individual.collection[1:]],
    #         train_semantics=torch.stack(
    #             [
    #                 new_block.train_semantics,
    #                 *individual.train_semantics[1:],
    #             ]   
    #         ),
    #         test_semantics=(
    #             torch.stack(
    #                 [
    #                     new_block.test_semantics,
    #                     *individual.test_semantics[1:],
    #                 ]
    #             )
    #             if X_test is not None
    #             else None
    #         ),
    #         reconstruct=reconstruct
    #     )

    #     # computing offspring attributes
    #     offs.size = individual.size
    #     offs.nodes_collection = [new_block.nodes,*individual.nodes_collection[1:]]
    #     offs.nodes_count = sum(offs.nodes_collection) + offs.size - 1

    #     offs.depth_collection = [new_block.depth, *individual.depth_collection[1:]]
    #     offs.depth = max(offs.depth_collection) + offs.size - 1


    #     offs.recently_mutated = True

    #     return offs
    
    
    def structure(individual,
                        X,
                        max_depth=8,
                        p_c=0.1,
                        X_test=None,
                        grow_probability=1,
                        reconstruct=True, 
                        decay_rate=0.2,
                        **args,
    ):
        """
        Perform a mutation on a given Individual by changing the main structure of the tree.

        Parameters
        ----------
        individual : Individual
            The Individual to be mutated.
        X : torch.Tensor
            Input data for calculating semantics.
        max_depth : int, optional
            Maximum depth for generated trees (default: 8).
        p_c : float, optional
            Probability of choosing constants (default: 0.1).
        p_prune : float, optional
            Probability of pruning the tree (default: 0.5).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        grow_probability : float, optional
            Probability of growing trees during mutation (default: 1). 
            If changed, trees will be completely replaced during mutation more often.
        replace_probability : float, optional
            Probability of replacing the main tree during mutation (default: 0.1).
        X_test : torch.Tensor, optional
            Test data for calculating test semantics (default: None).
        exp_decay : bool, optional
            Flag to indicate whether exponential decay should be used to soften the mutation (default: False).
        reconstruct : bool
            Whether to store the Individuals structure after mutation.

        Returns
        -------
        Individual
            The mutated individual
        """

        indices_with_levels = get_indices_with_levels(individual.structure[0])
        valid_indices_with_levels = [(index, level) for index, level in indices_with_levels if max_depth - len(index) >= 2]

        if not valid_indices_with_levels:
            raise ValueError("No valid indices satisfy the condition max_depth - len(index) >= 2")

        valid_indices, valid_levels = zip(*valid_indices_with_levels)
        probs = exp_decay_prob(max(valid_levels) + 1, decay_rate=decay_rate)
        level_probs = [probs[level] for level in valid_levels]
        random_index = random.choices(valid_indices, weights=level_probs)[0]

        if depth_dist == "norm":
            depth = choose_depth_norm(max_depth, random_index, mean=None, std_dev=None)
            
        else:
            depth = max_depth - len(random_index)   
            depths = np.arange(1, depth + 1) if len(random_index) > 1 else np.arange(2, depth + 1)
            
            if depth_dist == "exp":
                probs = exp_decay_prob(len(depths), decay_rate=decay_rate)
                depth = random.choices(depths, weights=probs)[0]    
                
            elif depth_dist == "uniform":
                depth = random.choice(depths)
                
            elif depth_dist == "max":
                depth = depths[-1]
                
        
        # If just a node is selected
        if depth == 1:
            if random.random() < p_c:
                new_block = random.choice(list(CONSTANTS.keys()))
            else:
                new_block = random.choice(list(TERMINALS.keys()))
            
            # Swap the subtree in the main tree
            new_structure = swap_sub_tree(individual.structure[0], new_block, list(random_index))
                        
        else:
            rt = get_random_tree(
            depth,
            FUNCTIONS,
            TERMINALS,
            CONSTANTS,
            inputs=X,
            p_c=p_c,
            grow_probability=grow_probability,
            logistic=False,
        )         

            # Swap the subtree in the main tree
            new_structure = swap_sub_tree(individual.structure[0], rt.structure, list(random_index))
    
        # Create the new block
        new_block = Tree(structure=new_structure,
                            train_semantics=None,
                            test_semantics=None,
                            reconstruct=True)
        
        new_block.calculate_semantics(X)            
        
        # Create the offspring individual
        if X_test is not None:
            new_block.calculate_semantics(X_test, testing=True, logistic=False)
            
        offs = Individual(
            collection=[new_block, *individual.collection[1:]],
            train_semantics=torch.stack(
                [
                    new_block.train_semantics,
                    *individual.train_semantics[1:],
                ]   
            ),
            test_semantics=(
                torch.stack(
                    [
                        new_block.test_semantics,
                        *individual.test_semantics[1:],
                    ]
                )
                if X_test is not None
                else None
            ),
            reconstruct=reconstruct
        )

        # computing offspring attributes
        offs.size = individual.size
        offs.nodes_collection = [new_block.nodes,*individual.nodes_collection[1:]]
        offs.nodes_count = sum(offs.nodes_collection) + offs.size - 1

        offs.depth_collection = [new_block.depth, *individual.depth_collection[1:]]
        offs.depth = max(offs.depth_collection) + offs.size - 1

        return offs    

    if type == "old":
        pass
        # return structure_old
    else:
        return structure