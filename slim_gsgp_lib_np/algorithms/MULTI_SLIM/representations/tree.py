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
Tree class implementation for representing tree structures in genetic programming.
"""

from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree_utils import _execute_tree, tree_depth_and_nodes

class Tree:
    """
    The Tree class representing the candidate solutions in MULTI-SLIM-GSGP.

    Attributes
    ----------
    repr_ : tuple or str
        Representation of the tree structure.
    FUNCTIONS : dict
        Dictionary of allowed functions in the tree representation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree representation.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree representation.
    SPECIALISTS : dict
        Dictionary of specialist functions allowed in the tree representation.
    depth : int
        Depth of the tree.
    fitness : float
        Fitness value of the tree.
    test_fitness : float
        Test fitness value of the tree.
    nodes_count : int
        Number of nodes in the tree.
    total_nodes : int
        Total number of nodes in the tree, including the specialists.
    """

    TERMINALS = None
    FUNCTIONS = None
    CONSTANTS = None
    SPECIALISTS = None

    def __init__(self, repr_):
        """
        Initializes a Tree object.

        Parameters
        ----------
        repr_ : tuple
            Representation of the tree structure.
        """
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS
        self.SPECIALISTS = Tree.SPECIALISTS

        self.repr_ = repr_
        self.depth, self.nodes_count, self.total_nodes = tree_depth_and_nodes(Tree.FUNCTIONS, Tree.SPECIALISTS)(repr_)
        self.fitness, self.test_fitness = None, None
        self.train_semantics, self.test_semantics = None, None 

    def apply_tree(self, inputs, testing, predict):
        """
        Apply the tree to the input data.

        Parameters
        ----------
        inputs : np.ndarray
            Input data.
        testing : bool
            Whether the tree is being evaluated on a testing set.
        predict : bool
            Whether the tree is being used for prediction (i.e., not with the intended datasets (train, test)).

        Returns
        -------
        semantics : np.ndarray
            Predicted values.
        """

        return _execute_tree(
            repr_=self.repr_,
            X=inputs,
            FUNCTIONS=self.FUNCTIONS,
            TERMINALS=self.TERMINALS,
            CONSTANTS=self.CONSTANTS,
            SPECIALISTS=self.SPECIALISTS,
            testing=testing,
            predict=predict
        )
    
    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for the tree.

        Parameters
        ----------
        inputs : np.ndarray
            Input data for calculating semantics.
        testing : bool
            Boolean indicating if the calculation is for testing semantics.
        """

        if testing:
            self.test_semantics = self.apply_tree(inputs, testing=True, predict=False)
        else:
            self.train_semantics = self.apply_tree(inputs, testing=False, predict=False)


    def evaluate(self, ffunction, X, y, testing=False):
        """
        Evaluate the tree on the given data.

        Parameters
        ----------
        ffunction : callable
            Fitness function.
        X : np.ndarray
            Input data.
        y : np.ndarray
            Target data.
        testing : bool
            Whether the tree is being evaluated on a testing set.
        """

        preds = self.apply_tree(X, testing, False)
        if testing:
            self.test_fitness = ffunction(y, preds)
        else:
            self.fitness = ffunction(y, preds)        

    def predict(self, X):
        """
        Predict the target values using the tree.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Predicted values.
        """

        return self.apply_tree(X, testing=False, predict=True)

    def get_tree_representation(self, indent=""):
        """
        Get the tree representation as a string.

        Parameters
        ----------
        indent : str
            Indentation string.

        Returns
        -------
        str
            Tree representation.
        """

        representation = []
        
        if isinstance(self.repr_, tuple):
            # Check if the node is a function node (its first element is a function key)
            if isinstance(self.repr_[0], str) and self.repr_[0] in Tree.FUNCTIONS:
                function_name = self.repr_[0]
                representation.append(indent + f"{function_name}(\n")
                
                # Process children based on the function's arity.
                if Tree.FUNCTIONS[function_name]["arity"] == 2:
                    left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                    representation.append(Tree(left_subtree).get_tree_representation(indent + "  "))
                    representation.append(Tree(right_subtree).get_tree_representation(indent + "  "))
                else:
                    left_subtree = self.repr_[1]
                    representation.append(Tree(left_subtree).get_tree_representation(indent + "  "))
                
                representation.append(indent + ")\n")
            else:
                # Otherwise, assume it is a conditional node.
                # Structure: (condition, branch_if_true, branch_if_false)
                representation.append(indent + "if (\n")
                representation.append(Tree(self.repr_[0]).get_tree_representation(indent + "  "))
                representation.append(indent + ") > 0 then\n")
                representation.append(Tree(self.repr_[1]).get_tree_representation(indent + "  "))
                representation.append(indent + "else\n")
                representation.append(Tree(self.repr_[2]).get_tree_representation(indent + "  "))
                representation.append(indent + "endif\n")
        else:
            # Terminal node.
            representation.append(indent + f"{self.repr_}\n")
        
        return "".join(representation)

    def print_tree_representation(self, indent=""):
        """
        Print the tree representation.
        
        Parameters
        ----------
        indent : str
            Indentation string.
        """

        print(self.get_tree_representation(indent=indent))