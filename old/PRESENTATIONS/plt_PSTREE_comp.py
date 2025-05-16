import graphviz

class TreePlotter:
    """
    A class to plot tree-based individuals for Standard GP, PSTree, and MultiGP.
    It uses unique node IDs for each plot to ensure nodes are distinct.
    """

    def __init__(self):
        self.node_counter = 0
        self.dot = None

    def _get_node_id(self):
        """Generates a unique ID for each node."""
        self.node_counter += 1
        return f"node{self.node_counter}"

    def _add_node(self, node_id, label_str):
        """Adds a node to the graph, styled like the example image (ellipse)."""
        self.dot.node(node_id, label=str(label_str), shape='box')

    def _add_edge(self, parent_id, child_id, label_str=""):
        """Adds an edge to the graph."""
        self.dot.edge(parent_id, child_id, label=str(label_str))

    # --- Standard GP Plotting ---
    def _plot_standard_gp_recursive(self, gp_tuple, parent_id=None):
        """
        Recursively plots a Standard GP individual.
        Assumes gp_tuple is like (operator, operand1, operand2, ...)
        or a terminal (variable/constant as string/number).
        """
        current_id = self._get_node_id()

        if isinstance(gp_tuple, tuple):
            operator = gp_tuple[0]
            self._add_node(current_id, str(operator))
            if parent_id:
                self._add_edge(parent_id, current_id)

            for i in range(1, len(gp_tuple)):
                operand = gp_tuple[i]
                self._plot_standard_gp_recursive(operand, current_id)
        else: # Terminal node (variable or constant)
            self._add_node(current_id, str(gp_tuple))
            if parent_id:
                self._add_edge(parent_id, current_id)
        return current_id


    def plot_standard_gp(self, individual_tuple, title="StdGP"):
        """
        Creates a plot for a Standard GP individual.
        Example individual: ('-', ('*', ('-', 'x', '1'), 'y'), '4')
        """
        self.node_counter = 0 # Reset for each new graph
        self.dot = graphviz.Digraph(comment=title)
        self.dot.attr(label=title+'\n\n', labelloc='t', fontsize='25') # Graph title

        self._plot_standard_gp_recursive(individual_tuple)
        return self.dot

    # --- PSTree Plotting ---
    def _plot_pstree_recursive(self, pstree_tuple, parent_id=None, edge_label=""):
        """
        Recursively plots a PSTree individual.
        Assumes pstree_tuple is like (condition, true_branch, false_branch)
        or a terminal expression (string).
        """
        current_id = self._get_node_id()

        if isinstance(pstree_tuple, tuple) and len(pstree_tuple) == 3: # Condition node
            condition = pstree_tuple[0]
            true_branch = pstree_tuple[1]
            false_branch = pstree_tuple[2]

            self._add_node(current_id, str(condition))
            if parent_id:
                self._add_edge(parent_id, current_id, label_str=edge_label)

            # True branch
            self._plot_pstree_recursive(true_branch, current_id, "YES")
            # False branch
            self._plot_pstree_recursive(false_branch, current_id, "NO")
        else: # Terminal expression node
            self._add_node(current_id, str(pstree_tuple))
            if parent_id:
                self._add_edge(parent_id, current_id, label_str=edge_label)
        return current_id

    def plot_pstree(self, individual_tuple, title="PSTree"):
        """
        Creates a plot for a PSTree individual.
        Example: ('f1 > 0.27',
                    ('f2 <= -1', 'y = f1^2 + f2', 'y = 1/log(f1)'),
                    'y = sin(f2)')
        """
        self.node_counter = 0
        self.dot = graphviz.Digraph(comment=title)
        self.dot.attr(label=title+'\n\n', labelloc='t', fontsize='25')

        self._plot_pstree_recursive(individual_tuple)
        return self.dot

    # --- MultiGP Plotting ---
    def _plot_multigp_recursive(self, multigp_tuple, parent_id=None, edge_label=""):
        """
        Recursively plots a MultiGP individual.
        Assumes multigp_tuple is like (condition, true_branch_subtree, false_branch_subtree)
        or a terminal specialist (string). True/false branches can be further tuples or specialists.
        """
        current_id = self._get_node_id()

        # Check if it's a condition node (tuple of 3 elements)
        # (condition, true_branch, false_branch)
        if isinstance(multigp_tuple, tuple) and len(multigp_tuple) == 3:
            condition = multigp_tuple[0]
            true_branch = multigp_tuple[1]
            false_branch = multigp_tuple[2]

            self._add_node(current_id, str(condition))
            if parent_id:
                self._add_edge(parent_id, current_id, label_str=edge_label)

            # True branch: Can be another condition tuple or a specialist string
            self._plot_multigp_recursive(true_branch, current_id, "YES")
            # False branch: Can be another condition tuple or a specialist string
            self._plot_multigp_recursive(false_branch, current_id, "NO")
        else: # Terminal node (Specialist)
            specialist_name = str(multigp_tuple)
            self._add_node(current_id, specialist_name) # Specialists also in ellipses as per image style
            if parent_id:
                self._add_edge(parent_id, current_id, label_str=edge_label)
        return current_id


    def plot_multigp(self, individual_tuple, title="PSSR"):
        """
        Creates a plot for a MultiGP individual.
        Example: ('condition1(x)',
                    ('condition2(x)', 'S1(x)', 'S8(x)'),
                    'S5(x)')
        """
        self.node_counter = 0
        self.dot = graphviz.Digraph(comment=title)
        self.dot.attr(label=title+'\n\n', labelloc='t', fontsize='25')

        self._plot_multigp_recursive(individual_tuple)
        return self.dot


if __name__ == '__main__':
    plotter = TreePlotter()

    # # 1. Standard GP Example
    # gp_individual = ('-',
    #                     ('*',
    #                         ('-', 'x', '1'), # Ensure terminals are strings or numbers
    #                         'y'),
    #                     '4')
    # gp_plot = plotter.plot_standard_gp(gp_individual)
    # # To save to file and view (optional: view=True opens the file)
    # gp_plot.render('standard_gp_tree', view=False, format='png')
    # print("Standard GP tree plot saved to standard_gp_tree.png")

    # # 2. PSTree Example (based on the image structure)
    # pstree_individual = ('f1 > 0.27',
    #                         ('f3 <= -1', 'y = f4 - 0.5*f2', 'y = 2*f1'),
    #                         'y = 0.3*f2 + f1')
    # pstree_plot = plotter.plot_pstree(pstree_individual)
    # pstree_plot.render('pstree_tree', view=False, format='png')
    # print("PSTree plot saved to pstree_tree.png")

    # 3. MultiGP Example
    multigp_individual = ('c1(x) > 0',
                            ('c2(x) > 0', 'S1(x)', 'S2(x)'),
                            'S9(x)')
    multigp_plot = plotter.plot_multigp(multigp_individual)
    multigp_plot.render('multigp_tree', view=False, format='png')
    print("MultiGP plot saved to multigp_tree.png")

    # If you want to display in a Jupyter notebook, you can just return the `dot` object:
    # In Jupyter:
    # plotter = TreePlotter()
    # gp_individual = ('-', ('*', ('-', 'x', '1'), 'y'), '4')
    # gp_plot = plotter.plot_standard_gp(gp_individual)
    # gp_plot # This will display the graph