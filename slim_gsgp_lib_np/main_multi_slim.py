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
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import uuid
import os
import warnings

from slim_gsgp_lib_np.main_slim import slim

# -----------------------------------  GP -----------------------------------
from slim_gsgp_lib_np.algorithms.GP.gp import GP
from slim_gsgp_lib_np.algorithms.GP.operators.mutators import mutate_tree_subtree
from slim_gsgp_lib_np.algorithms.GP.representations.tree_utils import tree_depth
from slim_gsgp_lib_np.config.gp_config import *
from slim_gsgp_lib_np.selection.selection_algorithms import tournament_selection_max, tournament_selection_min
from slim_gsgp_lib_np.utils.logger import log_settings
