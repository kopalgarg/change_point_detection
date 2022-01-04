# -*- coding: utf-8 -*-
"""
Created on Mon Jan 3 2022
@author: Kopal Garg
"""
import numpy as np
import pandas as pd
import os
import sys
import json
import numpy as np
import torch
import networkx as nx
import netrd
import pandas as pd
from netrd.utilities import entropy, ensure_undirected
import numpy as np
import community # install with "pip install python-louvain"
from collections import defaultdict
from collections import Counter
import scipy as sp
import scipy.sparse as sparse
import math
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import glob
import itertools
import networkx as nx
from netrd.reconstruction import MarchenkoPastur
from netrd.distance import PortraitDivergence
from netrd.distance import OnionDivergence
from netrd.distance import DistributionalNBD
from netrd.distance import NetSimile
from netrd.distance import PolynomialDissimilarity
from netrd.distance import CommunicabilityJSD
from netrd.distance import ResistancePerturbation
from netrd.distance import QuantumJSD
from netrd.distance import DeltaCon
from netrd.distance import HammingIpsenMikhailov
from netrd.distance import IpsenMikhailov
from netrd.distance import LaplacianSpectral
from netrd.distance import JaccardDistance
from netrd.distance import DegreeDivergence
from netrd.distance import Frobenius
from netrd.distance import Hamming
from netrd.distance import NetLSD
from netrd.distance import DMeasure

def plot_graph(G, with_labels=True):

    """
    Plot correlation network graph

    Arguments:
        G {graph} -- [Graph to plot]
        with_labels {bool} -- [include labels, Y/N]

    """
    pos = nx.kamada_kawai_layout(G)
    nodecolor = G.degree(weight='weight') 
    nodecolor2 = pd.DataFrame(nodecolor) 
    nodecolor3 = nodecolor2.iloc[:, 1]  
    edgecolor = range(G.number_of_edges()) 
    nx.draw(G,
            pos,
            with_labels=with_labels,
            node_size=200,
            node_color=nodecolor3 * 5,
            edge_color=edgecolor)
    plt.show()