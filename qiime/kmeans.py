#!/usr/bin/env python
# File created on 13 Feb 2013
from __future__ import division

from numpy import array
from numpy.linalg import norm

__author__ = "Adam Robbins-Pianka"
__copyright__ = "Copyright 2011, The QIIME project"
__credits__ = ["Adam Robbins-Pianka"]
__license__ = "GPL"
__version__ = "1.6.0-dev"
__maintainer__ = "Adam Robbins-Pianka"
__email__ = "adam.robbinspianka@colorado.edu"
__status__ = "Development"

"""Functions for computing kmeans clusters given a tab-separated dataset
"""

#cogent.maths.distance_transform.dist_euclidean (and other distance measures
#from cogent.maths.distance_transform)

def assign_data_to_means(data, means):
    """Assigns each data point in data to exactly one mean in means

    Points in data are assigned to points in means based on shortest euclidean
    distance.

    Inputs:
        Both data and means should be lists of numpy arrays where each array
        represents one point in N-dimensional space.

    Output:
        dict {i:array([points]), ...} where i is the index of the mean from
        means and points is a subset of indices from data; data will be
        partitioned into i groups (the values in the dict will be disjoint
        lists)
    """
    pass

def pick_new_means(data, current_means):
    pass

def kmeans_iteration(data, 
    pass

def kmeans(data, num_means, num_iterations):
    pass
