#!/usr/bin/env python
# File created on 13 Feb 2013
from __future__ import division

from numpy import array
from numpy.linalg import norm

from collections import defaultdict
from itertools import izip

from qiime.parse import parse_coords

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

def select_data_for_kmeans(coords_fp, mean_sample_ids,
                           principal_coordinates=[0,1,2]):
    """Selects a subset of data to use for kmeans clustering

    Input:
        coords_fp: the path to a coords file (output from
                   principal_coordinates.py)
        mean_sample_ids: the IDs of the samples that will be used as the means
        principal_coordinates: list of principal coordinates to use (e.g.,
                               [1,2,3])
    Output:
        ({sample_id: data_point, ...},
         {mean_id: mean_point, ...})

        where data_point and mean_point are numpy arrays
    """
    sample_ids, PCs, _, _ = parse_coords(open(coords_fp, 'U'))

    data = {}
    means = {}
    mean_counter = 0

    for sample_id, PC_list in izip(sample_ids, PCs):
        data_point = array([PC_list[i] for i in principal_coordinates])
        data[sample_id] = data_point

        if sample_id in mean_sample_ids:
            means[mean_counter] = data_point
            mean_counter += 1

    return (data, means)

def assign_data_to_means(data, means):
    """Assigns each data point in data to exactly one mean in means

    Points in data are assigned to points in means based on shortest euclidean
    distance.

    Inputs:
        data: dict of {sample_id: data_point, ...}
        means: dict of {mean_id: mean_point, ...}

        where data_point and mean_point are numpy arrays

    Output:
        dict {mean_id: sample_ids, ...} where mean_id is the id of
        a mean from means, and sample_ids is a list of the sample IDs for
        which this is the nearest mean
    """
    result = defaultdict(list)

    for sample_id, data_point in data.iteritems():
        # set the first mean in the list to be the nearest
        nearest_mean_id = 0
        dist_to_nearest_mean = norm(means[0] - data_point)

        # check for closer means
        for mean_id, mean_point in means.iteritems():
            if mean_id == 0:
                # we have already checked means[0]
                continue

            dist_to_this_mean = norm(mean_point - data_point)
            if dist_to_this_mean < dist_to_nearest_mean:
                nearest_mean_id = mean_id

        result[nearest_mean_id].append(sample_id)

    return result

def find_center(data):
    """Finds the cetner of a set of points

    Inputs:
        data: a list of numpy arrays representing points in space

    Output:
        numpy array representing the point at the center of data
    """
    return sum(data) / (1.0 * len(data))

def kmeans(data, means, epsilon = 0.01, max_iterations = 5000):
    """Runs kmeans on data using a list of means

    Inputs:
        data: dict of {point_id: data_point, ...} where data_point is a numpy
              array representing a point in space

        means: dict of {mean_id: mean_point, ...} where mean_point is a numpy
               array representing a point in space

        epsilon: keep iterating until the means move less than epsilon distance

        max_iterations: some high number to avoid infinite runtime in the case
                        that epsilon is never reached (might happin in certain
                        pathological cases)

    Outputs:
        dict of {mean: point_ids, ...} where mean is a tuple representing a
        point in space and point_ids is a set of point_ids for which that mean
        is the closest mean
    """
    current_state = None
    total_change = epsilon + 1
    iteration = 0

    while total_change > epsilon and iteration < max_iterations:
        total_change = 0.0
        current_state = assign_data_to_means(data, means)

        for mean_id, point_ids in current_state.iteritems():
            new_mean = find_center([data[point_id] for point_id in point_ids])
            total_change += norm(new_mean - means[mean_id])
            means[mean_id] = new_mean

        print iteration
        iteration += 1

    results = {}
    for mean_id, point_ids in current_state.iteritems():
        results[tuple(means[mean_id])] = set(point_ids)

    return results
