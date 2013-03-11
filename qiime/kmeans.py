#!/usr/bin/env python
# File created on 13 Feb 2013
from __future__ import division

from numpy import array, mean
from numpy.random import uniform, shuffle
from numpy.linalg import norm

from collections import defaultdict
from itertools import izip

__author__ = "Adam Robbins-Pianka"
__copyright__ = "Copyright 2011, The QIIME project"
__credits__ = ["Adam Robbins-Pianka"]
__license__ = "GPL"
__version__ = "1.6.0-dev"
__maintainer__ = "Adam Robbins-Pianka"
__email__ = "adam.robbinspianka@colorado.edu"
__status__ = "Development"

"""Functions for computing kmeans clusters
"""

#TODO:
#k-means++ -- way of picking initial means (when choosing random means) that
#             bounds how bad the solution can be compared to optimal
#             http://en.wikipedia.org/wiki/K-means%2B%2B

class UnknownSampleID(Exception):
    pass

class BadNumberOfClusters(Exception):
    pass

class MaxIterationsReached(Exception):
    pass

def select_pc_data_for_kmeans(coords_data, mean_sample_ids = None,
                              principal_coordinates = None):
    """Selects a subset of data to use for kmeans clustering

    Input:
        coords_data: output of qiime.parse.parse_coords
        mean_sample_ids: the IDs of the samples that will be used as the means.
                         can be None.
        principal_coordinates: list of principal coordinates to use (e.g.,
                               [0,1,2]). If None, use all.
    Output:
        ({sample_id: data_point, ...},
         {mean_id: mean_point, ...})

        where data_point and mean_point are numpy arrays
        Each mean_id will be a unique identifier (0, 1, 2, ...)
        Each sample_id will be the sample ID from the input coords_data
    """
    sample_ids, PCs = coords_data[0], coords_data[1]

    # if principal_coordinates is None, use all the principal coordinates
    if principal_coordinates is None:
        principal_coordinates = range(len(PCs))

    data = {}
    means = {}
    mean_counter = 0

    for sample_id, PC_list in izip(sample_ids, PCs):
        data_point = array([PC_list[i] for i in principal_coordinates])
        data[sample_id] = data_point

        if mean_sample_ids:
            if sample_id in mean_sample_ids:
                means[mean_counter] = data_point
                mean_counter += 1

    if mean_sample_ids:
        for mean_sample_id in mean_sample_ids:
            if mean_sample_id not in data:
                raise UnknownSampleID, ("Sample id %s not in coords file." %
                                        mean_sample_id)

    return (data, means)

def assign_data_to_means(data, means, distance_fn, num_clusters = None):
    """Assigns each data point in data to exactly one mean in means

    Points in data are assigned to points in means based on shortest euclidean
    distance. If no means are supplised, randomly partition data into
    num_clusters clusters.

    Inputs:
        data: dict of {sample_id: data_point, ...}
        means: dict of {mean_id: mean_point, ...}
               If None, randomly partitions data into the means
        distnce_fn: function that calculates "distance" between points. The
                    function must take two parameters (the vectors) and return
                    a single value.
        num_clusters: Used if means is None; number of clusters into which
                      the data points will be randomly partitioned. Note,
                      num_clusters is ignored if initial means are supplied

        where data_point and mean_point are numpy arrays

    Output:
        dict {mean_id: sample_ids, ...} where mean_id is the id of
        a mean from means, and sample_ids is a list of the sample IDs for
        which this is the nearest mean
    """
    result = defaultdict(list)

    # if no means are supplied, randomly partition data
    if not means:
        if not num_clusters:
            raise BadNumberOfClusters, ("This function requires initial means "
                                        "or, if the data is to be randomly "
                                        "partitioned, the number of clusters "
                                        "to create")
        if num_clusters > len(data):
            raise BadNumberOfClusters, ("Number of clusters requested (%d) "
                                        "exceeds number of data points (%d)" %
                                        (num_clusters, len(data)))

        # assign the first k points to different clusters, to avoid
        # empty clusters
        mean_assignments = range(num_clusters)
        shuffle(mean_assignments)

        # then assign the rest of the points randomly
        mean_assignments.extend(map(int, uniform(0, num_clusters,
                                                 len(data) - num_clusters)))

        for counter, (sample_id, data_point) in enumerate(data.iteritems()):
            result[mean_assignments[counter]].append(sample_id)

        return result

    # implicit "else" here
    for counter, (sample_id, data_point) in enumerate(data.iteritems()):
        # set the first mean in the list to be the nearest
        nearest_mean_id = 0
        dist_to_nearest_mean = distance_fn(means[0], data_point)

        # check for closer means
        for mean_id, mean_point in means.iteritems():
            if mean_id == 0:
                # we have already checked means[0]
                continue

            dist_to_this_mean = distance_fn(mean_point, data_point)
            if dist_to_this_mean < dist_to_nearest_mean:
                nearest_mean_id = mean_id
                dist_to_nearest_mean = dist_to_this_mean

        result[nearest_mean_id].append(sample_id)

    return result

def euclidean_distance(v1, v2):
    """Returns the euclidean distance between vectors v1 and v2

    v1 and v2 should be numpy arrays
    """
    return norm(v1-v2)

def euclidean_center(data):
    """Finds the cetner of a set of points

    Inputs:
        data: a list of n-length numpy arrays representing points in n-space

    Output:
        numpy array representing the point at the center of data
    """
    return mean(data, axis=0)

def kmeans(data, means, num_clusters, distance_fn=euclidean_distance, 
           find_center_fn=euclidean_center, epsilon = 1e-5,
           max_iterations = 5000):
    """Runs kmeans on data using a list of means

    Inputs:
        data: dict of {point_id: data_point, ...} where data_point is a numpy
              array representing a point in space

        means: dict of {mean_id: mean_point, ...} where mean_point is a numpy
               array representing a point in space. If None, the first
               iteration will randomly partition the set of data points into
               num_clusters clusters.

        num_clusters: integeral number of clusters to form. If means is not
                      None, then this number must match the number of means in
                      means.

        distance_fn: function that calculates "distance" between two
                     vectors. The function must take two parameters (the
                     vectors) and return a single value. The vectors should be
                     numpy arrays. Defaults to euclidean distance.

        find_center_fn: function that calculates the center of a set of
                        vectors. Must take as input a list of numpy arrays
                        and output a single numpy array.

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

    randomly_partitioning = not means

    while total_change > epsilon:
        # make sure we have not exceeded max iterations. If so, throw exception
        if iteration == max_iterations:
            raise MaxIterationsReached, ("K-means reached %d iterations, but "
                                         "the maximum number of iterations "
                                         "was set to %d") % (iteration,
                                                             max_iterations)

        # Note that if means is none, the data will be randomly partitioned
        current_state = assign_data_to_means(data, means, distance_fn,
                                             num_clusters)

        total_change = 0.0
        for mean_id, point_ids in current_state.iteritems():
            new_mean = find_center_fn([data[point_id] for point_id in
                                                          point_ids])

            # if we are randomly partitioning, means[mean_id] is undefined
            if not randomly_partitioning:
                total_change += distance_fn(new_mean, means[mean_id])

            means[mean_id] = new_mean

        if randomly_partitioning:
            total_change = epsilon + 1
            randomly_partitioning = False

        iteration += 1

    results = {}
    for mean_id, point_ids in current_state.iteritems():
        results[tuple(means[mean_id])] = set(point_ids)

    return results
