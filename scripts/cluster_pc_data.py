#!/usr/bin/env python
# File created on 14 Feb 2013
from __future__ import division

__author__ = "Adam Robbins-Pianka"
__copyright__ = "Copyright 2011, The QIIME project"
__credits__ = ["Adam Robbins-Pianka"]
__license__ = "GPL"
__version__ = "1.6.0-dev"
__maintainer__ = "Adam Robbins-Pianka"
__email__ = "adam.robbinspianka@colorado.edu"
__status__ = "Development"

from numpy.linalg import norm

from qiime.util import parse_command_line_parameters, make_option
from qiime.parse import parse_mapping_file, parse_coords
from qiime.format import format_mapping_file
from qiime.kmeans import select_pc_data_for_kmeans, kmeans

script_info = {}
script_info['brief_description'] = ""
script_info['script_description'] = ""
script_info['script_usage'] = [("","","")]
script_info['output_description']= ""
script_info['required_options'] = [
 make_option('-i','--input_fp',type="existing_filepath",
                               help=('The principal coordinates file (e.g., '
                                     'the output from principal_coordinates.py'
                                     ')')),
 make_option('-m', '--mapping_file', type='existing_filepath',
                help=('Mapping file to which to add the clustering results')),

 make_option('-n', '--num_clusters', type='int', help=('Number of clusters '
                                                       'to create'))

]
script_info['optional_options'] = [
 make_option('-p', '--PCs', type='string', help=('Subset of PCs to consider '
                                                 'when clustering, as a comma-'
                                                 'separated list(e.g., 1,2,3) '
                                                 '[default=%default]'),
             default='All'),

 make_option('-s', '--seeds', type='string', help=('Sample IDs to use as the '
                                                   'initial means. if no '
                                                   'seeds are supplied, '
                                                   'the samples will be '
                                                   'randomly partitioned into '
                                                   'num_clusters clusters '
                                                   '[default=%default]'),
             default=None)
                                
]

script_info['version'] = __version__

def main():
    option_parser, opts, args = parse_command_line_parameters(**script_info)

    if opts.PCs == 'All':
        # If 'All', do not use a subset, use all PCs by setting PCs to None
        PCs = None
    else:
        PCs = [int(x)-1 for x in opts.PCs.split(',')]

    if opts.seeds:
        seeds = opts.seeds.split(',')
    else:
        seeds = None

    coords_data = parse_coords(open(opts.input_fp, 'U'))

    data, means = select_pc_data_for_kmeans(coords_data,
                                            seeds,
                                            PCs)

    results = kmeans(data, means, opts.num_clusters)

    # write a new column to the mapping file that says for each sample which
    # cluster it is in

    mapping_data, headers, comments = parse_mapping_file(
                                        open(opts.mapping_file), 'U')

    headers = headers[:-1] + ['cluster'] + headers[-1:]

    # the mean_ids in the kmeans results are actually tuples of the
    # coordinates of the means. We do not want to put these coordinates
    # in the mapping file, but rather merely a unique identifier
    coords_to_ids = {}
    cluster_counter = 0
    for i,data in enumerate(mapping_data):
        sample_id = data[0]
        for mean_id, sample_ids in results.iteritems():
            if mean_id not in coords_to_ids:
                coords_to_ids[mean_id] = cluster_counter
                cluster_counter += 1
            cluster_id = coords_to_ids[mean_id]

            if sample_id in sample_ids:
                cluster = cluster_id
                break

        mapping_data[i] = data[:-1] + [str(cluster)] + data[-1:]

    out_f = open(opts.mapping_file, 'w')
    out_f.write(format_mapping_file(headers, mapping_data, comments))
    out_f.close()

if __name__ == "__main__":
    main()
