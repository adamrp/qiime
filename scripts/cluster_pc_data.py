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


from qiime.util import parse_command_line_parameters, make_option
from qiime.parse import parse_mapping_file
from qiime.format import format_mapping_file
from qiime.kmeans import select_data_for_kmeans, kmeans

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

 make_option('-s', '--seeds', type='string', help=('Sample IDs to use as the '
                                                   'initial means.'))
                                
]
script_info['optional_options'] = [\
 make_option('-p', '--PCs', type='string', help=('PCs to consider when '
                                                 'clustering '
                                                 '[default=%default]'),
             default='0,1,2')
]
script_info['version'] = __version__

def main():
    option_parser, opts, args =\
       parse_command_line_parameters(**script_info)

    PCs = map(int, opts.PCs.split(','))
    data, means = select_data_for_kmeans(opts.input_fp,
                                         opts.seeds.split(','),
                                         PCs)

    results = kmeans(data, means,)

    mapping_data, headers, comments = parse_mapping_file(
                                        open(opts.mapping_file), 'U')


    headers = headers[:-1] + ['cluster'] + headers[-1:]

    for i,data in enumerate(mapping_data):
        sample_id = data[0]
        for mean_id, sample_ids in results.iteritems():
            if sample_id in sample_ids:
                cluster = mean_id
                break

        mapping_data[i] = data[:-1] + [str(cluster)] + data[-1:]

    out_f = open(opts.mapping_file, 'w')
    out_f.write(format_mapping_file(headers, mapping_data, comments))
    out_f.close()

if __name__ == "__main__":
    main()
