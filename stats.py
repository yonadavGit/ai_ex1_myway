'''
This file should be runnable to print map_statistics using 
$ python stats.py
'''

from collections import namedtuple, Counter
from ways import load_map_from_csv
from statistics import mean


def map_statistics(roads):
    '''return a dictionary containing the desired information
    You can edit this function as you wish'''
    Stat = namedtuple('Stat', ['max', 'min', 'avg'])

    all_edges  = []
    for i in roads:
        all_edges += roads[i].links

    edges_per_junction = [roads[i].links for i in roads.keys()]
    edges_per_junction_lens = [len(link_list) for link_list in edges_per_junction]
    edges_per_junction_distances = []
    for link_list in edges_per_junction:
        edges_per_junction_distances.extend([link.distance for link in link_list])


    return {
        'Number of junctions' : len(roads.keys()),
        'Number of links' : len(all_edges),
        'Outgoing branching factor' : Stat(max=max(edges_per_junction_lens), min=min(edges_per_junction_lens), avg=mean(edges_per_junction_lens)),
        'Link distance' : Stat(max=max(edges_per_junction_distances), min=min(edges_per_junction_distances), avg=mean(edges_per_junction_distances)),
        # value should be a dictionary
        # mapping each road_info.TYPE to the no' of links of this type
        'Link type histogram' : Counter([edge.highway_type for edge in all_edges]),  # tip: use collections.Counter
    }


def print_stats():
    for k, v in map_statistics(load_map_from_csv()).items():
        print('{}: {}'.format(k, v))

        
if __name__ == '__main__':
    from sys import argv
    assert len(argv) == 1
    print_stats()

