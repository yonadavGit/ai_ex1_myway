'''
Parse input and run appropriate code.
Don't use this file for the actual work; only minimal code should be here.
We just parse input and call methods from other modules.
'''

#do NOT import ways. This should be done from other files
#simply import your modules and call the appropriate functions
import random
import csv

from ways import graph, info, compute_distance
from ways.info import SPEED_RANGES


class MapProblem:

    def __init__(self, s_start, goal, graph):
        self.s_start = graph[s_start]
        self.goal = graph[goal]
        self.graph = graph

    def actions(self, s):
        return s.links

    def succ(self, s, a):
        if a in s.links:
            return self.graph[a.target]
        raise ValueError(f'No route from {s} through {a}')

    def is_goal(self, s):
        return s == self.goal

    def step_cost(self, a):
        dis = a.distance
        speed = info.SPEED_RANGES[a.highway_type][1]
        return dis / (speed*1000)

    def state_str(self, s):
        return s

    def __repr__(self):
        return {'s_start': self.s_start, 'goal': self.goal}


from functools import total_ordering

def ordered_set(coll):

    return dict.fromkeys(coll).keys()


@total_ordering
class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def expand(self, problem):
        return ordered_set([self.child_node(problem, action)
                            for action in problem.actions(self.state)])

    def child_node(self, problem: MapProblem, action):
        next_state = problem.succ(self.state, action)
        next_node = Node(next_state, self, action,
                         self.path_cost + problem.step_cost(action))
        return next_node

    def solution(self):
        solution = []
        path = self.path()[1:]
        if (len(path) <=0):
            return [self.path()[0].state.index]
        solution.append(path[0].action.source)
        for node in path:
            solution.append(node.action.target)
        return solution, self.path_cost


    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __repr__(self):
        return f"<{self.state}>"

    def __lt__(self, node):
        return self.state < node.state

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        if (self.state.links is list):
            self.state.links = ()
        try:
            return hash(self.state.index)
        except TypeError:
            print(self.state)


import heapq

#There is no priority in Python which allows custom sorting function
class PriorityQueue:

    def __init__(self, f=lambda x: x):
        self.heap = []
        self.f = f

    def append(self, item):
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        for item in items:
            self.append(item)

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        return len(self.heap)

    def __contains__(self, key):
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

    def __repr__(self):
        return str(self.heap)


def huristic_function(lat1, lon1, lat2, lon2):
    return (compute_distance(lat1, lon1, lat2, lon2)/110)



def find_ucs_rout(source, target):
    'call function to find path, and return list of indices'
    g = graph.load_map_from_csv()
    problem = MapProblem(source, target, g)
    return uniform_cost_search(problem)

# f = g:
def uniform_cost_search(problem):
    def g(node):
        return node.path_cost
    return best_first_graph_search(problem, f=g)

def best_first_graph_search(source, target, roads, f):
    node = Node(state=roads[source], path_cost=0)
    frontier = PriorityQueue(f) #Priority Queue
    frontier.append(node)
    closed_list = set()
    while frontier:
        node = frontier.pop()
        if node.state.index == roads[target].index:
            return node.solution(), node.path_cost
        closed_list.add(node.state.index)
        children = []
        for link in node.state.links:
            children.append(Node(state=roads[link.target], parent=node,
                                 path_cost=((link.distance/1000) / SPEED_RANGES[link.highway_type][1]) + node.path_cost))
        for child in children:
            if child.state.index not in closed_list and child not in frontier:
                frontier.append(child)
            elif child in frontier and f(child) < frontier[child]:
                del frontier[child]
                frontier.append(child)
    return None


def find_astar_route(source, target):
    gr = graph.load_map_from_csv()
    def g(node):
        return node.path_cost
    def f(node):
        return g(node) + huristic_function(node.state.lat, node.state.lon, gr[target].lat, gr[target].lon)

    node = gr[source]
    sol = best_first_graph_search(source, target, gr, f=f)
    return sol[0], sol[1], huristic_function(node.lat, node.lon, gr[target].lat, gr[target].lon)



def find_idastar_route(source, target):
    'call function to find path, and return list of indices'
    raise NotImplementedError
    

def dispatch(argv):
    from sys import argv
    source, target = int(argv[2]), int(argv[3])
    if argv[1] == 'ucs':
        path = find_ucs_rout(source, target)
    elif argv[1] == 'astar':
        path = find_astar_route(source, target)
    elif argv[1] == 'idastar':
        path = find_idastar_route(source, target)
    print(' '.join(str(j) for j in path))


def run_100_on_usc():
    paths = []
    g = graph.load_map_from_csv()
    with open("db/problems.csv", 'r') as f:
        problems = f.readlines()
        problems = [prob.strip().split(',') for prob in problems]
        for prob in problems:
            problem = MapProblem(int(prob[0]), int(prob[1]), g)
            path, cost = uniform_cost_search(problem)
            path = " ".join([str(p) for p in path])
            paths.append(path + " - " + str(cost))
    with open("results/UCSRuns.txt", 'w') as f:
        f.writelines("\n".join(paths))



if __name__ == '__main__':
    from sys import argv
    dispatch(argv)
    #run_100_on_usc()
    # find_astar_route(112170,112230)
    # dispatch(*)
