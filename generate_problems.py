import random
import csv

from ways.graph import load_map_from_csv

roads = load_map_from_csv()

problems = {}

for i in range(0, 100):
    start = random.randint(0, len(roads))
    length = random.randint(30, 70)
    curr = roads[start]
    for j in range(0, length):
        links = curr.links
        if len(links) == 1:
            l = links[0]
        else:
            l = links[random.randint(0, len(links) - 1)]
            while l.target == curr.index:
                l = links[random.randint(0, len(links) - 1)]
        curr = roads[l.target]
    problems[str(start)] = str(l.target)
with open('db/problems.csv', 'w') as f:
    for key in problems.keys():
        f.write("%s,%s\n"%(key, problems[key]))

