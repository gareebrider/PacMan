import util

pq = util.PriorityQueue()
pq.push(["node1", "path1"], "cost1")
pq.push(["node2", "path2"], "cost2")
pq.push(["node3", "path3"], "cost3")
while not pq.isEmpty():
    print(pq.pop())


