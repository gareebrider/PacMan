# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"  
    from util import Stack

    queueFunc = Stack() # queuing function for DFS-search is Stack
    visited = set()     # a set of already visited nodes
    path = []           # a direction path from start to finish

    startNode = problem.getStartState()
    queueFunc.push((startNode, path))   # initialize Stack

    while not queueFunc.isEmpty():
        # Stack contains (node, path from start to node)
        node, path = queueFunc.pop()
        visited.add(node) # when a node has been popped from the stack, it's been pushed to the visited set

        if problem.isGoalState(node): # goal check
            return path

        # getSuccessors -> (successor, action, stepcost)
        successors = problem.getSuccessors(node) # successors is a list of all successors

        if successors: # if list is not empty
            for successor in successors: 
                child  = successor[0]
                action = successor[1]
                cost   = successor[2]

                # searching for unvisited nodes
                if child not in visited:
                    # each child needs a new unique path
                    new_path = path.copy()           # copy old path to a new path for each successor
                    new_path.append(action)          # add the new successor move  
                    queueFunc.push((child,new_path)) # push the node and it's unique path 
    else:
        return [] # if stack is empty return empty list


# helper function for checking if Queue contains args node 
def isInQueue(node, queue: util.Queue):
    for item in queue.list: # (node, path)
        if item[0] == node:
            return True
    
    return False

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue

    queueFunc = Queue() # queuing function for BFS-search is Queue
    visited = set()     # a set of already visited nodes
    path = []           # a direction path from start to finish
    
    startNode = problem.getStartState()
    queueFunc.push((startNode, path))   # initialize Queue

    while not queueFunc.isEmpty():
        # Queue contains (node, path from start to node)
        node, path = queueFunc.pop()
        visited.add(node) # when a node has been popped from the queue, it's been pushed to the visited set

        if problem.isGoalState(node): # goal check
            return path

        # getSuccessors -> (successor, action, stepcost)
        successors = problem.getSuccessors(node) # successors is a list of all successors

        if successors: # if list is not empty
            for successor in successors: 
                child  = successor[0]
                action = successor[1]
                cost   = successor[2]
            
                # searching for unvisited nodes
                if child not in visited and not isInQueue(child, queueFunc):
                    # each child needs a new unique path
                    new_path = path.copy()           # copy old path to a new path for each successor
                    new_path.append(action)          # add the new successor move  
                    queueFunc.push((child,new_path)) # push the node and it's unique path 
    else:
        return [] # if queue is empty return empty list


# helper function for checking if PQueue contains args node 
def isInPQueue(node, pqueue: util.PriorityQueue):
    for item in pqueue.heap:
        # item[2][0] because pqueue.heap => (cost, count, (node, path))
        if item[2][0] == node:
            return True
    
    return False

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    queueFunc = PriorityQueue() # queuing function for UCS-search is PriorityQueue
    visited = set()     # a set of already visited nodes
    path = []           # a direction path from start to finish

    startNode = problem.getStartState()
    queueFunc.push((startNode, path), problem.getCostOfActions(path))   # initialize PriorityQueue

    while not queueFunc.isEmpty():
        # PQueue contains ((node, start=>node path), cost of the path)
        node, path = queueFunc.pop() # Pqueue pop() returns item
        visited.add(node) # when a node has been popped from the priorityQueue, it's been pushed to the visited set

        if problem.isGoalState(node): # goal check
            return path

        # getSuccessors -> (successor, action, stepcost)
        successors = problem.getSuccessors(node) # successors is a list of all successors

        if successors: # if list is not empty
            for successor in successors: 
                child  = successor[0]
                action = successor[1]
                cost   = successor[2]

                # searching for unvisited nodes
                if child not in visited and not isInPQueue(child, queueFunc):
                    # each child needs a new unique path
                    new_path = path.copy()                 # copy old path to a new path for each successor
                    new_path.append(action)                # add the new successor move  
                    queueFunc.push((child,new_path), problem.getCostOfActions(new_path)) # push the node with it's path and new cost 
                # if its already in the PQueue I check if there's a quickest path with less cost than before and if so, update it    
                elif child not in visited and isInPQueue(child, queueFunc):
                    # each child needs a new unique path
                    new_path = path.copy()                 # copy old path(Parent's) to a new path for each successor
                    new_path.append(action)                # add the new successor move  
                    # targetting the current child in the Pqueue that needs new pathCost check 
                    for item in queueFunc.heap: # item[2][0] because pqueue.heap => (cost, count, (node, path))
                        node = item[2][0]
                        if node == child: # found it
                            old_cost = item[0]
                            new_cost = problem.getCostOfActions(new_path)
                            if new_cost < old_cost: # update it with the lowest cost and fastest path
                                queueFunc.update((child, new_path), new_cost)                       
    else:
        return [] # if PriorityQueue is empty return empty list


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    queueFunc = PriorityQueue() # queuing function for A*search is PriorityQueue
    visited = set()     # a set of already visited nodes
    path = []           # a direction path from start to finish

    startNode = problem.getStartState()
    # f(n) = c(n) + h(n) 
    f = problem.getCostOfActions(path) + heuristic(startNode, problem)
    queueFunc.push((startNode, path), f) # initialize PriorityQueue

    while not queueFunc.isEmpty():    
        # PQueue contains ((node, start=>node path), f(n))  
        node, path = queueFunc.pop() # Pqueue pop() returns item
        
        if node in visited: # We don't want duplicate nodes
            continue

        visited.add(node) # when a node has been popped from the priorityQueue, it's been pushed to the visited set
        
        if problem.isGoalState(node): # goal check
            return path      

        # getSuccessors -> (successor, action, stepcost)
        successors = problem.getSuccessors(node) # successors is a list of all successors

        if successors: # if list is not empty
            for successor in successors: 
                child  = successor[0]
                action = successor[1]
                cost   = successor[2]

                # searching for unvisited nodes
                if child not in visited:
                    # each child needs a new unique path
                    new_path = path.copy()  # copy old path to a new path for each successor
                    new_path.append(action) # add the new successor move  
                    f = problem.getCostOfActions(new_path) + heuristic(child, problem) # f(n) = c(n) + h(n)
                    queueFunc.push((child, new_path), f) # push the node with it's path and new f(n) 


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
