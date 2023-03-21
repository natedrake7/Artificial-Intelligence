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

import queue
from game import Actions
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

def depthFirstSearch(problem: SearchProblem()):
    frontier = util.Stack() #the frontier is a stack since we want to explore the deepest node in the graph
    visited = [] #we need to know which nodes have been visited so we dont check them again
    Actions = [] #a list of the actions that lead to our goal
    frontier.push((problem.getStartState(),Actions)) #add the starting state along with the empty list of actions in the stack
    while(not frontier.isEmpty()): #while the frontier is not empty
        node , actions  = frontier.pop() #pop the last item from the stack
        if not node in visited: #check if it has been visited before
            visited.append(node) #add it to the visited list so we dont check it again
            if(problem.isGoalState(node)): #if it is our goal
                return actions #return the list of actions that lead to it
            else:
                Successors = problem.getSuccessors(node) #get the successors of the current node
                for items in Successors: #check all of them(the node might have more than one)
                    newnode,path,value = items #each successor node returns the node name ,the path to the node from our current one and the cost(but the DFS doesnt care about costs)
                    newactions = actions + [path] #create the path to the current node
                    frontier.push((newnode,newactions)) #add it to the frontier so it is explored later

def breadthFirstSearch(problem: SearchProblem):
    frontier = util.Queue() #the frontier is a queue since we want to check by breadth and not depth(bfs is similar to dfs,the only thing that changes is how the frontier is handled)
    visited = [] #we need to know which nodes have been visited so we dont check them again
    Actions = [] #a list of the actions that lead to our goal
    frontier.push((problem.getStartState(), Actions)) #add the starting state along with the empty list of actions in the queue
    while(not frontier.isEmpty()): #while the frontier is not empty
        node , actions  = frontier.pop() #pop the frist item from the queue
        if not node in visited: #check if it has been visited before
            visited.append(node) #add it to the visited list so we dont check it again
            if(problem.isGoalState(node)): #if it is our goal
                return actions #return the list of actions that lead to it
            else:
                Successors = problem.getSuccessors(node) #get the successors of the current node
                for items in Successors: #check all of them(the node might have more than one)
                    newnode,path,value = items #each successor node returns the node name ,the path to the node from our current one and the cost(but the BFS doesnt care about costs)
                    newactions = actions + [path] #create the path to the current node
                    frontier.push((newnode,newactions)) #add it to the frontier so it is explored later


def uniformCostSearch(problem: SearchProblem):
    frontier = util.PriorityQueue() #frontier is a priority queue with a min heap(it returns the element with the least priority)
    visited = []#we need to know which nodes have been visited so we dont check them again
    Actions = []#a list of the actions that lead to our goal
    frontier.push((problem.getStartState(),Actions),problem.getCostOfActions(Actions))#add the starting state along with the empty list of actions in the priority queue
    while(not frontier.isEmpty()): #while the frontier is not empty
        node , actions  = frontier.pop() #pop the item with the least priority from the queue
        if not node in visited: #check if it has been visited before
            visited.append(node) #add it to the visited list so we dont check it again
            if(problem.isGoalState(node)): #if it is our goal
                return actions #return the list of actions that lead to it
            else:
                Successors = problem.getSuccessors(node) #get the successors of the current node
                for items in Successors: #check all of them(the node might have more than one)
                    newnode,path,value = items #each successor node returns the node name ,the path to the node from our current one and the cost(the cost is the priority but we dont need it here,since we get the total cost of all the actions leading to the current node,from the getCostOfActions())
                    newactions = actions + [path] #create the path to the current node
                    frontier.push((newnode,newactions),problem.getCostOfActions(newactions)) #add it to the frontier so it is explored later

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    frontier = util.PriorityQueue() #frontier is a priority queue that returns the item we the least heuristic value
    visited = []#we need to know which nodes have been visited so we dont check them again
    Actions = []#a list of the actions that lead to our goal
    frontier.push((problem.getStartState(),Actions),heuristic(problem.getStartState(),problem))#add the starting state along with the empty list of actions in the priority queue
    while(not frontier.isEmpty()): #while the frontier is not empty
        node , actions  = frontier.pop() #pop the item with the least priority from the queue
        if not node in visited: #check if it has been visited before
            visited.append(node) #add it to the visited list so we dont check it again
            if(problem.isGoalState(node)): #if it is our goal
                return actions #return the list of actions that lead to it
            else:
                Successors = problem.getSuccessors(node) #get the successors of the current node
                for items in Successors: #check all of them(the node might have more than one)
                    newnode,path,value = items #each successor node returns the node name ,the path to the node from our current one and the cost(the cost is the priority but we dont need it here,since we get the total cost of all the actions leading to the current node,from the getCostOfActions())
                    newactions = actions + [path] #create the path to the current node
                    totalcost = problem.getCostOfActions(newactions) + heuristic(newnode,problem) #calculate the total cost by getting the cost of the total actions needed to get to this node plus the heuristic value of the node
                    frontier.push((newnode,newactions),totalcost) #add it to the frontier so it is explored later


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
