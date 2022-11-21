# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        score = 0                                       # returning variable
        distance = 0                                    # distance from currPos to food
        oldFood = currentGameState.getFood().asList()   # food with current state
        newFood = successorGameState.getFood().asList() # food with next state
        ateFood = False                                 # flag 

        if action == 'Stop':              # we don't want to stop
            return -99999999

        if len(newFood) < len(oldFood):   # if pacman ate a piece of food by doing this move
            ateFood = True                # flag becomes true
        
        if ateFood:                       # score is increased if food has been eaten. We want this
            score += 200
        
        for food in newFood:              # for every piece of food calculate the Manhattan distance for it
            distance = manhattanDistance(newPos, food)
            score += 100/distance         # the closer the better. Score is increasing for the nearest food

        for ghost in newGhostStates:      # for every ghost find it's position
            ghostPos = ghost.getPosition()
           
            if newPos == ghostPos and ghost.scaredTimer == 0:   # if pacman's position == ghost's position and 
                return -999999999                               # ghost isn't scared then it's game over and we dont want this
 
            elif newPos == ghostPos and ghost.scaredTimer != 0: # if pacman's position == ghost's position and 
                score += 100                                    # ghost is scared then score increases cause pacman ate the ghost

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state, pacmanID, depth):                                   # find the max value for the maximizer => pacman
            legalActions = state.getLegalActions(pacmanID)                      # all legal moves for the pacman agent
            maxValue = -9999999999                                              # initialize to ~= -infinity
            bestAction = None                                                   # best move to return

            for action in legalActions:                     
                succState = state.generateSuccessor(pacmanID, action)           # new successor state after pacman action
                tempValue, NULLaction = minimax(succState, pacmanID+1, depth)   # continue minimax with the next agent and same depth
                                                                                # minimax returns (value, action). We don't need action at this time
                maxValue = max(tempValue, maxValue)                             # keep the max value. Either minimax's or previous value.
                if maxValue == tempValue:                                       # if new maxValue was from this minimax we keep it's action as best
                    bestAction = action
            
            return (maxValue, bestAction)                                       # return (maxValue, bestAction) after minimax finishes


        def minValue(state, pacmanID, depth):                                   # find the min value for the minimizer => ghosts
            legalActions = state.getLegalActions(pacmanID)                      # all legal moves for the ghost agent
            minValue = 9999999999                                               # initialize to ~= +infinity
            bestAction = None                                                   # best move to return

            for action in legalActions:
                succState = state.generateSuccessor(pacmanID, action)           # new successor state after ghost action
                tempValue, NULLaction = minimax(succState, pacmanID+1, depth)   # continue minimax with the next agent(either new ghost or pacman if this is the last agent) and same depth
                                                                                # minimax returns (value, action). We don't need action at this time
                minValue = min(tempValue, minValue)                             # keep the min value. Either minimax's or previous value.
                if minValue == tempValue:                                       # if new minValue was from this minimax we keep it's action as best
                    bestAction = action
            
            return (minValue, bestAction)                                       # return (minValue, bestAction) after minimax finishes

        def minimax(gameState, agentID, depth):     # pacman agentID == 0. Ghost agentID > 0 and agentID < agentsNumber 
            if agentID >= gameState.getNumAgents(): # if minimax was called from the last ghost then
                agentID = 0                         # its time for pacman to play and
                depth  += 1                         # increase the depth

            if gameState.isWin() or gameState.isLose() or (depth == self.depth): # minimax stop check
                bestAction = None
                return (self.evaluationFunction(gameState), bestAction)
            if agentID == 0:    # find the maxValue for the maximizer => Pacman
                return maxValue(gameState, agentID, depth)
            else:               # find the minValue for the minimizer => Ghost
                return minValue(gameState, agentID, depth)

        # ----------- getAction section ------------
        agentId = 0 # pacman agent
        depth   = 0 # starting depth
        value, bestAction = minimax(gameState, agentId, depth)

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, pacmanID, depth, a, b):                                 # find the max value for the maximizer => pacman
            maxValue = -9999999999                                                      # all legal moves for the pacman agent
            bestAction = None                                                           # initialize to ~= -infinity
            legalActions = gameState.getLegalActions(pacmanID)                          # best move to return

            for action in legalActions:
                succState = gameState.generateSuccessor(pacmanID, action)               # new successor state after pacman action
                tempValue, NULLaction = minimax(succState, pacmanID+1, depth, a, b)     # continue minimax with the next agent and same depth
                                                                                        # minimax returns (value, action). We don't need action at this time
                maxValue = max(tempValue, maxValue)                                     # keep the max value. Either minimax's or previous value.
                if maxValue == tempValue:                                               # if new maxValue was from this minimax we keep it's action as best
                    bestAction = action

                a = max(a, maxValue)                                                    # a is the max value

                if a > b:                                                               # a-b pruning termination state
                    return (maxValue, bestAction)                                       
            
            return (maxValue, bestAction)                                               # return (maxValue, bestAction) after minimax finishes

        def minValue(gameState, pacmanID, depth, a, b):                                 # find the min value for the minimizer => ghosts
            minValue = 9999999999                                                       # all legal moves for the ghost agent
            bestAction = None                                                           # initialize to ~= +infinity
            legalActions = gameState.getLegalActions(pacmanID)                          # best move to return

            for action in legalActions:                                                 # new successor state after ghost action
                succState = gameState.generateSuccessor(pacmanID, action)               # continue minimax with the next agent(either new ghost or pacman if this is the last agent) and same depth
                tempValue, NULLaction = minimax(succState, pacmanID+1, depth, a, b)     # minimax returns (value, action). We don't need action at this time
                                                                                        # keep the min value. Either minimax's or previous value.
                minValue = min(tempValue, minValue)                                     # if new minValue was from this minimax we keep it's action as best
                if minValue == tempValue:
                    bestAction = action

                b = min(b, minValue)                                                    # b is the min value

                if b < a:                                                               # a-b pruning termination state
                    return (minValue, bestAction)
            
            return (minValue, bestAction)                                               # return (minValue, bestAction) after minimax finishes

        def minimax(gameState, agentID, depth, a, b): # pacman agentID == 0. Ghost agentID > 0 and agentID < agentsNumber 
            if agentID >= gameState.getNumAgents():   # if a-b pruning(minimax) was called from the last ghost then
                agentID = 0                           # its time for pacman to play and
                depth  += 1                           # increase the depth

            if gameState.isWin() or gameState.isLose() or (depth == self.depth): # a-b pruning(minimax) stop check
                return (self.evaluationFunction(gameState), None)
            if agentID == 0:    # find the maxValue for the maximizer => Pacman
                return maxValue(gameState, agentID, depth, a, b)
            else:               # find the minValue for the minimizer => Ghost
                return minValue(gameState, agentID, depth, a, b)
        
        # ----------- getAction section ------------
        agentId = 0 # pacman agent
        depth   = 0 # starting depth
        a       = -9999999 # starting alpha
        b       =  9999999 # starting beta
        value, bestAction = minimax(gameState, agentId, depth, a, b)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, pacmanID, depth):                                   # find the max value for the maximizer => pacman
            maxValue = -9999999999                                                  # all legal moves for the pacman agent
            bestAction = None                                                       # initialize to ~= -infinity
            legalActions = gameState.getLegalActions(pacmanID)                      # best move to return

            for action in legalActions:
                succState = gameState.generateSuccessor(pacmanID, action)           # new successor state after pacman action
                tempValue, NULLaction = expectimax(succState, pacmanID+1, depth)    # continue expectimax with the next agent and same depth
                                                                                    # expectimax returns (value, action). We don't need action at this time
                maxValue = max(tempValue, maxValue)                                 # keep the max value. Either expectimax's or previous value.
                if maxValue == tempValue:                                           # if new maxValue was from this expectimax we keep it's action as best
                    bestAction = action
            
            return (maxValue, bestAction)                                           # return (maxValue, bestAction) after expectimax finishes


        def chanceValue(gameState, pacmanID, depth):                                # find the average value for the ghosts
            legalActions = gameState.getLegalActions(pacmanID)                      # all legal moves for the ghost agent
            probability = 1.0/len(legalActions)                                     # legal moves probability of ghost 
            average = 0                                                             # initialize average to 0
            bestAction = None                                                       # best move to return

            for action in legalActions:                                             
                succState = gameState.generateSuccessor(pacmanID, action)           # new successor state after ghost action
                tempScore, NULLaction = expectimax(succState, pacmanID+1, depth)    # continue expectimax with the next agent(either new ghost or pacman if this is the last agent) and same depth
                                                                                    # expectimax returns (value, action). We don't need action at this time
                average += tempScore*probability                                    # average value is the sum of expectimax * probability

            return (average, bestAction)                                            # return (average, bestAction) after expectimax finishes


        def expectimax(gameState, agentID, depth):  # pacman agentID == 0. Ghost agentID > 0 and agentID < agentsNumber 
            if agentID >= gameState.getNumAgents(): # if expectimax was called from the last ghost then
                agentID = 0                         # its time for pacman to play and
                depth  += 1                         # increase the depth
        

            if gameState.isWin() or gameState.isLose() or (depth == self.depth): # expectimax stop check
                return (self.evaluationFunction(gameState), None)
            if agentID == 0:    # find the maxValue for the maximizer => Pacman
                return maxValue(gameState, agentID, depth)
            else:               # find the expected value for ghosts
                return chanceValue(gameState, agentID, depth)
        
        # ----------- getAction section ------------
        agentId = 0 # pacman agent
        depth   = 0 # starting depth
        value, bestAction = expectimax(gameState, agentId, depth)

        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    foodList = currentGameState.getFood().asList()
    newPos = currentGameState.getPacmanPosition()

    foodDistances     = []
    ghostDistances    = []
    captuleDistances  = []
    minFoodDist    = 9999999
    minGhostDist   = 9999999
    minCaptuleDist = 9999999

    captules = currentGameState.getCapsules()
    captulesCount = len(currentGameState.getCapsules())
    foodCount = len(foodList)
    ghosts = currentGameState.getGhostPositions()

    if currentGameState.isWin():    return 9999999
    elif currentGameState.isLose():    return -9999999
    else:
        for food in foodList:
            minFoodDist = min(manhattanDistance(food, newPos), minFoodDist)    

        for captule in captules:
            minCaptuleDist = min(manhattanDistance(captule, newPos), minCaptuleDist)    

        for ghost in ghosts:
            minGhostDist = min(manhattanDistance(ghost, newPos), minGhostDist)
            
        if minGhostDist == 0:
            return -9999999

        score = currentGameState.getScore()

        evaluation = 0.01 * score + 0.0001 * minGhostDist + 1/minFoodDist*10 + 1/(minCaptuleDist+0.1)*1000 + 1/(foodCount+0.1)*9999 + 1/(captulesCount+0.1)*5000
        
        return evaluation

# Abbreviation
better = betterEvaluationFunction
