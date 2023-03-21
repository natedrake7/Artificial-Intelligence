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
from math import exp

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
        childGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        FoodValue = 100.00
        CapsuleValue = 150.0
        GhostPenalty = -500.0
        FoodDecay = 0.25
        CapsuleDecay = 0.25
        GhostDecay = 0.2
        StopPenalty = -100.0
        TotalScore = 0.0

        if currentGameState.isWin() or currentGameState.isLose():
            TotalScore -= 1e6
            return TotalScore
        if action == Directions.STOP: #if pacman doesnt move
            TotalScore += StopPenalty #apply the stop penalty
        TotalScore += FoodValue *currentGameState.hasFood(newPos[0], newPos[1]) #update the score if the current state has food by the current food factor
        FoodList = childGameState.getFood().asList() #get a list of all the foods
        for food in FoodList:
            TotalScore -= FoodValue * (1 - exp(-2.0 * FoodDecay  * util.manhattanDistance(newPos, food))) #apply the decay factor
        CapsuleList = currentGameState.data.capsules #Get a list of all the capsules
        for capsule in CapsuleList: 
            if newPos == capsule: #if the new position of pacman has a capsule
                TotalScore += CapsuleValue #update the current score
        CapsuleList = childGameState.data.capsules
        for capsule in CapsuleList:
            TotalScore -= CapsuleValue * exp(-2.0 * CapsuleDecay * util.manhattanDistance(newPos, capsule)) #apply the decay factor for each capsule left in the game
        CurGhostPos = currentGameState.getGhostState(1).getPosition() #Get the ghost current position
        NewGhostPos = childGameState.getGhostState(1).getPosition() #Get the ghost next position
        if newPos in [CurGhostPos, NewGhostPos]: #if pacman gets hit by a ghost
            TotalScore += GhostPenalty #update the score accordingly
        else:
            TotalScore += GhostPenalty * exp(-2.0 * GhostDecay * util.manhattanDistance(newPos, NewGhostPos)) #apply the decay factor for the ghosts in the game

        return TotalScore #return the final score


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
    def minimax(self, CurrentState, depth, Agent = 0, MaximizingPlayer = True):  
        Actions = CurrentState.getLegalActions(Agent)#Get the legal action
        if MaximizingPlayer: #if the maximizing player is playing
            Score = [] #Create an empty list to hold all the scores
            if depth == 0 or CurrentState.isWin() or CurrentState.isLose(): #if the depth of the tree is 0 or pacman won or pacman lost
                return self.evaluationFunction(CurrentState), Directions.STOP #return the evaluation function for the current state,plus the Stop Direction    
            for action in Actions: #for each action
                Score.append(self.minimax(CurrentState.generateSuccessor(Agent,action), depth - 1, 1, False)[0]) #add the value of the score to the end of the score list
            BestScore = max(Score) #Since the maximizing player is playing we get the maximum out of the leaves
            BestActions = [] #create and empty list to hold each index has the best score in order to track the actions that lead to that score
            for i in range(len(Score)): #for each item in score
               if Score[i] == BestScore: #if it has the best value
                BestActions.append(i) #append it to the list
            return BestScore,Actions[random.choice(BestActions)] #Return the best Score plus the actions that lead to the best score
        else: #if the maximizing player doesnt play
            if depth == 0 or CurrentState.isWin() or CurrentState.isLose(): #if the depth of the tree is 0 or pacman won or pacman lost
                return self.evaluationFunction(CurrentState), Directions.STOP #return the evaluation function for the current state,plus the Stop Direction 
            Score = [] #Create an empty list to hold all the scores
            if Agent == CurrentState.getNumAgents() - 1: #If there is only one last ghost left
                for action in Actions:#for each action
                    Score.append(self.minimax(CurrentState.generateSuccessor(Agent, action), depth - 1, 0, True)[0]) #add the value of the score to the end of the score list
            else: #if there are more than one ghosts
                for action in Actions: #for each action
                    Score.append(self.minimax(CurrentState.generateSuccessor(Agent, action), depth, Agent + 1, False)[0]) #add the value of the score to the end of the score list
            BestScore = min(Score) #since the maximizing player doesnt play ,we need the minimum score
            BestActions = [] #create and empty list to hold each index has the best score in order to track the actions that lead to that score
            for i in range(len(Score)):#for each item in score
                if Score[i] == BestScore:#if it has the best value,which now is the minimum value since the maximizing player(pacman) doesnt play
                    BestActions.append(i)#append it to the list
            return BestScore, Actions[random.choice(BestActions)] #return the score plus the actions that lead to that Score

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
        return self.minimax(gameState, self.depth * 2, 0, True)[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def minimax(self, state, depth, a, b, Agent = 0, MaximizingPlayer = True):
        Actions = state.getLegalActions(Agent) #get the legal actions of the agent
        if MaximizingPlayer: #if the maximizing player is playing(pacman)
            if depth == 0 or state.isWin() or state.isLose(): #if the depth of the tree is 0 or pacman won or pacman lost
                return self.evaluationFunction(state), Directions.STOP #return the evaluation function for the current state,plus the Stop Direction    
            BestScore = -1e100 #Set the score to -Infinity
            BestActions = [] #a list that hold the actions it needs to reach the best score
            for action in Actions: #for each action
                Score = self.minimax(state.generateSuccessor(Agent, action), depth - 1, a, b, 1, False)[0]#set the score variable at the current value of the minimax algorithm
                if Score > BestScore: #if the current score is better than the best score thus far
                    BestScore = Score #then the best score is the current one
                    BestActions = [action] #the best set of actions are the current actions
                elif Score == BestScore: #if the best score is the same as the current one
                    BestActions.append(action) #add at the end of the list of actions the current one
                if BestScore > b: break #Best Score is better than b ,we prune the rest of the actions
                a = max(a, Score)#in a we get the max value between itself and the current score
            return BestScore, random.choice(BestActions) #return the best score plus the list of actions
        else: #if the maximizing player doesnt play
            if depth == 0 or state.isWin() or state.isLose(): #if the depth of the tree is 0 or pacman won or pacman lost
                return self.evaluationFunction(state), Directions.STOP #return the evaluation function for the current state,plus the Stop Direction    
            BestScore = 1e100 #best score now is Infinity,since the maximizing player doesnt play
            BestActions = [] #a list that hold the actions it needs to reach the best score,which now is the minimum
            if Agent == state.getNumAgents() - 1: # if there is only one ghost left
                for action in Actions: #for each action
                    score = self.minimax(state.generateSuccessor(Agent, action), depth - 1, a, b, 0, True)[0]#set the score variable at the current value of the minimax algorithm
                    if score < BestScore: #if the score is less than the best one
                        BestScore = score #the new best score is the minimum between the two
                        BestActions = [action] #the action to reach that score is added to the list
                    elif score == BestScore: #if the current score is the same as the best one
                        BestActions.append(action) #append it to the end of the actions to take list
                    if a > BestScore: break #if a is better than the best score,prune the rest of the actions
                    b = min(b, score) #b is the minimum between its current value and the current score
            else: #if there are more than one ghosts left
                for action in Actions:
                    score = self.minimax(state.generateSuccessor(Agent, action), depth, a, b, Agent + 1, False)[0]
                    if score < BestScore:  #if the score is lower than the best one
                        BestScore = score #the new best score is the minimum between the two
                        BestActions = [action] #the best action is now the current one
                    elif score == BestScore: #if the current score is the same as the best one
                         BestActions.append(action) #append it to the end of the actions to take list
                    if a > BestScore: break #if a is better than the best score,prune the rest of the actions
                    b = min(b, score) #b again is the minimum of the two
            return BestScore, random.choice(BestActions) # return the best score plus the set of actions it takes to reach it
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, self.depth * 2, -1e100, 1e100, 0, True)[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def Expectimax(self, depth, agent, State):
        Actions = State.getLegalActions(agent)  #get the legal actions of the agent
        BestScore = []#create a list that holds all the scores
        if State.isWin() or State.isLose() or depth > self.depth: #if endgame conditions are met
            return self.evaluationFunction(State) #return the evaluation function for the current state
        if Directions.STOP in Actions:
            Actions.remove(Directions.STOP) #remove the stop directions from the actions
        for action in Actions: #for each action
            Child = State.generateSuccessor(agent, action) #find all the successors of the current state
            if agent + 1 >= State.getNumAgents(): #if the pacman is not the current agent
               BestScore.append(self.Expectimax(depth + 1, 0, Child)) #find the scores of pacman(maximizing) for the next depth and the next move
            else: #else 
                BestScore.append(self.Expectimax(depth, agent + 1, Child)) #find the scores of the current depth for the other agents (not pacman)
        if agent == 0: #if the agent is pacman
            if(depth == 1):#if we are at the root of the tree
                Score = max(BestScore) #the best score is the maximum
                for i in range(len(BestScore)): #search the score list for the best score
                    if (BestScore[i] == Score): #if the best score is found
                       return Actions[i] #return the actions it took to reach it
            else: #calculate a probable value
                ProbableValue = max(BestScore) #the value of the upper node is the maximum of the score list
        else:
            s = sum(BestScore) #get the sum of the leaves
            l = len(BestScore) #get the number of leaves
            ProbableValue = float(s/l) #get the probable value of the node
        return ProbableValue

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.Expectimax(1, 0, gameState)
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    FoodValue = 100.0 #base score of food
    CapsuleValue = 150.0 #base score of capsules
    GhostValue =  -200.0 #base score of ghosts
    FoodDecay = 0.2 #food decay factor
    CapsuleDecay = 0.2 #fcapsule decay factor
    GhostDecay = 0.1 #ghost decay factor
    score = 0.0 #Initialize the score
    CurrentPosition = currentGameState.getPacmanPosition() #Get the current pacman position
    score += FoodValue * currentGameState.hasFood(CurrentPosition[0], CurrentPosition[1]) #update the value if pacman has found food
    FoodList = currentGameState.getFood().asList() #get all the food as a list
    for food in FoodList:
        score -= FoodValue * (1 - exp(-1.0 * FoodDecay * util.manhattanDistance(CurrentPosition, food))) #apply the decay factor for each food left
    CapsuleList = currentGameState.data.capsules #Get all the capsules
    for capsule in CapsuleList:
        score -= CapsuleValue * (1 - exp(-1.0 * CapsuleDecay * util.manhattanDistance(CurrentPosition, capsule))) #apply the decay factor for each capsules left
    GhostList = currentGameState.getGhostPositions() #get all the ghost positions
    for ghost in GhostList:
        score += GhostValue * exp(-1.0 * GhostDecay * util.manhattanDistance(CurrentPosition, ghost)) #apply the decay factor for each ghost left
        if util.manhattanDistance(CurrentPosition, ghost) < 2: #if pthe real distance between a ghost and pacman is less than 2 return minus infinity (pacman lost)
            score -= 1e6 #score is minus infinity
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
