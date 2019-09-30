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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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

    def DFMinimax(self, depth, gameState, currAgent):
        """
        DFMiniMax(n, Player) //return Utility of state n given that
        //Player is MIN or MAX
        If n is TERMINAL
        Return V(n) //Return terminal states utility
                    //(V is specified as part of game)
        //Apply Player's moves to get successor states.
        ChildList = n.Successors(Player)
        If Player == MIN
          return minimum of DFMiniMax(c, MAX) over c \in ChildList
        Else //Player is MAX
          return maximum of DFMiniMax(c, MIN) over c \in ChildList
        """
        actions = gameState.getLegalActions(currAgent)
        if depth > self.depth or len(actions) == 0:
            return (self.evaluationFunction(gameState),Directions.STOP)
        if currAgent == 0: # MAX node
            maxVal = []
            for action in actions:
                state = gameState.generateSuccessor(currAgent,action)
                maxVal.append((self.DFMinimax(depth,state,1)[0],action))
            return max(maxVal)
        else: # MIN node
            minVal = []
            for action in actions:
                state = gameState.generateSuccessor(currAgent,action)
                if currAgent == gameState.getNumAgents() - 1:
                    minVal.append((self.DFMinimax(depth+1,state,0)[0],action))
                else: # one by one action
                    minVal.append((self.DFMinimax(depth,state,currAgent+1)[0],action))
            return min(minVal)

    def getAction(self, gameState):
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
        """
        _, action = self.DFMinimax(1,gameState,0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def DFMinimax(self, depth, gameState, currAgent, alpha, beta):
        """
        DFMiniMax(n, Player) //return Utility of state n given that
        //Player is MIN or MAX
        If n is TERMINAL
        Return V(n) //Return terminal states utility
                    //(V is specified as part of game)
        //Apply Player's moves to get successor states.
        ChildList = n.Successors(Player)
        If Player == MIN
          return minimum of DFMiniMax(c, MAX) over c \in ChildList
        Else //Player is MAX
          return maximum of DFMiniMax(c, MIN) over c \in ChildList
        """
        actions = gameState.getLegalActions(currAgent)
        if depth > self.depth or len(actions) == 0:
            return (self.evaluationFunction(gameState),Directions.STOP)
        if currAgent == 0: # MAX node
            val = (-0x3f3f3f3f,Directions.STOP)
            for action in actions:
                state = gameState.generateSuccessor(currAgent,action)
                val = max(val,(self.DFMinimax(depth,state,1,alpha,beta)[0],action))
                if val[0] > beta:
                    return val
                alpha = max(alpha,val[0])
            return val
        else: # MIN node
            val = (0x3f3f3f3f,Directions.STOP)
            for action in actions:
                state = gameState.generateSuccessor(currAgent,action)
                if currAgent == gameState.getNumAgents() - 1:
                    val = min(val,(self.DFMinimax(depth+1,state,0,alpha,beta)[0],action))
                else: # one by one action
                    val = min(val,(self.DFMinimax(depth,state,currAgent+1,alpha,beta)[0],action))
                if val[0] < alpha:
                    return val
                beta = min(beta,val[0])
            return val

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, action = self.DFMinimax(1,gameState,0,-0x3f3f3f3f,0x3f3f3f3f)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

