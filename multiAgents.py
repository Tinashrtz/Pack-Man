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
import math

class ReflexAgent(Agent):
    def getAction(self, gameState: GameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]
    
    
    def manhattan_distance(self, point1, point2):
        return (abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]))
    
    
    def evaluationFunction(self, currentGameState: GameState, action):

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Constants for weights
        FOOD_WEIGHT = 25
        GHOST_WEIGHT = -100
        SCARED_GHOST_WEIGHT = 100
        DISTANCE_TO_GHOST_THRESHOLD = 10

        # Evaluate distance to nearest food
        food_distances = [self.manhattan_distance(newPos, food) for food in newFood.asList()]
        min_food_distance = min(food_distances, default=0)

        # Evaluate distance to nearest ghost
        ghost_positions = [ghost.getPosition() for ghost in newGhostStates]
        ghost_distances = [self.manhattan_distance(newPos, ghost_pos) for ghost_pos in ghost_positions]
        min_ghost_distance = min(ghost_distances, default=0)

        # Evaluate if any ghost is scared
        scared_ghost_present = any(scared_time > 0 for scared_time in newScaredTimes)

        # Calculate evaluation score
        evaluation_score = successorGameState.getScore() + \
                        FOOD_WEIGHT / (min_food_distance + 1) + \
                        GHOST_WEIGHT / (min_ghost_distance + 1)
                        
        if min_ghost_distance <= DISTANCE_TO_GHOST_THRESHOLD:
            evaluation_score -= GHOST_WEIGHT *10 #  being too close to ghosts is so dangerous (large weight)

        if scared_ghost_present:
            evaluation_score += SCARED_GHOST_WEIGHT


        return evaluation_score

def scoreEvaluationFunction(currentGameState: GameState):
    return currentGameState.getScore()
 
   
class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.temp = 1.0
        
        
    
class MinimaxAgent(MultiAgentSearchAgent):
    def minimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0: #agent0=pacman (pacman wants to maximize)
            return self.maxValue(gameState, depth) 
        else:
            return self.minValue(gameState, agentIndex, depth) #agent1=ghosts (they want to minimize)
        
    def maxValue(self, state, depth):
        v = float("-inf") #negative infinity
        bestAction = None 
        for action in state.getLegalActions(0):
            successor = state.generateSuccessor(0, action)
            successorValue = self.minimax(successor, 1, depth)[0]
            if successorValue > v:
                v = successorValue
                bestAction = action
        return v, bestAction
    
    def minValue(self, state, agentIndex, depth):
        v = float("inf")
        bestAction = None
        if agentIndex == state.getNumAgents() - 1:
            nextAgentIndex = 0
            nextDepth = depth + 1
        else:
            nextAgentIndex = agentIndex + 1
            nextDepth = depth
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            successorValue = self.minimax(successor, nextAgentIndex, nextDepth)[0]
            if successorValue < v:
                v = successorValue
                bestAction = action
        return v, bestAction
    
    def getAction(self, gameState: GameState):
        bestScore, bestAction = self.minimax(gameState, 0, 0)
        return bestAction



class AlphaBetaAgent(MultiAgentSearchAgent):

    def alphaBeta(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0: #using for pacman
            return self.maxValue(gameState, depth, alpha, beta)
        else: #using for gohsts
            return self.minValue(gameState, agentIndex, depth, alpha, beta)

    def maxValue(self, state, depth, alpha, beta):
        return self.search(state, depth, alpha, beta, True)

    def minValue(self, state, agentIndex, depth, alpha, beta):
        return self.search(state, depth, alpha, beta, False, agentIndex)

    def search(self, state, depth, alpha, beta, maximizingPlayer, agentIndex=None):
        agentIndex = 0 if maximizingPlayer else agentIndex
        v = float("-inf") if maximizingPlayer else float("inf")
        bestAction = None
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            successorValue, _ = self.alphaBeta(successor, (agentIndex + 1) % state.getNumAgents(), depth + (agentIndex == state.getNumAgents() - 1), alpha, beta)
            if maximizingPlayer:
                if successorValue > v:
                    v = successorValue
                    bestAction = action
                alpha = max(alpha, v)
                if v > beta:
                    break
            else:
                if successorValue < v:
                    v = successorValue
                    bestAction = action
                beta = min(beta, v)
                if v < alpha:
                    break
        return v, bestAction

    def getAction(self, gameState: GameState):
        self.temp -= 0.005
        self.temp = max(self.temp, 0.01)
        _, bestAction = self.alphaBeta(gameState, 0, 0, float("-inf"), float("inf"))
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):

    def expectimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            return self.maxValue(gameState, depth, agentIndex)
        else:
            return self.expValue(gameState, depth, agentIndex)

    def maxValue(self, state, depth, agentIndex):
        v = float("-inf")
        bestAction = None
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            successorValue, _ = self.expectimax(successor, (agentIndex + 1) % state.getNumAgents(), depth + (agentIndex == state.getNumAgents() - 1))
            if successorValue > v:
                v = successorValue
                bestAction = action
        return v, bestAction

    def expValue(self, state, depth, agentIndex):
        legalActions = state.getLegalActions(agentIndex)
        numActions = len(legalActions)
        expectedValue = sum(self.expectimax(state.generateSuccessor(agentIndex, action), (agentIndex + 1) % state.getNumAgents(), depth + (agentIndex == state.getNumAgents() - 1))[0] for action in legalActions) / numActions
        return expectedValue, None

    def getAction(self, gameState: GameState):
        _, bestAction = self.expectimax(gameState, 0, 0)
        return bestAction




def betterEvaluationFunction(currentGameState):
    def closest_dot(cur_pos, food_pos):
        return min(util.manhattanDistance(food, cur_pos) for food in food_pos) if food_pos else 1

    def closest_ghost(cur_pos, ghosts):
        return min(util.manhattanDistance(ghost.getPosition(), cur_pos) for ghost in ghosts) if ghosts else 1

    def food_stuff(cur_pos, food_positions):
        return sum(util.manhattanDistance(food, cur_pos) for food in food_positions)

    pacman_pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()

    closest_dot_dist = closest_dot(pacman_pos, food)
    closest_ghost_dist = closest_ghost(pacman_pos, ghosts)
    
    # Adjust score based on dot and ghost distances
    score *= 2 if closest_dot_dist < closest_ghost_dist + 3 else 1
    # Penalize score based on distance to remaining dots
    score -= 0.35 * food_stuff(pacman_pos, food)
    # Give more weight to the presence of large dots (power pellets)
    score -= 50 * len(capsules)
    return score
# Abbreviation
better = betterEvaluationFunction






