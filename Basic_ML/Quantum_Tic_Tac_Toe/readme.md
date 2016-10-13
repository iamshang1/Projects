# Quantum Tic Tac Toe

This exercise utilizes the Monte Carlo Tree Search algorithm to create an intelligent agent that can play quantum tic-tac-toe.

Quantum tic-tac-toe is a variant of tic-tac-toe that incorporates concepts from quantum mechanics. Quantum tic-tac-toe is exponentially more complex than 
regular tic-tac-toe. On each move, each player chooses two squares, either of which could end up being the player's true move. When the board reaches a state in which
paired moves form a closed circuit, each pair in the circuit collapses and forces one of the moves to become the true move. Whereas in regular tic-tac-toe, 
there at most 9 possible moves per turn, in quantum tic-tac-toe, there are up to 72 possible moves per turn.

Because of the sheer number of possible board configurations in quantum tic-tac-toe, regular minimax is an unfeasible strategy to build an intelligent game-playing agent
because it would take too long to calculate the best move in each situation. Instead, monte carlo tree search is used to build up a tree of possible moves and the probability
of winning associated with each move.

The monte carlo tree search algorithm used for this exercise works as follows:
 - Given a board state, find all possible moves at that board state.
 - If the board state has never been seen before, choose a random move.
 - If the board state has been visited before, choose the move with the highest upper confidence bound, which is calculated as (score of move)/(times move has been tried + 1) + sqrt(2 * ln(times board state has been visited)/(times move has been tried)).
 - Play until the game is over. If the player wins, add 1 to the score of all moves played. If the player loses, substract 1 from the score of all moves played. Tie games do not affect the score of any moves.
 - Repeat until a substantial tree of moves and their corresponding statistics have been created.

The bigger the tree built by monte carlo tree search, the better the agent becomes at selecting the best move given a board state.

## Results

The agent trained by monte carlo tree search was played against another agent which always made random moves. The results below indicate the number of games the mcts agent won out of 100 games
using a tree built from X iterations of monte carlo tree search.

**After 10,000 games:**  
Wins: 47  
Ties: 27  
Losses: 26

**After 100,000 games:**  
Wins: 52  
Ties: 24  
Losses: 24

**After 1,000,000 games:**  
Wins: 59  
Ties: 26  
Losses: 15

**After 5,000,000 games:**  
Wins: 64  
Ties: 27  
Losses: 9

We see that monte carlo tree search will create a game playing agent that can effectively win at quantum tic-tac-toe most of the time. However, due to the complexity of 
quantum tic-tac-toe, the agent still loses some of the time because there are too many possible outcomes for monte carlo tree search to fully explore.


