import itertools
import copy
import re
import math
import random
import shelve

class board(object):
    def __init__(self,humans=0,AI1=None,AI2=None):
        self.board = {(1,1):[],\
                      (1,2):[],\
                      (1,3):[],\
                      (2,1):[],\
                      (2,2):[],\
                      (2,3):[],\
                      (3,1):[],\
                      (3,2):[],\
                      (3,3):[]}
        self.turn = 1
        self.score = 0
        self.end = False
        self.humans = humans
        self.AI1 = AI1
        self.AI2 = AI2
    def _possible_moves(self):
        moves = []
        for key,val in self.board.iteritems():
            if val != 'X' and val != 'O':
                moves.append(key)
        return moves
    def playermove(self,square1,square2):
        moves = self._possible_moves()
        if square1 not in moves:
            raise Exception('invalid move')
        self.board[square1].append(self.turn)
        moves.remove(square1)
        if square2 not in moves:
            raise Exception('invalid move')
        self.board[square2].append(self.turn)
    def _find_circuit(self,start,current,traversed,board_copy):
        traversed.append(current)
        circuit = None
        while board_copy[current]:
            entry = board_copy[current].pop(0)
            for key,val in board_copy.iteritems():
                try:
                    if entry in val:
                        if key == start:
                            return traversed
                        circuit = self._find_circuit(start,key,traversed,board_copy)
                        if circuit:
                            return circuit
                except:
                    pass
        return False
    def _prompt_circuit(self,choice1,choice2):
        choice = raw_input("Choose between %s and %s: " % (str(choice1)[:3]+str(choice1)[4:],str(choice2)[:3]+str(choice2)[4:]))
        while choice != (str(choice1)[:3]+str(choice1)[4:]) and choice != (str(choice2)[:3]+str(choice2)[4:]):
            print "invalid choice"
            choice = raw_input("Choose between %s and %s: " % (str(choice1),str(choice2)))
        choice = (int(choice[1]),int(choice[3]))
        return choice
    def _ai_circuit(self,choice1,choice2,AI):
        choice = AI.break_circuit(choice1,choice2)
        return choice
    def _break_circuit(self,square,turn):
        self.board[square].remove(turn)
        for entry in self.board[square]:
            for key,val in self.board.iteritems():
                try:
                    if entry in val and key != square:
                        pair = key
                        self._break_circuit(pair,entry)
                except:
                    pass
        if turn % 2 == 1:
            self.board[square] = 'X'
        elif turn % 2 == 0:
            self.board[square] = 'O'
    def _evaluate_board(self):
        if (self.board[(1,1)]=='X' and self.board[(1,2)]=='X' and self.board[(1,3)]=='X'):
            self.score += 1
            self.end = True
        if (self.board[(2,1)]=='X' and self.board[(2,2)]=='X' and self.board[(2,3)]=='X'):
            self.score += 1
            self.end = True
        if (self.board[(3,1)]=='X' and self.board[(3,2)]=='X' and self.board[(3,3)]=='X'):
            self.score += 1
            self.end = True
        if (self.board[(1,1)]=='X' and self.board[(2,1)]=='X' and self.board[(3,1)]=='X'):
            self.score += 1
            self.end = True
        if (self.board[(1,2)]=='X' and self.board[(2,2)]=='X' and self.board[(3,2)]=='X'):
            self.score += 1
            self.end = True
        if (self.board[(1,3)]=='X' and self.board[(2,3)]=='X' and self.board[(3,3)]=='X'):
            self.score += 1
            self.end = True
        if (self.board[(1,1)]=='X' and self.board[(2,2)]=='X' and self.board[(3,3)]=='X'):
            self.score += 1
            self.end = True
        if (self.board[(1,3)]=='X' and self.board[(2,2)]=='X' and self.board[(3,1)]=='X'):
            self.score += 1
            self.end = True
        if (self.board[(1,1)]=='O' and self.board[(1,2)]=='O' and self.board[(1,3)]=='O'):
            self.score += -1
            self.end = True
        if (self.board[(2,1)]=='O' and self.board[(2,2)]=='O' and self.board[(2,3)]=='O'):
            self.score += -1
            self.end = True
        if (self.board[(3,1)]=='O' and self.board[(3,2)]=='O' and self.board[(3,3)]=='O'):
            self.score += -1
            self.end = True
        if (self.board[(1,1)]=='O' and self.board[(2,1)]=='O' and self.board[(3,1)]=='O'):
            self.score += -1
            self.end = True
        if (self.board[(1,2)]=='O' and self.board[(2,2)]=='O' and self.board[(3,2)]=='O'):
            self.score += -1
            self.end = True
        if (self.board[(1,3)]=='O' and self.board[(2,3)]=='O' and self.board[(3,3)]=='O'):
            self.score += -1
            self.end = True
        if (self.board[(1,1)]=='O' and self.board[(2,2)]=='O' and self.board[(3,3)]=='O'):
            self.score += -1
            self.end = True
        if (self.board[(1,3)]=='O' and self.board[(2,2)]=='O' and self.board[(3,1)]=='O'):
            self.score += -1
            self.end = True
        filled_squares = 0
        for key,val in self.board.iteritems():
            if val =='X' or val =='O':
                filled_squares += 1
        if filled_squares >= 8:
            self.end = True
    def display_board(self):
        s1 = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
        s2 = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
        s3 = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
        s4 = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
        s5 = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
        s6 = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
        s7 = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
        s8 = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
        s9 = [' ',' ',' ',' ',' ',' ',' ',' ',' ']
        for s,b in zip([s1,s2,s3,s4,s5,s6,s7,s8,s9],[self.board[(1,1)],self.board[(1,2)],self.board[(1,3)],\
            self.board[(2,1)],self.board[(2,2)],self.board[(2,3)],self.board[(3,1)],self.board[(3,2)],self.board[(3,3)]]):
            if b == 'X':
                for i in range(len(s)):
                    s[i] = 'X'
            elif b == 'O':
                for i in range(len(s)):
                    s[i] = 'O'
            else:
                for i in range(len(b)):
                    s[i] = str(b[i])
        print 'turn %i' % self.turn
        print '%s %s %s | %s %s %s | %s %s %s' % (s1[0],s1[1],s1[2],s2[0],s2[1],s2[2],s3[0],s3[1],s3[2])
        print '%s %s %s | %s %s %s | %s %s %s' % (s1[3],s1[4],s1[5],s2[3],s2[4],s2[5],s3[3],s3[4],s3[5])
        print '%s %s %s | %s %s %s | %s %s %s' % (s1[6],s1[7],s1[8],s2[6],s2[7],s2[8],s3[6],s3[7],s3[8])
        print '---------------------'
        print '%s %s %s | %s %s %s | %s %s %s' % (s4[0],s4[1],s4[2],s5[0],s5[1],s5[2],s6[0],s6[1],s6[2])
        print '%s %s %s | %s %s %s | %s %s %s' % (s4[3],s4[4],s4[5],s5[3],s5[4],s5[5],s6[3],s6[4],s6[5])
        print '%s %s %s | %s %s %s | %s %s %s' % (s4[6],s4[7],s4[8],s5[6],s5[7],s5[8],s6[6],s6[7],s6[8])
        print '---------------------'
        print '%s %s %s | %s %s %s | %s %s %s' % (s7[0],s7[1],s7[2],s8[0],s8[1],s8[2],s9[0],s9[1],s9[2])
        print '%s %s %s | %s %s %s | %s %s %s' % (s7[3],s7[4],s7[5],s8[3],s8[4],s8[5],s9[3],s9[4],s9[5])
        print '%s %s %s | %s %s %s | %s %s %s' % (s7[6],s7[7],s7[8],s8[6],s8[7],s8[8],s9[6],s9[7],s9[8])
    def _prompt_move(self):
        moves = self._possible_moves()
        square1 = raw_input("Choose your first move in format (row,column): ")
        while square1 not in moves:
            while not re.match('\([1-9],[1-9]\)',square1):
                print "invalid choice"
                square1 = raw_input("Choose your first move in format (row,column): ")
            square1 = (int(square1[1]),int(square1[3]))
            if square1 not in moves:
                print "invalid choice"
                square1 = raw_input("Choose your first move in format (row,column): ")
        moves.remove(square1)
        square2 = raw_input("Choose your second move in format (row,column): ")
        while square2 not in moves:
            while not re.match('\([1-9],[1-9]\)',square2):
                print "invalid choice"
                square2 = raw_input("Choose your second move in format (row,column): ")
            square2 = (int(square2[1]),int(square2[3]))
            if square2 not in moves:
                print "invalid choice"
                square2 = raw_input("Choose your second move in format (row,column): ")
        return square1,square2
    def _ai_move(self,AI):
        moves = self._possible_moves()
        square1,square2 = AI.choose_move(moves)
        return square1,square2
    def play_game(self):
        if self.humans == 1:
            turn = raw_input("Enter '1' to go first or '2' to go second: ") 
            while turn != "1" and turn != "2":
                print "invalid choice"
                turn = raw_input("Enter '1' to go first or '2' to go second: ") 
            turn = int(turn)-1
        while self.end == False:
            if self.humans >= 1:
                self.display_board()
            if self.humans == 1 and (self.turn-turn)%2 == 1:
                square1,square2 = self._prompt_move()
                self.AI1.opponent_move(square1,square2)
            elif self.humans == 1 and (self.turn-turn)%2 == 0:
                square1,square2 = self._ai_move(self.AI1)
            elif self.humans == 2:
                square1,square2 = self._prompt_move()
            elif self.humans == 0 and self.turn%2 == 1:
                square1,square2 = self._ai_move(self.AI1)
                self.AI2.opponent_move(square1,square2)
            else:
                square1,square2 = self._ai_move(self.AI2)
                self.AI1.opponent_move(square1,square2)
            self.playermove(square1,square2)
            circuit1 = self._find_circuit(square1,square1,[],copy.deepcopy(self.board))
            if circuit1:
                if self.humans == 1 and (self.turn-turn)%2 == 1:
                    choice = self._prompt_circuit(square1,square2)
                    self.AI1.opponent_break_circuit(choice)
                elif self.humans == 1 and (self.turn-turn)%1 == 0:
                    choice = self._ai_circuit(square1,square2,self.AI1)
                elif self.humans == 2:
                    choice = self._prompt_circuit(square1,square2)
                elif self.humans == 0 and self.turn%2 == 1:
                    choice = self._ai_circuit(square1,square2,self.AI1)
                    self.AI2.opponent_break_circuit(choice)
                else:
                    choice = self._ai_circuit(square1,square2,self.AI2)
                    self.AI1.opponent_break_circuit(choice)
                self._break_circuit(choice,self.turn)
            else:
                circuit2 = self._find_circuit(square2,square2,[],copy.deepcopy(self.board))
                if circuit2:
                    if self.humans == 1 and (self.turn-turn)%2 == 1:
                        choice = self._prompt_circuit(square1,square2)
                        self.AI1.opponent_break_circuit(choice)
                    elif self.humans == 1 and (self.turn-turn)%1 == 0:
                        choice = self._ai_circuit(square1,square2,self.AI1)
                    elif self.humans == 2:
                        choice = self._prompt_circuit(square1,square2)
                    elif self.humans == 0 and self.turn%2 == 1:
                        choice = self._ai_circuit(square1,square2,self.AI1)
                        self.AI2.opponent_break_circuit(choice)
                    else:
                        choice = self._ai_circuit(square1,square2,self.AI2)
                        self.AI1.opponent_break_circuit(choice)
                    self._break_circuit(choice,self.turn)
            self._evaluate_board()
            self.turn += 1     
        if self.humans >= 1:
            self.display_board()
            if self.score > 0:
                print "Player 1 wins"
            elif self.score < 0:
                print "Player 2 wins"
            else:
                print "Tie game"
        if self.humans == 1 and turn == 1:
            self.AI1.backpropogate(-self.score)
            self.AI1.reset()
        elif self.humans == 1 and turn == 2:
            self.AI1.backpropogate(self.score)
            self.AI1.reset()
        elif self.humans == 0:
            self.AI1.backpropogate(self.score)
            self.AI2.backpropogate(-self.score)
            self.AI1.reset()
            self.AI2.reset()
            self.AI1,self.AI2 = self.AI2,self.AI1
        self.reset()
    def _simulate_game(self):
        pass
    def reset(self):
        self.board = {(1,1):[],
                      (1,2):[],
                      (1,3):[],
                      (2,1):[],
                      (2,2):[],
                      (2,3):[],
                      (3,1):[],
                      (3,2):[],
                      (3,3):[]}
        self.turn = 1
        self.score = 0
        self.end = False

class monte_carlo_tree(object):
    def __init__(self):
        self.entries = {}
        self.entries['total_games'] = 0
    def choose_move(self,valid_moves,pointer,path):
        nodes = []
        max_upper_bound = -float("inf")
        total_visits = 0
        for tuple in itertools.combinations(valid_moves,2):
            t_pointer = re.sub(r'\D','',str(tuple))
            t_pointer2 = re.sub(r'\D','',str((tuple[1],tuple[0])))
            try:
                score = self.entries[pointer+t_pointer][0]
                visits = self.entries[pointer+t_pointer][1]
                total_visits += visits
                nodes.append(t_pointer)
            except:
                try:
                    score = self.entries[pointer+t_pointer2][0]
                    visits = self.entries[pointer+t_pointer2][1]
                    total_visits += visits
                    nodes.append(t_pointer2)
                except:
                    self.entries[pointer+t_pointer] = (0,0)
                    nodes.append(t_pointer)
        if self.entries['total_games'] == 0:
            choice = random.choice(nodes)
            choice1 = (int(choice[0]),int(choice[1])) 
            choice2 = (int(choice[2]),int(choice[3]))
            path.append(pointer+choice)
            pointer = pointer+choice
            return choice1,choice2,pointer,path
        for n in nodes:
            score = self.entries[pointer+n][0]
            visits = self.entries[pointer+n][1]
            upper_bound = float(score)/(visits+1.) + (2*math.log(total_visits+1.)/(visits+1.))**0.5
            if upper_bound > max_upper_bound:
                choice = n
                choice1 = (int(n[0]),int(n[1])) 
                choice2 = (int(n[2]),int(n[3]))
                max_upper_bound = upper_bound
        path.append(pointer+choice)
        pointer = pointer+choice
        return choice1,choice2,pointer,path
    def opponent_move(self,square1,square2,pointer):
        tuple = (square1,square2)
        t_pointer = re.sub(r'\D','',str(tuple))
        t_pointer2 = re.sub(r'\D','',str((tuple[1],tuple[0])))
        try:
            n = self.entries[pointer+t_pointer]
            pointer += t_pointer
        except:
            try:
                n = self.entries[pointer+t_pointer2]
                pointer += t_pointer2
            except:
                self.entries[pointer+t_pointer] = (0,0)
                pointer += t_pointer
        return pointer
    def break_circuit(self,choice1,choice2,pointer,path):
        t_pointer1 = re.sub(r'\D','',str(choice1))+'c'
        t_pointer2 = re.sub(r'\D','',str(choice2))+'c'
        max_upper_bound = -float("inf")
        total_visits = 0
        for t_pointer in (t_pointer1,t_pointer2):
            try:
                score = self.entries[pointer+t_pointer][0]
                visits = self.entries[pointer+t_pointer][1]
                total_visits += visits
            except:
                self.entries[pointer+t_pointer] = (0,0)
        if self.entries['total_games'] == 0:
            n = random.choice([t_pointer1,t_pointer2])
            choice = (int(n[0]),int(n[1])) 
            path.append(pointer+n)
            pointer = pointer+n
            return choice,pointer,path
        for t_pointer in (t_pointer1,t_pointer2):
            score = self.entries[pointer+t_pointer][0]
            visits = self.entries[pointer+t_pointer][1]
            upper_bound = float(score)/(visits+1.) + (2*math.log(total_visits+1.)/(visits+1.))**0.5
            if upper_bound > max_upper_bound:
                n = t_pointer
                choice = (int(t_pointer[0]),int(t_pointer[1])) 
                max_upper_bound = upper_bound
        path.append(pointer+n)
        pointer = pointer+n
        return choice,pointer,path
    def opponent_break_circuit(self,choice,pointer):
        t_pointer = re.sub(r'\D','',str(choice))+'c'
        try:
            n = self.entries[pointer+t_pointer]
            pointer += t_pointer
        except:
            self.entries[pointer+t_pointer] = (0,0)
            pointer += t_pointer
        return pointer
    def backpropogate(self,result,path):
        for p in path:
            score = self.entries[p][0]
            visits = self.entries[p][1]
            if result > 0:
                score += 1
            elif result < 0:
                score -= 1
            visits += 1
            self.entries[p] = (score,visits)
    def reset(self):
        self.entries['total_games'] += 1

class dumb_ai(object):
    def __init__(self):
        self.wins = 0
        self.games = 0
    def choose_move(self,valid_moves):
        square1 = random.choice(valid_moves)
        valid_moves.remove(square1)
        square2 = random.choice(valid_moves)
        return square1,square2
    def opponent_move(self,square1,square2):
        pass
    def break_circuit(self,choice1,choice2):
        choices = [choice1,choice2]
        choice = random.choice(choices)
        return choice
    def opponent_break_circuit(self,choice):
        pass
    def backpropogate(self,result):
        self.games += 1
        if result > 0:
            self.wins += 1
    def reset(self):
        pass
        
class mcts_ai(object):
    def __init__(self,mcts):
        self.mcts = mcts
        self.wins = 0
        self.games = 0
        self.path = []
        self.pointer = ''
    def choose_move(self,valid_moves):
        square1,square2,self.pointer,self.path = self.mcts.choose_move(valid_moves,self.pointer,self.path)
        return square1,square2
    def opponent_move(self,square1,square2):
        self.pointer = self.mcts.opponent_move(square1,square2,self.pointer)
    def break_circuit(self,choice1,choice2):
        choice,self.pointer,self.path = self.mcts.break_circuit(choice1,choice2,self.pointer,self.path)
        return choice
    def opponent_break_circuit(self,choice):
        self.pointer = self.mcts.opponent_break_circuit(choice,self.pointer)
    def backpropogate(self,result):
        self.games += 1
        if result > 0:
            self.wins += 1
        self.mcts.backpropogate(result,self.path)
    def reset(self):
        self.mcts.reset()
        self.path = []
        self.pointer = ''


mcts = monte_carlo_tree()
AI1 = mcts_ai(mcts)
entries = shelve.open('mcts.shelve')
mcts.entries = entries
try:
    test = mcts.entries['total_games']
except:
    mcts.entries['total_games'] = 0

for j in range(1000):
    AI2 = dumb_ai()
    game = board(humans=0,AI1=AI1,AI2=AI2)
    for i in range(5000):
        game.play_game()
        print "games played: %i" % (mcts.entries['total_games'])
    AI2 = mcts_ai(mcts)
    game = board(humans=0,AI1=AI1,AI2=AI2)
    for i in range(2500):
        game.play_game()
        print "games played: %i" % (mcts.entries['total_games'])
    AI1.wins = 0
    AI2 = dumb_ai()
    game = board(humans=0,AI1=AI1,AI2=AI2)
    for i in range(100):
        print "round %i" % (i+1)
        game.play_game()
    with open("scores.txt","a") as f:
        f.write("mcts AI wins out of 100 games after %i iterations: %i\n" % (mcts.entries['total_games'],AI1.wins))
        f.write("random choice AI wins out of 100 games after %i iterations: %i\n" % (mcts.entries['total_games'],AI2.wins))
    entries.sync()
entries.close()