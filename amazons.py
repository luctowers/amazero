import numpy as np
import random

class Player:
    count = 2
    WHITE, BLACK, = range(count)

    def alternate(player):
        if player == Player.WHITE:
            return Player.BLACK
        elif player == Player.BLACK:
            return Player.WHITE

class Piece:
    count = 4
    NONE, ARROW, WHITE_QUEEN, BLACK_QUEEN = range(count)

    @staticmethod
    def queen_of_player(player):
        if player == Player.WHITE:
            return Piece.WHITE_QUEEN
        elif player == Player.BLACK:
            return Piece.BLACK_QUEEN


class ActionType:
    count = 3
    QUEEN_PICK, QUEEN_MOVE, ARROW_SHOT = range(count)

class Direction:
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (-1, 1)
    DOWN_LEFT = (1, -1)
    DOWN_RIGHT = (1, 1)
    orthogonal = [UP, DOWN, LEFT, RIGHT]
    diagonal = [UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT]
    all = orthogonal + diagonal

class State:
    def __init__(self, board_shape):
        assert (len(board_shape) == 2)
        self.board = np.zeros(board_shape)
        self.queens = [set() for i in range(Player.count)]
        self.player_to_move = Player.WHITE
        self.next_action_type = ActionType.QUEEN_PICK
        self.last_action_of_type = [None for i in range(ActionType.count)]

    @staticmethod
    def standard():
        state = State((10, 10))
        state.place_queen(Player.BLACK, (3, 0))
        state.place_queen(Player.BLACK, (0, 3))
        state.place_queen(Player.BLACK, (0, 6))
        state.place_queen(Player.BLACK, (3, 9))
        state.place_queen(Player.WHITE, (6, 0))
        state.place_queen(Player.WHITE, (9, 3))
        state.place_queen(Player.WHITE, (9, 6))
        state.place_queen(Player.WHITE, (6, 9))
        return state

    def get_legal_actions(self):
        match self.next_action_type:
            case ActionType.QUEEN_PICK:
                return [
                    q for q in self.queens[self.player_to_move]
                    if self.has_liberties(q)
                ]
            case ActionType.QUEEN_MOVE:
                return self.trace(self.last_action_of_type[ActionType.QUEEN_PICK])
            case ActionType.ARROW_SHOT:
                return self.trace(self.last_action_of_type[ActionType.QUEEN_MOVE])
    
    def do_action(self, action):
        match self.next_action_type:
            case ActionType.QUEEN_PICK:
                assert self.board[action] == Piece.queen_of_player(self.player_to_move)
                self.last_action_of_type[ActionType.QUEEN_PICK] = action
                self.next_action_type = ActionType.QUEEN_MOVE
            case ActionType.QUEEN_MOVE:
                assert self.board[action] == Piece.NONE
                queen_to_move = self.last_action_of_type[ActionType.QUEEN_PICK]
                self.queens[self.player_to_move].remove(queen_to_move)
                self.board[queen_to_move] = Piece.NONE
                self.place_queen(self.player_to_move, action)
                self.last_action_of_type[ActionType.QUEEN_MOVE] = action
                self.next_action_type = ActionType.ARROW_SHOT
            case ActionType.ARROW_SHOT:
                assert self.board[action] == Piece.NONE
                self.board[action] = Piece.ARROW
                self.last_action_of_type[ActionType.ARROW_SHOT] = action
                self.next_action_type = ActionType.QUEEN_PICK
                self.player_to_move = Player.alternate(self.player_to_move)
    
    def has_liberties(self, position):
        py, px = position
        width, height = self.board.shape
        for dy, dx in Direction.all:
            y, x = py+dy, px+dx
            if x >= 0 and x < width and y >= 0 and y < height:
                if self.board[y][x] == Piece.NONE:
                    return True
        return False
    
    def trace(self, position):
        result = []
        py, px = position
        width, height = self.board.shape
        for dy, dx in Direction.all:
            y, x = py+dy, px+dx
            while x >= 0 and x < width and y >= 0 and y < height:
                if self.board[y][x] == Piece.NONE:
                    result.append((y, x))
                else:
                    break
                y, x = y+dy, x+dx
        return result
    
    def place_queen(self, player, position):
        assert self.board[position] == Piece.NONE
        self.board[position] = Piece.queen_of_player(player)
        self.queens[player].add(position)

def simulate():
    s = State.standard()
    a = s.get_legal_actions()
    while len(a) > 0:
        s.do_action(random.choice(a))
        a = s.get_legal_actions()

if __name__ == "__main__":
    import timeit
    print(1000/timeit.timeit(simulate, number=1000))
