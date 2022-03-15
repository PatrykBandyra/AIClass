"""Patryk Bandyra"""

import numpy as np
import vector_2d as v
V = v.Vector

'''
Wolf - max
Sheep - min
Assumption:
if wolf has not yet get to the 8 row but is not stuck
and if sheep cannot move (somehow all sheep get stuck in lower rows and wolf is just wandering around)
then wolf wins automatically
'''


def check_available_moves(player_turn, wolf_pos, sheep_pos):
    '''
    Given the player turn (wolf or sheep), wolf and sheep positions, returns available moves for a given player.
    :param player_turn: 'w' or 's'
    :param wolf_pos: V(x, y)
    :param sheep_pos: [V(x1, y1), V(x2, y2), ...]
    :return available_moves: list for wolf, dictionary for sheep
    '''
    if player_turn == 'w':
        # possible moves for wolf
        possible_moves = [wolf_pos+V(-1, 1), wolf_pos+V(1, 1), wolf_pos+V(-1, -1), wolf_pos+V(1, -1)]
        checked_moves = []
        # check conditions
        for move in possible_moves:
            if 1 <= move.x <= 8 and 1 <= move.y <= 8 and move not in sheep_pos:
                checked_moves.append(move)
        return checked_moves
    if player_turn == 's':
        possible_moves = {pos: [pos+V(-1, -1), pos+V(1, -1)] for pos in sheep_pos}
        checked_moves = {pos: [] for pos in sheep_pos}
        # check conditions
        for sheep_p, sheep_moves in possible_moves.items():
            for move in sheep_moves:
                if 1 <= move.x <= 8 and 1 <= move.y <= 8 and move not in sheep_pos and move != wolf_pos:
                    checked_moves[sheep_p].append(move)
        return checked_moves


def check_if_terminal_state(wolf_pos, sheep_pos):
    '''
    :param wolf_pos: V(x, y)
    :param sheep_pos: [V(x1, y1), V(x2, y2), ...]
    :return: bool value
    '''
    # check if wolf got to the 8 row
    if wolf_pos.y == 8:
        return True
    wolf_possible_moves = check_available_moves('w', wolf_pos, sheep_pos)
    # check if wolf is stuck
    if not wolf_possible_moves:
        return True
    # the least possible state - wolf can move, sheep cannot
    sheep_possible_states = get_states_from_sheep_moves(sheep_pos, check_available_moves('s', wolf_pos, sheep_pos))
    if not sheep_possible_states:
        return True
    return False


def heuristic(wolf_pos, sheep_pos):
    '''
    Assesses game state
    :param wolf_pos: V(x, y)
    :param sheep_pos: [V(x1, y1), V(x2, y2), ...]
    :return grade: int value
    '''
    grade = 0
    # assess wolf position - closer to row 8 = better
    grade += wolf_pos.y
    # assess wolf ability to move - move forward = 3, move backward = 1
    wolf_possible_moves = check_available_moves('w', wolf_pos, sheep_pos)
    for move in wolf_possible_moves:
        if move.y > wolf_pos.y:
            grade += 3
        else:
            grade += 1
    # assess sheep position in relation to each other - 4 sheep in a row = -4, 3 sheep = -3, 2 sheep = -2
    rows = [pos.y for pos in sheep_pos]
    most_frequent_row = max(rows, key=rows.count)
    occurrences = rows.count(most_frequent_row)
    if occurrences >= 2:
        grade -= occurrences
    # assess position in relation to the wolf - blocking forward move = -3, backward move = -1
    for sheep in sheep_pos:
        if sheep + V(-1, -1) == wolf_pos or sheep + V(1, -1) == wolf_pos:
            grade -= 3
        elif sheep + V(1, 1) == wolf_pos or sheep + V(-1, 1) == wolf_pos:
            grade -= 1
    # assess sheep distances from wolf
    # max distance from wolf = 14, d = abs(W.x-O.x) + abs(W.y-O.y)
    points = {2: -12, 4: -10, 6: -8, 8: -6, 10: -4, 12: -2, 14: 0}  # key = distance, value = points
    for sheep in sheep_pos:
        dist = abs(wolf_pos.x - sheep.x) + abs(wolf_pos.y - sheep.y)
        grade += points[dist]
    return grade


def get_states_from_sheep_moves(sheep_pos, sheep_moves):
    sheep_states = []
    for key, positions in sheep_moves.items():
        index = sheep_pos.index(key)
        for pos in positions:
            temp_sheep_pos = sheep_pos.copy()
            temp_sheep_pos[index] = pos
            sheep_states.append(temp_sheep_pos)
    return sheep_states


def minmax(player_turn, wolf_pos, sheep_pos, d):
    '''
    Explores game tree
    :param player_turn: 'w' or 's'
    :param wolf_pos: V(x, y)
    :param sheep_pos: [V(x1, y1), V(x2, y2), ...]
    :param d: search depth
    :return new state:
    '''
    randomness = 0.1

    # if state is terminal (end of recursion)
    if check_if_terminal_state(wolf_pos, sheep_pos):
        # return payout
        if player_turn == 'w':  # current state is terminal and it is wolf turn -> wolf looses - cannot move
            return -100
        # current state is terminal and it is sheep turn - sheep looses - wolf get to the 8 row or sheep cannot move
        else:
            return 100

    # if maximal depth of search is reached (end of recursion) - assess state by heuristic
    if d == 0:
        return heuristic(wolf_pos, sheep_pos)

    # get successors (next potential states) and dive into the tree
    if player_turn == 'w':
        next_states = check_available_moves('w', wolf_pos, sheep_pos)
        grades = []
        for state in next_states:
            grades.append(minmax('s', state, sheep_pos, d-1))
        if d != origin_d:   # if we are still in recursion
            return max(grades)

    else:   # sheep turn
        next_states = get_states_from_sheep_moves(sheep_pos, check_available_moves('s', wolf_pos, sheep_pos))
        # assess new states
        grades = []
        for state in next_states:
            grades.append(minmax('w', wolf_pos, state, d-1))
        if d != origin_d:   # if we are still in recursion
            return min(grades)

    # actual return values
    # choose best player move - wolf always plays optimally
    if player_turn == 'w':
        best = max(grades)
        best_moves_indexes = []     # there might be few best moves
        for i in range(len(grades)):
            if grades[i] == best:
                best_moves_indexes.append(i)
        # choose randomly best move from best moves indexes
        index = np.random.choice(best_moves_indexes)
        return next_states[index]   # returns new wolf position

    else:   # sheep turn
        # allow for randomness during sheep moves
        if np.random.uniform() < randomness:
            index = np.random.randint(0, len(next_states))
            return next_states[index]
        else:
            best_grade = min(grades)
            best_moves_indexes = []
            for i in range(len(grades)):
                if grades[i] == best_grade:
                    best_moves_indexes.append(i)
            # choose randomly best index
            index = np.random.choice(best_moves_indexes)
            return next_states[index]


def display(history):
    '''
    Displays checkerboard for every turn in game
    :param history: history of game states
    '''
    for turn in history:
        final = [['.' for _ in range(8)] for i in range(8)]
        wolf = turn[0]
        sheep = turn[1]
        for y in range(7, -1, -1):
            for x in range(8):
                if wolf.x - 1 == x and wolf.y - 1 == y:
                    final[y][x] = 'W'
                for s in sheep:
                    if s.x - 1 == x and s.y - 1 == y:
                        final[y][x] = 'S'
        for row in range(7, -1, -1):
            print(final[row])
        print()


def play_game(wolf_pos, d):
    '''
    :param wolf_pos: wolf starting position
    :param d: search depth for min-max algorithm
    :return moves_history: history of game states
    '''
    global origin_d
    origin_d = d
    turns = ['w', 's']
    player_turn = 'w'
    move = 0    # moves counter
    sheep_pos = [V(2, 8), V(4, 8), V(6, 8), V(8, 8)]
    history = [[wolf_pos, sheep_pos]]
    print('GAME STARTED')
    while True:
        player_turn = turns[move % 2]
        if player_turn == 'w':
            print('Wolf turn')
            wolf_pos = minmax(player_turn, wolf_pos, sheep_pos, d)
            print(f'Wolf position: {wolf_pos}')
            history.append([wolf_pos, sheep_pos])
            # check if wolf wins - if wolf gets to 8 row or sheep cannot move
            if wolf_pos.y == 8 or not get_states_from_sheep_moves(sheep_pos, check_available_moves('s', wolf_pos, sheep_pos)):
                print('Wolf wins!!!')
                break
        else:
            print('Sheep turn')
            sheep_pos = minmax(player_turn, wolf_pos, sheep_pos, d)
            print(f'Sheep positions: {sheep_pos}')
            history.append([wolf_pos, sheep_pos])
            # check if sheep win - if wolf cannot move
            if not check_available_moves('w', wolf_pos, sheep_pos):
                print('Sheep win!!!')
                break
        move += 1
    return history


if __name__ == '__main__':
    display(play_game(V(1, 1), 4))

