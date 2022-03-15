# Patryk Bandyra

import numpy as np
import random
import csv

"""
    It's better to restart program if console is cluttered with 'Failed to create passable map' infos 
    in order to get better seed (sometimes it may take a lot of time to create a map but usually it is instant if n and 
    holes parameters are in good ratio).

    I assume that the map in the file is passable and has only 0, 1, 2, 3 values as fields.
    0 - passable
    1 - start pos
    2 - end pos
    3 - hole
    There is a check if map has NxN dimensions.

    Q-Learning Algorithm implemented using epsilon greedy algorithm.
"""


class MapBuilder:
    """
    Class to create and validate map or to load one from a csv file
    """

    @staticmethod
    def create_random_map(start_pos, end_pos, holes, n=8):
        """
        :param start_pos: tuple (x, y) where x < n and y < n
        :param end_pos: tuple (x, y) where x < n and y < n
        :param holes: number of holes on the map
        :param n: number of rows and columns in map matrix
        :return: map - NxN matrix
        """
        while True:
            map = np.zeros((n, n), int)
            map[start_pos[1]][start_pos[0]] = 1  # 1 will symbolize start pos
            map[end_pos[1]][end_pos[0]] = 2  # 2 will symbolize end pos

            # get positions of holes
            i = 0
            while i < holes:
                temp_pos = (random.randint(0, n - 1), random.randint(0, n - 1))
                if map[temp_pos[1]][temp_pos[0]] == 0:
                    map[temp_pos[1]][temp_pos[0]] = 3  # 3 will symbolize a hole
                    i += 1
            if MapBuilder._validate_map(map, n, start_pos):
                break
            print('Failed to create passable map')
        print('Map created')
        return map

    @staticmethod
    def load_map_from_csv_file(filename):
        """
        :param filename: string
        :return: map - NxN matrix
        """
        # first validate the map file - we want NxN map format
        with open(filename) as file:
            reader = csv.reader(file, delimiter=',')
            row_count = sum(1 for row in reader)
            for row in reader:
                element_count = 0
                for _ in row:
                    element_count += 1
                if element_count != row_count:
                    print('Wrong map format!')
                    exit(1)
        # map correct - load from file into matrix
        return np.genfromtxt(filename, delimiter=',', dtype=int)

    @staticmethod
    def get_start_pos_from_map(map):
        for y in range(len(map)):
            for x in range(len(map[0])):
                if map[y][x] == 1:
                    return x, y

    @staticmethod
    def get_end_pos_from_map(map):
        for y in range(len(map)):
            for x in range(len(map[0])):
                if map[y][x] == 2:
                    return x, y

    @staticmethod
    def get_n_from_map(map):
        return len(map)

    @staticmethod
    def _validate_map(map, n, start_pos):
        """
        :param map: NxN matrix
        :return: true if path from start to finish exists
        """
        # create graph from matrix
        graph = Graph()
        for y in range(len(map)):
            for x in range(len(map[0])):
                if map[y][x] == 0 or map[y][x] == 1:
                    graph.add_vertex(Vertex((y, x)))
                elif map[y][x] == 2:
                    graph.add_vertex(Vertex((y, x), 'gold'))

                if map[y][x] != 3:
                    # add edges
                    # check field on left
                    if 0 <= x - 1 < n and map[y][x - 1] != 3:  # not a hole
                        graph.add_edge(f'{x}{y}', f'{x - 1}{y}')
                    # check field on right
                    if 0 <= x + 1 < n and map[y][x + 1] != 3:
                        graph.add_edge(f'{x}{y}', f'{x + 1}{y}')
                    # check field above
                    if 0 <= y - 1 < n and map[y - 1][x] != 3:
                        graph.add_edge(f'{x}{y}', f'{x}{y - 1}')
                    # check field below
                    if 0 <= y + 1 < n and map[y + 1][x] != 3:
                        graph.add_edge(f'{x}{y}', f'{x}{y + 1}')
        # for k, v in graph.vertices.items():
        #     print(f'Vertex: {k} - neighbours: {v.neighbours}')
        return graph.dfs(graph.vertices[f'{start_pos[1]}{start_pos[0]}'])


class Vertex:
    """
    DSF algorithm class
    """

    def __init__(self, pos, color='black'):
        self.name = f'{pos[1]}{pos[0]}'  # example: given pos (1, 5) -> self.name = 15
        self.neighbours = list()

        self.discovery = 0
        self.finish = 0
        self.color = color

    def add_neighbour(self, v):
        temp_set = set(self.neighbours)
        if v not in temp_set:
            self.neighbours.append(v)
            self.neighbours.sort()


class Graph:
    """
    DSF algorithm class
    """
    vertices = {}
    time = 0
    flag = False

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex) and vertex.name not in self.vertices:
            self.vertices[vertex.name] = vertex
            return True
        return False

    def add_edge(self, u, v):
        if u in self.vertices and v in self.vertices:
            for key, value in self.vertices.items():
                if key == u:
                    value.add_neighbour(v)
                if key == v:
                    value.add_neighbour(u)
            return True
        return False

    def _dfs(self, vertex):
        global time
        vertex.color = 'red'
        vertex.discovery = time
        time += 1
        for v in vertex.neighbours:
            if self.vertices[v].color == 'black':
                self._dfs(self.vertices[v])
            elif self.vertices[v].color == 'gold':
                self.flag = True
                self._dfs(self.vertices[v])
        vertex.color = 'blue'
        vertex.finish = time
        time += 1

    def dfs(self, vertex):
        global time
        time = 1
        self._dfs(vertex)
        return self.flag


class Pinky:
    def __init__(self, pos_x, pos_y, n):
        self.start_x = pos_x
        self.start_y = pos_y
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.n = n

    def reset_pos(self):
        self.pos_x = self.start_x
        self.pos_y = self.start_y

    def make_move(self):
        """
        Makes move in random direction
        """
        while True:
            dir_choice = random.choice(['x', 'y'])
            amount_choice = random.choice([-1, 1])
            if dir_choice == 'x' and 0 <= self.pos_x + amount_choice < self.n:
                self.pos_x += amount_choice
                break
            elif dir_choice == 'y' and 0 <= self.pos_y + amount_choice < self.n:
                self.pos_y += amount_choice
                break


def test_pinky_behaviour(map, episodes):
    """
    Tests how Pinky behaves in a given environment making random decisions
    :param map:
    """
    wins = 0
    start_x, start_y = MapBuilder.get_start_pos_from_map(map)
    n = MapBuilder.get_n_from_map(map)
    p = Pinky(start_x, start_y, n)
    for i in range(episodes):
        p.reset_pos()
        while True:
            p.make_move()
            # print(f'X:{p.pos_x}, Y:{p.pos_y}')
            if map[p.pos_y][p.pos_x] == 3:
                # print('Failure')
                break
            elif map[p.pos_y][p.pos_x] == 2:
                print(f'Win! Pinky got to X:{p.pos_x}, Y:{p.pos_y} at episode {i}')
                wins += 1
                break
    print(f'Win-Failure Ratio: {wins/episodes} wins')


def main1():
    EPISODES = 10000
    test_pinky_behaviour(MapBuilder.load_map_from_csv_file('map1.csv'), EPISODES)


def test_brain_behaviour(map, episodes, epsilon, discount_factor, learning_rate):
    n = MapBuilder.get_n_from_map(map)
    start_x, start_y = MapBuilder.get_start_pos_from_map(map)
    end_x, end_y = MapBuilder.get_end_pos_from_map(map)

    # Creates a 3D numpy array to hold the current Q-values for each state and action pair: Q(s, a)
    # The value of each (state, action) pair is initialized to 0
    q_values = np.zeros((n, n, 4))

    # Actions for Manhattan movement - numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
    actions = ['up', 'right', 'down', 'left']

    # Create 2D array to hold rewards for each state
    rewards = np.full((n, n), -100.)    # -100 -> holes

    # set rewards for all non-terminal states
    for y in range(n):
        for x in range(n):
            if map[y, x] != 3:
                rewards[y, x] = -1.     # -1 -> non-terminal states

    rewards[end_y, end_x] = 100.  # 100 -> our target
    print(map)
    print(rewards)

    # Helper Functions for Q-Learning Algorithm
    def is_terminal_state(y, x):
        if rewards[y, x] == -1.:
            return False
        else:
            return True

    # Chooses a non-terminal starting location
    def get_starting_location():
        # get a random row and column index
        y = np.random.randint(n)
        x = np.random.randint(n)
        # continue choosing random row and column indexes until a non-terminal state is identified
        while is_terminal_state(y, x):
            y = np.random.randint(n)
            x = np.random.randint(n)
        return y, x

    # Epsilon greedy algorithm that will choose which action to take next
    def get_next_action(y, x, epsilon):
        if np.random.random() < epsilon:
            return np.argmax(q_values[y, x])
        else:  # choose a random action
            return np.random.randint(4)

    # Gets the next location based on the chosen action
    def get_next_location(y, x, action_index):
        new_y = y
        new_x = x
        if actions[action_index] == 'up' and y > 0:
            new_y -= 1
        elif actions[action_index] == 'right' and x < n - 1:
            new_x += 1
        elif actions[action_index] == 'down' and y < n - 1:
            new_y += 1
        elif actions[action_index] == 'left' and x > 0:
            new_x -= 1
        return new_y, new_x

    # Gets the shortest path between any location on the map
    def get_shortest_path(start_y, start_x):
        if is_terminal_state(start_y, start_x):
            return []
        else:  # if this is a 'legal' starting location
            current_y, current_x = start_y, start_x
            shortest_path = []
            shortest_path.append([current_y, current_x])
            # continue moving along the path until we reach the goal
            while not is_terminal_state(current_y, current_x):
                # get the best action to take
                action_index = get_next_action(current_y, current_x, 1.)
                # move to the next location on the path, and add the new location to the list
                current_y, current_x = get_next_location(current_y, current_x, action_index)
                shortest_path.append([current_y, current_x])
            return shortest_path

    # run through training episodes
    for episode in range(episodes):
        # get the starting location for this episode
        """
        Here leave the first line uncommented to throw Brain onto a random non-terminal map field in the beginning of each episode.
        Leave the second line uncommented to throw Brain onto the same non-terminal map field defined in map in the beginning each episode.
        """
        row_index, column_index = get_starting_location()
        # row_index, column_index = start_y, start_x

        # continue taking actions until we reach a terminal state
        while not is_terminal_state(row_index, column_index):
            # choose which action to take
            action_index = get_next_action(row_index, column_index, epsilon)

            # perform the chosen action, and transition to the next state
            old_row_index, old_column_index = row_index, column_index  # store the old row and column indexes
            row_index, column_index = get_next_location(row_index, column_index, action_index)

            # receive the reward for moving to the new state, and calculate the temporal difference
            reward = rewards[row_index, column_index]
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

            # update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value

    print('Training complete!')
    print(f'Shortest path from X:{start_x}, Y:{start_y} to X:{end_x}, Y:{end_y}')
    print(get_shortest_path(start_y, start_x))
    print('Q-Values:')
    print(q_values)


def main2():
    # const values
    EPISODES = 1000
    EPSILON = 0.9    # the percentage of time when we should take the best action
    DISCOUNT_FACTOR = 0.9   # discount factor for future rewards
    LEARNING_RATE = 0.9     # the rate at which the AI agent should learn
    test_brain_behaviour(MapBuilder.load_map_from_csv_file('map1.csv'), EPISODES, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE)


def main3():
    EPISODES = 10000
    EPSILON = 0.9
    DISCOUNT_FACTOR = 0.9
    LEARNING_RATE = 0.9
    start_pos = [0, 0]
    end_pos = [6, 3]
    holes = 20
    map = MapBuilder.create_random_map(start_pos, end_pos, holes)
    test_pinky_behaviour(map, EPISODES)
    test_brain_behaviour(map, EPISODES, EPSILON, DISCOUNT_FACTOR, LEARNING_RATE)


"""
    main1() - Pinky behaviour on the map from a file; if Pinky got to the goal state - display location and episode;
              after all episodes - show win-failure ratio
    main2() - Brain behaviour on the map from a file; display map, rewards matrix, shortest path from a starting location
              to the goal state (the path that was learnt by Brain) and q-values 3D array
    main3() - Pinky and Brain behaviour on randomly generated passable map
    
    
"""

if __name__ == '__main__':
    # main1()
    main2()
    # main3()