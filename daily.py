import numpy as np
from enum import Enum
import random
import tkinter as tk
import itertools
import collections
import matplotlib.pyplot as plt


class Movement(Enum):
    REMOVE = 0
    PLACE = 1

class Queens_game:
    def __init__(self,N):
        self.board = np.zeros((N,N))
        self.board_len = N
        self.queens_located = list()
        self.solutions = list()

    def queen(self,x, y, movement):
        if movement == Movement.PLACE:
            self.queens_located.append([x,y])
        elif movement == Movement.REMOVE:
            self.queens_located.remove([x,y])
        self.board[x,:] = movement.value
        self.board[x:,y] = movement.value
        x_act = x + 1
        y_act = y + 1
        while (x_act < self.board_len) and (y_act < self.board_len):
            self.board[x_act, y_act] = movement.value
            x_act += 1
            y_act += 1
        x_act = x + 1
        y_act = y - 1
        while x_act < self.board_len and y_act > 0:
            self.board[x_act,y_act] = movement.value
            x_act += 1
            y_act -= 1

    def possible_location(self,x,y):
        if self.board[x,y] == 0:
            return True
        else:
            return False

    def try_to_place(self,i, j):
        if queen_g.possible_location(i,j):
            queen_g.queen(i,j,Movement.PLACE)
            if len(queen_g.queens_located) == queen_g.board_len:
                queen_g.solutions.append(queen_g.queens_located.copy())
            return True
        else:
            return False

    def play_queen(self,i):
        for j in range(0,queen_g.board_len):
            print(self.queens_located)
            if queen_g.try_to_place(i,j):
                if i + 1 < queen_g.board_len:
                    queen_g.play_queen(i + 1)
        queen_g.queen(queen_g.queens_located[-1][0],queen_g.queens_located[-1][1],Movement.REMOVE)  # Remove last queen
    def play(self):
        queen_g.play_queen(0)


def is_not_three_times(array):
    '''
    Given an array of integers where every integer occurs three times except for one integer, which only occurs once,
    find and return the non-duplicated integer.

    For example, given [6, 1, 3, 3, 3, 6, 6], return 1. Given [13, 19, 13, 13], return 19.
    :param array:
    :return:
    '''
    my_dict = {}
    for x in array:
        if x not in my_dict:
            my_dict[x] = 1
        else:
            my_dict[x] += 1
    return list(my_dict.keys())[list(my_dict.values()).index(1)]

def travel(itinerary, origin, destination, list_of_flights):
    '''
    Given an unordered list of flights taken by someone, each represented as (origin, destination) pairs, and a
    starting airport, compute the person's itinerary to a destination

    For example, given the list of [('A','B'),('B','C'),('C','A')] it should return ['A','B','C']
    '''
    possible_destination = None
    for act_origin, act_destination in list_of_flights:
        if origin == act_origin and destination == act_destination:
            itinerary.append(act_destination)
            return itinerary
        elif origin == act_origin:
            possible_destination = act_destination
    if possible_destination is not None:
        itinerary.append(possible_destination)
        travel(itinerary, itinerary[-1], destination, list_of_flights)
    else:
        return []

def count_inversions(array):
    '''
    Count inversions in array. The array [2, 4, 1, 3, 5] has three inversions: (2, 1), (4, 1), and (4, 3).
    '''
    counter = 0
    for i in range(len(array)):
        for j in range(len(array)):
            if array[i] > array[j] and i < j:
                counter += 1
    return counter

def rand5():
    '''
    Returns an uniform distributed random number between 1 to 5 inclusive.
    '''
    return random.randrange(1,6)

def rand7():
    '''
    Returns an uniform distributed random number between 1 to 7 inclusive.
    '''
    return random.randrange(1,8)

def max_profit(stock_prices):
    '''
    Given a array of numbers representing the stock prices of a company in chronological order, write a function that
    calculates the maximum profit you could have made from buying and selling that stock once. You must buy before you
    can sell it.

    For example, given [9, 11, 8, 5, 7, 10], you should return 5, since you could buy the stock at 5 dollars and sell it
     at 10 dollars.
    '''
    profits = list()
    for i in range(len(stock_prices)):
        for j in range(i, len(stock_prices)):
            profit = stock_prices[j] - stock_prices[i]
            if profit > 0:
                profits.append(profit)
    return max(profits)

def contiguous_max(array):
    '''
    Given an array of numbers, find the maximum sum of any contiguous subarray of the array.

    For example, given the array [34, -50, 42, 14, -5, 86], the maximum sum would be 137, since we would take elements
    42, 14, -5, and 86.
    :param array: array of elements to be computed.
    :return:
    '''
    sum = 0
    for i in range(len(array)):
        act_sum = 0
        for j in range(i,len(array)):
            act_sum += array[j]
            print('I add the element {}'.format(array[j]))
            print('Index: {}'.format(j))
            print(act_sum)
            if act_sum > sum:
                sum = act_sum
    return sum


'''
THE TREE OPERATORS
Suppose an arithmetic expression is given as a binary tree. Each leaf is an integer and each internal node is one of '+', '−', '∗', or '/'.

Given the root to such a tree, write a function to evaluate it.
'''
class Node:
    def __init__(self,data):
        self.left = None
        self.right = None
        self.data = data

    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print(self.data),
        if self.right:
            self.right.PrintTree()

    # Inorder traversal
    # Left -> Root -> Right
    def inorderTraversal(self,root):
        res = []
        if root:
            res = self.inorderTraversal(root.left)
            res.append(root.data)
            res += self.inorderTraversal(root.right)
        return res

    # Preorder traversal
    # Root -> Left -> Rigth
    def PreorderTraversal(self,root):
        res = []
        if root:
            res.apppend(root.data)
            res += + self.PreorderTraversal(root.left)
            res += + self.PreorderTraversal(root.right)
        return res


    # Postorder traversal
    # Left -> Right -> Root
    def PostorderTraversal(self,root):
        res = []
        if root:
            res = self.PostorderTraversal(root.left)
            res += self.PostorderTraversal(root.right)
            res.append(root.data)
        return res

    def evaluate(self,root):
        res = self.PostorderTraversal(root)
        solution = []
        for element in res:
            if isinstance(element, (int, float)):
                solution.append(element)
            if element == '+':
                sol = solution[-1] + solution[-2]
                solution.pop()
                solution.pop()
                solution.append(sol)
            if element == '-':
                sol = solution[-1] - solution[-2]
                solution.pop()
                solution.pop()
                solution.append(sol)
            if element == '*':
                sol = solution[-1] * solution[-2]
                solution.pop()
                solution.pop()
                solution.append(sol)
            if element == '/':
                sol = solution[-1] / solution[-2]
                solution.pop()
                solution.pop()
                solution.append(sol)
        return solution[0]

def breaker(text, n_lines):
    words = text.split()
    print(words)
    text_n_lines = list()
    counter = 0
    for word in words:
        if counter == n_lines:
            counter = 0
        if counter == 0:
            text_n_lines.append(word)
        else:
            text_n_lines[-1] += " " + word
        counter += 1
    return text_n_lines

def get_index_of_element(array,element):
    for i in range(len(array)):
        if array[i] == element:
            return i
    return None

def add_splitter(array,suma):
    if sum(array) <= suma:
        return None
    permutations_1 = list(itertools.permutations(array))
    add_to_index = list()
    for permutation in permutations_1:
        act_sum = 0
        for i in range(len(permutation)):
            act_sum += permutation[i]
            if act_sum == suma:
                add_to_index.append(i)
                break
            elif act_sum > suma:
                add_to_index.append(None)
                break
    for i in range(len(add_to_index)):
        if add_to_index[i] is not None:
            if sum(permutations_1[i][add_to_index[i]+1:]) == suma:
                return permutations_1[i][:add_to_index[i]+1], permutations_1[i][add_to_index[i]+1:]
    return

class Path:
    def __init__(self,M,N):
        self.map = np.zeros((M,N))
        self.M = M
        self.N = N
        self.position = [0,0]
        self.number_of_solutions = 0
        self.act_positions = list()
    def _move_right(self):
        if self.position[1] + 1 <= self.N:
            self.position[1] += 1
            return True
        else: return False
    def _move_down(self):
        if self.position[0] + 1 <= self.M:
            self.position[0] += 1
            return True
        else: return False
    def expand(self):
        pass

def find_word(soup, word):
    letters = list(word)
    found = False
    for i in range(len(soup)):
        if letters[i] == soup[i][i]:
            for j in range(len(soup)):
                if letters[j] != soup[i][j]:
                    break
                elif j == len(soup) - 1:
                    found = True
            if found:
                return True
            for j in range(len(soup)):
                if letters[j] != soup[j][i]:
                    break
                elif j == len(soup) - 1:
                    found = True
            if found:
                return True
    return False

def is_prime(n) :
    for i in range (2,n):
        if n % i == 0:
            return False
    if n == 0 or n == 1:
        return False
    return True

def prime_counter(n):
    '''
    count the number of prime numbers in range [0,n)
    :param n:
    :return:
    '''
    counter = 0
    for i in range(n):
        if is_prime(i):
            counter += 1
    return counter

def print_spiral(matrix):
    rotx1 = 0
    roty1 = 0
    rotx2 = M
    roty2 = N
    while (rotx1 < rotx2 and roty1 < roty2):
        for i in range(roty1,roty2):
            print(matrix[rotx1][i])
        rotx1 += 1
        if (rotx1 >= rotx2):
            break
        for i in range(rotx1,rotx2):
            print(matrix[i][roty2-1])
        roty2 -= 1
        if (roty1 >= roty2):
            break
        for i in range(roty2 - 1,roty1 - 1,-1):
            print(matrix[rotx2-1][i])
        rotx2 -= 1
        if (rotx1 >= rotx2):
            break
        for i in range(rotx2 - 1,rotx1 - 1,-1):
            print(matrix[i][roty1])
        roty1 += 1

# Problem 68
class BishopGame:
    '''
    On our special chessboard, two bishops attack each other if they share the same diagonal. This includes bishops
    that have another bishop located between them, i.e. bishops can attack through pieces.

    You are given N bishops, represented as (row, column) tuples on a M by M chessboard. Write a function to count the
    number of pairs of bishops that attack each other. The ordering of the pair doesn't matter: (1, 2) is considered
    the same as (2, 1).
    '''
    def __init__(self,M,bishop_list):
        self._M = M
        self._bishop_list = bishop_list
        self.solutions = []
        self._menaces = 0
        self._bishop_attacks()

    def _bishop_attacks(self):
        for bishop in self._bishop_list:
            act_pos = list(bishop)
            for _ in range(bishop[0], self._M):
                act_pos = [x + 1 for x in act_pos]
                if act_pos[0] > M or act_pos[1] > M:
                    break
                if tuple(act_pos) in bishop_list:
                    self.solutions.append([bishop,tuple(act_pos)])
                    self._menaces += 1
            act_pos = list(bishop)
            for _ in range(bishop[0], 0, -1):
                act_pos[0] -= 1
                act_pos[1] += 1
                if act_pos[0] < 0 or act_pos[1] > M:
                    break
                if tuple(act_pos) in bishop_list:
                    self.solutions.append([bishop,tuple(act_pos)])
                    self._menaces += 1

    def get_menaces(self):
        return self._menaces

# Problem 69
def maximum_product(elements):
    '''
    Given a list of integers, return the largest product that can be made by multiplying any three integers.
    '''
    combis = list(itertools.combinations(elements,3))
    print(combis)
    max_product = 0
    for combi in combis:
        product = 1
        for element in combi:
            product *= element
        if product > max_product:
            max_product = product
    return max_product

# Problem 70
def perfect_number(integer):
    string_number = str(integer)
    suma = 0
    for digit in string_number:
        suma += int(digit)
    if suma > 10:
        print('cannot be converted into a perfect number')
        return
    elif suma == 10:
        print('already a perfect number')
        return
    else:
        string_number += str(10-suma)
        return(int(string_number))

#Problem 71
def random5():
    '''
    Given a random [1,7] get a uniform random [1,5]
    '''
    x = np.random.randint(1,8,1)
    while x > 5:
        x = np.random.randint(1,8,1)
    return int(x)

#Problem 73
def reverse(list):
    return list.reverse()

#Problem 74
def times_in_mult_table(N, n):
    times = 0
    for i in range(1,N + 1):
        for j in range(1, N + 1):
            if i * j == n:
                times += 1
    return times

#Problem 76
'''
You are given an N by M 2D matrix of lowercase letters. Determine the minimum number of columns that can be removed to 
ensure that each row is ordered from top to bottom lexicographically. That is, the letter at each column is 
lexicographically later as you go down each row. It does not matter whether each row itself is ordered lexicographically.
You are given an N by M 2D matrix of lowercase letters. Determine the minimum number of columns that can be removed to
ensure that each row is ordered from top to bottom lexicographically. That is, the letter at each column is 
lexicographically later as you go down each row. It does not matter whether each row itself is ordered 
lexicographically.
'''
def is_ordered(lista):
    for i in range(len(lista)):
        for j in range(i,len(lista)):
            if lista[i] > lista[j]:
                return False
    return True

def solution_76(char_matrix):
    columns_to_remove = 0
    columns = list(zip(*char_matrix))
    for column in columns:
        if not is_ordered(column):
            columns_to_remove += 1
    return columns_to_remove

#Problem 77
'''
Get rid of those annoying overlapping intervals.
'''
def solution_77(list_of_intervals):
    for interval_1 in list_of_intervals:
        for interval_2 in list_of_intervals:
            if interval_1[0] < interval_2[1] and interval_1[0] > interval_2[0]:
                if interval_1[0] > interval_2[0]:
                    interval_1[0] = interval_2[0]
                if interval_1[1] < interval_2[1]:
                    interval_1[1] = interval_2[1]
                list_of_intervals.remove(interval_2)
    return list_of_intervals

'''
Given k sorted singly linked lists, write a function to merge all the lists into one sorted singly linked list.
'''
def solution_78(list_of_lists):
    list_merged = []
    for lista in list_of_lists:
        for element in lista:
            list_merged.append(element)
    return list_merged

'''
Given an array of integers, write a function to determine whether the array could become non-decreasing by modifying at 
most 1 element.
'''
def solution_79(array_of_integers):
    for i in range(len(array_of_integers)):
        aux = array_of_integers[i]
        array_of_integers[i] = 1
        print(array_of_integers)
        if sorted(array_of_integers) == array_of_integers:
            return True
        else:
            array_of_integers[i] = aux
    return False

'''# Problem 80
Given the root of a binary tree, return a deepest node. 
'''


class Node80(Node):
    def __init__(self, data):
        super().__init__(data)
        self.max_deep = -1
        self.act_deep = -1
        self.deepest_node = Node(data)

    def find_deepest_node(self,root):
        if root:
            self.act_deep += 1
            if self.act_deep > self.max_deep:
                self.max_deep = self.act_deep
                self.deepest_node = root
            self.find_deepest_node(root.left)
            self.find_deepest_node(root.right)
        else:
            self.act_deep -= 1
            return

#Problem 81
'''
Given a mapping of digits to letters (as in a phone number), and a digit string, return all possible letters the number 
could represent. You can assume each valid number in the mapping is a single digit.
'''
def solution_81(value):

    key_letters = []
    longitud = 3**len(str(value))
    possibilities = [''] * longitud
    numbers_to_letters = {1:['a','b','c'], 2:['d','e','f'], 3:['g','h','i'], 4:['j','k','l'], 5:['m','n','ñ'],
                          6:['o','p','q'], 7:['r','s','t'], 8:['u','v','w'], 9:['x','y','z']}
    value_list = [int(x) for x in str(value)]
    for value in value_list:
        key_letters.append(numbers_to_letters[value])
    print(key_letters)

    i = 0
    j = 0

    for key in key_letters[0]:
        for i in range(j,int(j + longitud/3)):
            possibilities[i] += key
        j += 3
    j = 0
    while j < longitud:
        for i, key in enumerate(key_letters[1]):
            possibilities[i+j] += key
        j += 3

    print(possibilities)

'''
Using a read7() method that returns 7 characters from a file, implement readN(n) which reads n characters.

For example, given a file with the content “Hello world”, three read7() returns “Hello w”, “orld” and then “”.
'''
file_pointer = 0
def read_7(filename):
    global file_pointer
    with open(filename) as f:
        f.seek(file_pointer)
        read_chars = f.read(7)
    file_pointer += 7
    return read_chars


'''
Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A 1 represents land and 0 represents water, 
so an island is a group of 1s that are neighboring whose perimeter is surrounded by water.
Map example
map = [[0, 0, 1],
           [1, 0, 1],
           [0, 1, 0],
           [0, 0, 0]]
'''
class IslandSearcher():
    def __init__(self, map):
        self._map = map
        self._rows = len(map)
        self._cols = len(map[0])
        self._visited_locations = []

    def count_islands(self):
        island_counter = 0
        for i in range(self._rows):
            for j in range(self._cols):
                if self.is_land(i,j):
                    if (i,j) not in self._visited_locations:
                        print('New search [{} {}]'.format(i,j))
                        self.is_island(i,j)
                        island_counter += 1
        return island_counter

    def is_island(self,i,j):
        self._visited_locations.append((i,j))
        print('Looking [{} {}]'.format(i,j))
        if i - 1 >= 0:  # exist
            print('Up exists')
            if self.is_land(i - 1,j):
                print('Up is land')
                if (i - 1,j) not in self._visited_locations:  # not visited
                    print('Up is not visited')
                    self.is_island(i - 1,j)
        if j + 1 < self._cols:
            print('Right exists')
            if self.is_land(i,j + 1):
                print('Right is land')
                if (i,j + 1) not in self._visited_locations:
                    print('Right is not visited')
                    self.is_island(i,j + 1)
        if i + 1 < self._rows:
            print('Down exists')
            if self.is_land(i + 1,j):
                print('Down is land')
                if (i + 1,j) not in self._visited_locations:
                    print('Down is not visited')
                    self.is_island(i + 1,j)
        if j - 1 >= 0:
            print('Left exists')
            if self.is_land(i, j - 1):
                print('Left is land')
                if (i, j - 1) not in self._visited_locations:
                    print('Left is not visited')
                    self.is_island(i, j-1)
        return


    def is_land(self,i,j):
        if self._map[i][j] == 1:
            return True
        else: return False

'''
Given three 32-bit integers x, y, and b, return x if b is 1 and y if b is 0, using only mathematical or bit operations. 
You can assume b can only be 1 or 0.
'''
def solution_85(x, y, b):
    for i in range(31):
        solution += x & b + y & b
if __name__ == '__main__':
    b = 3
    print(~b)