# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import copy
import functools
import heapq
import math
import re
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations, product

import networkx as nx
import numpy as np

# %% [markdown] heading_collapsed=true
# # Day 1
#

# %% hidden=true
puzzle_input = open("../data/aoc/01.txt", mode="r").read()

puzzle_example1 = """1abc2
pqr3stu8vwx
a1b2c3d4e5f
treb7uchet
"""

puzzle_example2 = """two1nine
eightwothree
abcone2threexyz
xtwone3four
4nineeightseven2
zoneight234
7pqrstsixteen
"""

# %% hidden=true
total = 0
for line in puzzle_input.splitlines():
    numbers = re.sub("[a-z]", "", line)
    total += int(numbers[0]) * 10 + int(numbers[-1])

total

# %% hidden=true
numbers = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

for i, number in enumerate(numbers):
    puzzle_input = puzzle_input.replace(number, f"{number[0]}{i}{number[1:]}")

total = 0
for line in puzzle_input.splitlines():
    numbers = re.sub("[a-z]", "", line)
    total += int(numbers[0]) * 10 + int(numbers[-1])

total

# %% [markdown] heading_collapsed=true
# # Day 2
#

# %% hidden=true
puzzle_input = open("../data/aoc/02.txt", mode="r").read()

puzzle_example = """Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green"""

# %% hidden=true
limit = {"red": 12, "green": 13, "blue": 14}

impossible_games = set()

for line in puzzle_input.splitlines():
    game_number, draws = line.split(": ")
    game_number = int(game_number[5:])
    for draw in draws.split("; "):
        for sample in draw.split(", "):
            number, colour = sample.split()
            if int(number) > limit[colour]:
                impossible_games.add(game_number)

sum(set(range(1, 101)) - impossible_games)

# %% hidden=true
total = 0

for line in puzzle_input.splitlines():
    _, draws = line.split(": ")
    dd = defaultdict(list)
    for draw in draws.split("; "):
        for sample in draw.split(", "):
            number, colour = sample.split()
            dd[colour].append(int(number))

    total += max(dd["red"]) * max(dd["green"]) * max(dd["blue"])

total

# %% [markdown] heading_collapsed=true
# # Day 3
#

# %% hidden=true
puzzle_input = open("../data/aoc/03.txt", mode="r").read()

puzzle_example = """467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598.."""

# %% hidden=true
digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
lines = puzzle_input.splitlines()
numbers = []


def extract(k, l):
    start = l
    end = l
    while start > 0 and lines[k][start - 1] in digits:
        start -= 1

    while end < len(lines[k]) and lines[k][end] in digits:
        end += 1

    number = int(lines[k][start:end])
    lines[k] = lines[k][:start] + ("." * (end - start)) + lines[k][end:]
    return number


for i, line in enumerate(lines):
    for j, c in enumerate(line):
        if c not in ["."] + digits:
            for k in range(i - 1, i + 2):
                for l in range(j - 1, j + 2):
                    if lines[k][l] in digits:
                        number = extract(k, l)
                        numbers.append(number)
sum(numbers)

# %% hidden=true
digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
lines = puzzle_input.splitlines()
gear_ratios = []


def extract(k, l):
    start = l
    end = l
    while start > 0 and lines[k][start - 1] in digits:
        start -= 1

    while end < len(lines[k]) and lines[k][end] in digits:
        end += 1

    number = int(lines[k][start:end])
    lines[k] = lines[k][:start] + ("." * (end - start)) + lines[k][end:]
    return number


for i, line in enumerate(lines):
    for j, c in enumerate(line):
        if c not in ["."] + digits:
            line1, line2, line3 = lines[i - 1], lines[i], lines[i + 1]
            numbers = []
            for k in range(i - 1, i + 2):
                for l in range(j - 1, j + 2):
                    if lines[k][l] in digits:
                        number = extract(k, l)
                        numbers.append(number)
            if len(numbers) == 2:
                gear_ratios.append(numbers[0] * numbers[1])
            else:  # restore lines
                lines[i - 1] = line1
                lines[i] = line2
                lines[i + 1] = line3

sum(gear_ratios)

# %% [markdown] heading_collapsed=true
# # Day 4
#

# %% hidden=true
puzzle_input = open("../data/aoc/04.txt", mode="r").read()

puzzle_example = """Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11"""


# %% hidden=true
def convert(correct):
    if correct == 0:
        return 0
    elif correct == 1:
        return 1
    else:
        return 2 ** (correct - 1)


# %% hidden=true
lines = puzzle_input.splitlines()
total = 0
for line in lines:
    a, b = line.split(": ")
    winning, chosen = b.split(" | ")
    winning_numbers = set(winning.split())
    chosen_numbers = set(chosen.split())
    total += convert(len(winning_numbers.intersection(chosen_numbers)))

total

# %% hidden=true
lines = reversed(puzzle_input.splitlines())
cards = dict()
for line in lines:
    a, b = line.split(": ")
    card_number = int(a[5:])
    winning, chosen = b.split(" | ")
    winning_numbers = set(winning.split())
    chosen_numbers = set(chosen.split())
    correct = len(winning_numbers.intersection(chosen_numbers))
    cards[card_number] = correct
    for i in range(card_number + 1, card_number + 1 + correct):
        cards[card_number] += cards[i]

sum(cards.values()) + len(cards)

# %% [markdown] heading_collapsed=true
# # Day 5
#

# %% hidden=true
puzzle_input = open("../data/aoc/05.txt", mode="r").read()

puzzle_example = """seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4"""

# %% hidden=true
lines = puzzle_input.splitlines()
seeds = list(map(int, lines[0].split()[1:]))

dd = defaultdict(list)
map_name = None
for line in lines[2:]:
    tokens = line.split()
    if len(tokens) == 2:
        map_name = tokens[0]
    elif len(tokens) == 3:
        dd[map_name].append(tuple(map(int, tokens)))


def process(x, d):
    for destination_range_start, source_range_start, range_length in d:
        if source_range_start <= x < source_range_start + range_length:
            return x - source_range_start + destination_range_start
    return x


locations = []
for seed in list(seeds):
    x = seed
    x = process(x, dd["seed-to-soil"])
    x = process(x, dd["soil-to-fertilizer"])
    x = process(x, dd["fertilizer-to-water"])
    x = process(x, dd["water-to-light"])
    x = process(x, dd["light-to-temperature"])
    x = process(x, dd["temperature-to-humidity"])
    x = process(x, dd["humidity-to-location"])
    locations.append(x)

min(locations)

# %% hidden=true
lines = puzzle_input.splitlines()
seeds = list(map(int, lines[0].split()[1:]))

dd = defaultdict(list)
map_name = None
for line in lines[2:]:
    tokens = line.split()
    if len(tokens) == 2:
        map_name = tokens[0]
    elif len(tokens) == 3:
        dd[map_name].append(tuple(map(int, tokens)))


def process(x, d):
    output = []
    for x_start, x_end in x:
        for destination_start, source_start, range_ in d:
            source_end = source_start + range_
            destination_end = destination_start + range_
            if x_start < source_end and source_start < x_end:  # overlap
                output.append(
                    (
                        max(x_start, source_start) + destination_start - source_start,
                        min(x_end, source_end) + destination_start - source_start,
                    )
                )
    return sorted(output)


x = [(start, start + r) for start, r in sorted(list(zip(seeds[::2], seeds[1::2])))]
x = process(x, dd["seed-to-soil"])
x = process(x, dd["soil-to-fertilizer"])
x = process(x, dd["fertilizer-to-water"])
x = process(x, dd["water-to-light"])
x = process(x, dd["light-to-temperature"])
x = process(x, dd["temperature-to-humidity"])
x = process(x, dd["humidity-to-location"])
x

# %% [markdown] heading_collapsed=true
# # Day 6
#

# %% hidden=true
puzzle_input = open("../data/aoc/06.txt", mode="r").read()

puzzle_example = """Time:      7  15   30
Distance:  9  40  200"""

# %% hidden=true
lines = puzzle_input.splitlines()

times = list(map(int, lines[0].split()[1:]))
record_distances = list(map(int, lines[1].split()[1:]))

combinations_ = []
for time, record_distance in zip(times, record_distances):
    print(time, record_distance)

    a = [t * (time - t) for t in range(time + 1)]
    a = [q for q in a if q > record_distance]
    combinations_.append(len(a))

functools.reduce(lambda x, y: x * y, combinations_)

# %% hidden=true
lines = puzzle_input.splitlines()

time = int("".join(lines[0].split()[1:]))
distance = int("".join(lines[1].split()[1:]))

result = 0

for t in range(time + 1):
    if t * (time - t) >= distance:
        result = t
        break

print(time - result * 2 + 1)

# %% [markdown] heading_collapsed=true
# # Day 7
#

# %% hidden=true
puzzle_input = open("../data/aoc/07.txt", mode="r").read()

puzzle_example = """32T3K 765
T55J5 684
KK677 28
KTJJT 220
QQQJA 483"""

# %% hidden=true
lines = puzzle_input.splitlines()

hands = []

d = {"T": "B", "J": "C", "Q": "D", "K": "E", "A": "F"}

for line in lines:
    hand, bid = line.split()

    for key in d.keys():
        hand = hand.replace(key, d[key])

    c = Counter(hand)
    result = c.most_common(2)
    if result[0][1] == 5:
        type_ = 7
    elif result[0][1] == 4:
        type_ = 6
    elif result[0][1] == 3 and result[1][1] == 2:
        type_ = 5
    elif result[0][1] == 3:
        type_ = 4
    elif result[0][1] == 2 and result[1][1] == 2:
        type_ = 3
    elif result[0][1] == 2:
        type_ = 2
    else:
        type_ = 1

    hands.append((f"{type_}{hand}", int(bid)))

hands.sort(key=lambda item: item[0])

score = 0
for i, (hand, bid) in enumerate(hands):
    score += (i + 1) * bid

score

# %% hidden=true
lines = puzzle_input.splitlines()

hands = []

d = {"T": "B", "J": "1", "Q": "D", "K": "E", "A": "F"}

for line in lines:
    hand, bid = line.split()

    c = Counter(hand)
    result = c.most_common(2)
    most_common_card = result[0][0]

    if most_common_card == "J" and len(result) > 1:
        most_common_card = result[1][0]

    hand = hand.replace("J", most_common_card)

    c = Counter(hand)
    result = c.most_common(2)

    if result[0][1] == 5:
        type_ = 7
    elif result[0][1] == 4:
        type_ = 6
    elif result[0][1] == 3 and result[1][1] == 2:
        type_ = 5
    elif result[0][1] == 3:
        type_ = 4
    elif result[0][1] == 2 and result[1][1] == 2:
        type_ = 3
    elif result[0][1] == 2:
        type_ = 2
    else:
        type_ = 1

    hand, bid = line.split()
    for key in d.keys():
        hand = hand.replace(key, d[key])

    hands.append((f"{type_}{hand}", int(bid)))

hands.sort(key=lambda item: item[0])

score = 0
for i, (hand, bid) in enumerate(hands):
    score += (i + 1) * bid

score

# %% [markdown] heading_collapsed=true
# # Day 8
#

# %% hidden=true
puzzle_input = open("../data/aoc/08.txt", mode="r").read()

puzzle_example1 = """RL

AAA = (BBB, CCC)
BBB = (DDD, EEE)
CCC = (ZZZ, GGG)
DDD = (DDD, DDD)
EEE = (EEE, EEE)
GGG = (GGG, GGG)
ZZZ = (ZZZ, ZZZ)"""

puzzle_example2 = """LLR

AAA = (BBB, BBB)
BBB = (AAA, ZZZ)
ZZZ = (ZZZ, ZZZ)"""

puzzle_example3 = """LR

11A = (11B, XXX)
11B = (XXX, 11Z)
11Z = (11B, XXX)
22A = (22B, XXX)
22B = (22C, 22C)
22C = (22Z, 22Z)
22Z = (22B, 22B)
XXX = (XXX, XXX)"""

# %% hidden=true
lines = puzzle_input.splitlines()
instructions = lines[0]

m = {}
for line in lines[2:]:
    m[line[:3]] = [line[7:10], line[12:15]]

state = "AAA"
steps = 0
while True:
    for instruction in instructions:
        steps += 1
        state = m[state][0 if instruction == "L" else 1]
        if state == "ZZZ":
            break
    if state == "ZZZ":
        break

steps

# %% hidden=true
lines = puzzle_input.splitlines()
instructions = lines[0]

m = {}
for line in lines[2:]:
    m[line[:3]] = [line[7:10], line[12:15]]

states = [key for key in m.keys() if key.endswith("A")]
steps_list = []
for state in states:
    steps = 0
    while True:
        for instruction in instructions:
            steps += 1
            state = m[state][0 if instruction == "L" else 1]
            if state.endswith("Z"):
                break
        if state.endswith("Z"):
            break

    steps_list.append(steps)

math.lcm(*steps_list)

# %% [markdown] heading_collapsed=true
# # Day 9
#

# %% hidden=true
puzzle_input = open("../data/aoc/09.txt", mode="r").read()

puzzle_example = """0 3 6 9 12 15
1 3 6 10 15 21
10 13 16 21 30 45"""

# %% hidden=true
lines = puzzle_input.splitlines()

values = []
for line in lines:
    arr = list(map(int, line.split()))

    levels = None
    for i in range(len(arr)):
        if sum([item != 0 for item in arr[i:]]) == 0:
            levels = i
            break

        for j in reversed(range(i + 1, len(arr))):
            arr[j] -= arr[j - 1]

    arr += [0]

    for i in reversed(range(levels)):
        for j in range(i + 1, len(arr)):
            arr[j] += arr[j - 1]

    values.append(arr[-1])

sum(values)

# %% hidden=true
lines = puzzle_input.splitlines()

values = []
for line in lines:
    arr = list(reversed(list(map(int, line.split()))))

    levels = None
    for i in range(len(arr)):
        if sum([item != 0 for item in arr[i:]]) == 0:
            levels = i
            break

        for j in reversed(range(i + 1, len(arr))):
            arr[j] -= arr[j - 1]

    arr += [0]

    for i in reversed(range(levels)):
        for j in range(i + 1, len(arr)):
            arr[j] += arr[j - 1]

    values.append(arr[-1])

sum(values)

# %% [markdown] heading_collapsed=true
# # Day 10
#

# %% hidden=true
puzzle_input = open("../data/aoc/10.txt", mode="r").read()

puzzle_example1 = """.....
.S-7.
.|.|.
.L-J.
....."""

puzzle_example2 = """..F7.
.FJ|.
SJ.L7
|F--J
LJ..."""

# %% hidden=true
sketch = [list(line) for line in puzzle_input.splitlines()]
m, n = len(sketch), len(sketch[0])

s = puzzle_input.replace("\n", "").index("S")
s_i, s_j = s // n, s % n
assert sketch[s_i][s_j] == "S"

states = [(s_i, s_j)]
steps = 0

while states:
    steps += 1
    i, j = states.pop()
    top, right, bottom, left = False, False, False, False
    if sketch[i][j] in ["S", "|", "L", "J"]:
        top = True
    if sketch[i][j] in ["S", "-", "L", "F"]:
        right = True
    if sketch[i][j] in ["S", "|", "7", "F"]:
        bottom = True
    if sketch[i][j] in ["S", "-", "7", "J"]:
        left = True

    sketch[i][j] = "V"

    if top and i > 0 and sketch[i - 1][j] in ["|", "7", "F"]:
        states.append((i - 1, j))
    elif bottom and i < m - 1 and sketch[i + 1][j] in ["|", "L", "J"]:
        states.append((i + 1, j))
    elif left and j > 0 and sketch[i][j - 1] in ["-", "L", "F"]:
        states.append((i, j - 1))
    elif right and j < n - 1 and sketch[i][j + 1] in ["-", "7", "J"]:
        states.append((i, j + 1))

# print("\n".join(["".join(line) for line in sketch]))

print(steps // 2)

# %% hidden=true
sketch = [list(line) for line in puzzle_input.splitlines()]
big_sketch = [["."] * (n * 3) for i in range(m * 3)]
m, n = len(sketch), len(sketch[0])

s = puzzle_input.replace("\n", "").index("S")
s_i, s_j = s // n, s % n
assert sketch[s_i][s_j] == "S"

states = [(s_i, s_j)]
steps = 0
c = Counter()

while states:
    steps += 1
    i, j = states.pop()
    c[sketch[i][j]] += 1
    top, right, bottom, left = False, False, False, False
    if sketch[i][j] in ["S", "|", "L", "J"]:
        top = True
    if sketch[i][j] in ["S", "-", "L", "F"]:
        right = True
    if sketch[i][j] in ["S", "|", "7", "F"]:
        bottom = True
    if sketch[i][j] in ["S", "-", "7", "J"]:
        left = True

    sketch[i][j] = "V"
    for k in range(i * 3, i * 3 + 3):
        for l in range(j * 3, j * 3 + 3):
            big_sketch[k][l] = "*"

    big_sketch[i * 3 + 1][j * 3 + 1] = "V"

    if top:
        big_sketch[i * 3][j * 3 + 1] = "V"
    if right:
        big_sketch[i * 3 + 1][j * 3 + 2] = "V"
    if bottom:
        big_sketch[i * 3 + 2][j * 3 + 1] = "V"
    if left:
        big_sketch[i * 3 + 1][j * 3] = "V"

    if top and i > 0 and sketch[i - 1][j] in ["|", "7", "F"]:
        states.append((i - 1, j))
    elif bottom and i < m - 1 and sketch[i + 1][j] in ["|", "L", "J"]:
        states.append((i + 1, j))
    elif left and j > 0 and sketch[i][j - 1] in ["-", "L", "F"]:
        states.append((i, j - 1))
    elif right and j < n - 1 and sketch[i][j + 1] in ["-", "7", "J"]:
        states.append((i, j + 1))


# Flood fill
states = (
    [(0, j) for j in range(n * 3)]
    + [(m * 3 - 1, j) for j in range(n * 3)]
    + [(i, 0) for i in range(m * 3)]
    + [(i, n * 3 - 1) for i in range(m * 3)]
)

while states:
    i, j = states.pop()
    big_sketch[i][j] = "V"
    if i > 0 and big_sketch[i - 1][j] != "V":
        states.append((i - 1, j))
    if i < 3 * m - 1 and big_sketch[i + 1][j] != "V":
        states.append((i + 1, j))
    if j > 0 and big_sketch[i][j - 1] != "V":
        states.append((i, j - 1))
    if j < 3 * n - 1 and big_sketch[i][j + 1] != "V":
        states.append((i, j + 1))

# print("\n".join(["".join(line) for line in big_sketch]))

sum([big_sketch[i][j] == "." for i in range(m * 3) for j in range(n * 3)]) // 9

# %% [markdown] heading_collapsed=true
# # Day 11
#

# %% hidden=true
puzzle_input = open("../data/aoc/11.txt", mode="r").read()

puzzle_example = """...#......
.......#..
#.........
..........
......#...
.#........
.........#
..........
.......#..
#...#....."""

# %% hidden=true
image = [list(line) for line in puzzle_input.splitlines()]
m, n = len(image), len(image[0])
galaxies = []
for i in range(m):
    for j in range(n):
        if image[i][j] == "#":
            galaxies.append((i, j))

empty_rows = set(range(m)) - set(i for i, j in galaxies)
empty_columns = set(range(n)) - set(j for i, j in galaxies)

expanded_galaxies = []
for i, j in galaxies:
    expanded_galaxies.append(
        (
            sum([1 for r in empty_rows if r < i]) + i,
            sum([1 for c in empty_columns if c < j]) + j,
        )
    )

distances = 0
for (i1, j1), (i2, j2) in combinations(expanded_galaxies, 2):
    distances += abs(i1 - i2) + abs(j1 - j2)

distances

# %% hidden=true
image = [list(line) for line in puzzle_example.splitlines()]
m, n = len(image), len(image[0])
galaxies = []
for i in range(m):
    for j in range(n):
        if image[i][j] == "#":
            galaxies.append((i, j))

empty_rows = set(range(m)) - set(i for i, j in galaxies)
empty_columns = set(range(n)) - set(j for i, j in galaxies)

expanded_galaxies = []
for i, j in galaxies:
    expanded_galaxies.append(
        (
            999999 * sum([1 for r in empty_rows if r < i]) + i,
            999999 * sum([1 for c in empty_columns if c < j]) + j,
        )
    )

distances = 0
for (i1, j1), (i2, j2) in combinations(expanded_galaxies, 2):
    distances += abs(i1 - i2) + abs(j1 - j2)

distances

# %% [markdown] heading_collapsed=true
# # Day 12
#

# %% hidden=true
puzzle_input = open("../data/aoc/12.txt", mode="r").read()

puzzle_example = """???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1"""

# %% hidden=true
lines = puzzle_input.splitlines()

arrangements = 0

for line in lines:
    record, groups = line.split()
    groups = list(map(int, groups.split(",")))

    # Clean up
    record = record.strip(".")  # remove leading and trailing .
    record = re.sub(r"(\.+)", ".", record)  # remove duplicate .

    record = [".#" if c == "?" else c for c in record]
    pattern = "\.*" + "\.+".join(["#" * num for num in groups]) + "\.*"

    for r in product(*record):
        if re.fullmatch(pattern, "".join(r)) is not None:
            arrangements += 1

arrangements


# %% hidden=true
@functools.lru_cache(maxsize=None)
def test(record, groups):
    if not groups:
        if "#" not in set(record):
            return 1
        else:
            return 0
        return 1
    else:
        first, *rest = groups
        rest = tuple(rest)

    if not record:
        return 0

    if record[0] == ".":
        return test(record[1:], groups)
    elif record[0] == "?":
        return test("#" + record[1:], groups) + test("." + record[1:], groups)
    elif record[0] == "#":
        if "." not in set(record[:first]) and record[first] != "#":
            return test(record[first + 1 :], rest)
        else:
            return 0


arrangements = []
lines = puzzle_input.splitlines()
for line in lines:
    record, groups = line.split()

    # Unfold
    record = "?".join([record] * 5)
    groups = ",".join([groups] * 5)

    # Clean up
    record = record.strip(".")  # remove leading and trailing .
    record = re.sub(r"(\.+)", ".", record)  # remove duplicate .

    groups = list(map(int, groups.split(",")))

    arrangements.append(test(record + ".", tuple(groups)))

sum(arrangements)

# %% [markdown] heading_collapsed=true
# # Day 13
#

# %% hidden=true
puzzle_input = open("../data/aoc/13.txt", mode="r").read()

puzzle_example = """#.##..##.
..#.##.#.
##......#
##......#
..#.##.#.
..##..##.
#.#.##.#.

#...##..#
#....#..#
..##..###
#####.##.
#####.##.
..##..###
#....#..#"""

# %% hidden=true
patterns = [pattern.split() for pattern in puzzle_input.split("\n\n")]
total = 0
for pattern in patterns:
    m, n = len(pattern), len(pattern[0])

    for i in range(1, m):
        for c in range(min(i, m - i)):
            if pattern[i - 1 - c] != pattern[i + c]:
                break
        else:
            total += 100 * i

    for j in range(1, n):
        for c in range(min(j, n - j)):
            a = "".join(pattern[i][j - 1 - c] for i in range(m))
            b = "".join(pattern[i][j + c] for i in range(m))
            if a != b:
                break
        else:
            total += j

total

# %% hidden=true
patterns = [pattern.split() for pattern in puzzle_input.split("\n\n")]
total = 0
for pattern in patterns:
    m, n = len(pattern), len(pattern[0])

    for i in range(1, m):
        edits = 0
        for c in range(min(i, m - i)):
            for j in range(n):
                if pattern[i - 1 - c][j] != pattern[i + c][j]:
                    edits += 1

        if edits == 1:
            total += 100 * i
            continue

    for j in range(1, n):
        edits = 0
        for c in range(min(j, n - j)):
            for i in range(m):
                if pattern[i][j - 1 - c] != pattern[i][j + c]:
                    edits += 1

        if edits == 1:
            total += j
            continue

total

# %% [markdown] heading_collapsed=true
# # Day 14
#

# %% hidden=true
puzzle_input = open("../data/aoc/14.txt", mode="r").read()

puzzle_example = """O....#....
O.OO#....#
.....##...
OO.#O....O
.O.....O#.
O.#..O.#.#
..O..#O..O
.......O..
#....###..
#OO..#...."""

# %% hidden=true
platform = [list(line) for line in puzzle_input.splitlines()]
m, n = len(platform), len(platform[0])


def roll(s):
    s = s.replace(".", "X")
    s = re.split(r"(\#)", s)
    s = ["".join(sorted(a)) for a in s]
    s = "".join(s)
    s = s.replace("X", ".")
    return s


for j in range(n):
    s = "".join(platform[i][j] for i in range(m))
    s = roll(s)
    for i, c in enumerate(s):
        platform[i][j] = c

score = 0
for i in range(m):
    score += (m - i) * "".join(platform[i]).count("O")
score

# %% hidden=true
platform = [list(line) for line in puzzle_input.splitlines()]
m, n = len(platform), len(platform[0])


def roll(s, reverse=False):
    s = s.replace(".", "X")
    s = re.split(r"(\#)", s)
    s = ["".join(sorted(a, reverse=reverse)) for a in s]
    s = "".join(s)
    s = s.replace("X", ".")
    return s


def score(platform):
    score = 0
    for i in range(m):
        score += (m - i) * "".join(platform[i]).count("O")
    return score


def spin(platform):
    # North
    for j in range(n):
        s = "".join(platform[i][j] for i in range(m))
        s = roll(s)
        for i, c in enumerate(s):
            platform[i][j] = c

    # West
    for i in range(m):
        s = "".join(platform[i][j] for j in range(n))
        s = roll(s)
        for j, c in enumerate(s):
            platform[i][j] = c

    # South
    for j in range(n):
        s = "".join(platform[i][j] for i in range(m))
        s = roll(s, reverse=True)
        for i, c in enumerate(s):
            platform[i][j] = c

    # East
    for i in range(m):
        s = "".join(platform[i][j] for j in range(n))
        s = roll(s, reverse=True)
        for j, c in enumerate(s):
            platform[i][j] = c

    return platform


for i in range(200):
    platform = spin(platform)
    score_ = score(platform)
    print(i, i % 14 == 5, score_)

# %% [markdown] heading_collapsed=true
# # Day 15
#

# %% hidden=true
puzzle_input = open("../data/aoc/15.txt", mode="r").read()

puzzle_example = """rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7"""


# %% hidden=true
def hash_(s):
    v = 0
    for c in s:
        v += ord(c)
        v *= 17
        v %= 256
    return v


assert hash_("HASH") == 52

outputs = [hash_(step) for step in puzzle_input.strip().split(",")]
sum(outputs)

# %% hidden=true
dd = defaultdict(OrderedDict)

for step in puzzle_input.strip().split(","):
    if step[-1] == "-":
        action = "REMOVE"
        label = step[:-1]
    else:
        action = "ADD"
        label, number = step.split("=")
        number = int(number)

    box = hash_(label)
    if action == "ADD":
        dd[box][label] = number
    else:
        if label in dd[box]:
            del dd[box][label]

total = 0
for key, od in dd.items():
    for i, (k, v) in enumerate(od.items()):
        total += (key + 1) * (i + 1) * v

total

# %% [markdown] heading_collapsed=true
# # Day 16
#

# %% hidden=true
puzzle_input = open("../data/aoc/16.txt", mode="r").read()

puzzle_example = r""".|...\....
|.-.\.....
.....|-...
........|.
..........
.........\
..../.\\..
.-.-/..|..
.|....-|.\
..//.|...."""

# %% hidden=true
grid = [list(line) for line in puzzle_input.splitlines()]
m, n = len(grid), len(grid[0])

beams = [(0, 0, ">")]
tiles = set()

while beams:
    i, j, direction = beams.pop(0)
    if i < 0 or i >= m or j < 0 or j >= n:
        continue

    if grid[i][j] == direction:
        continue

    tiles.add((i, j))

    if grid[i][j] == "-" and direction in ["^", "V"]:
        beams.append((i, j - 1, "<"))
        beams.append((i, j + 1, ">"))
        continue
    elif grid[i][j] == "|" and direction in [">", "<"]:
        beams.append((i - 1, j, "^"))
        beams.append((i + 1, j, "V"))
        continue
    elif grid[i][j] == "/":
        if direction == "^":
            beams.append((i, j + 1, ">"))
            continue
        elif direction == ">":
            beams.append((i - 1, j, "^"))
            continue
        elif direction == "V":
            beams.append((i, j - 1, "<"))
            continue
        elif direction == "<":
            beams.append((i + 1, j, "V"))
            continue

    elif grid[i][j] == "\\":
        if direction == "^":
            beams.append((i, j - 1, "<"))
            continue
        elif direction == ">":
            beams.append((i + 1, j, "V"))
            continue
        elif direction == "V":
            beams.append((i, j + 1, ">"))
            continue
        elif direction == "<":
            beams.append((i - 1, j, "^"))
            continue

    if direction == "^":
        beams.append((i - 1, j, direction))
    elif direction == ">":
        beams.append((i, j + 1, direction))
    elif direction == "V":
        beams.append((i + 1, j, direction))
    elif direction == "<":
        beams.append((i, j - 1, direction))

    if grid[i][j] not in ["-", "|", "/", "\\"]:
        if grid[i][j] == ".":
            grid[i][j] = direction
        elif grid[i][j] in ["^", ">", "V", "<"]:
            grid[i][j] = "2"
        else:
            grid[i][j] = str(1 + int(grid[i][j]))

len(tiles)


# %% hidden=true
def compute_energized_tiles(beam):
    grid = [list(line) for line in puzzle_input.splitlines()]
    m, n = len(grid), len(grid[0])

    beams = [beam]
    tiles = set()

    while beams:
        i, j, direction = beams.pop(0)
        if i < 0 or i >= m or j < 0 or j >= n:
            continue

        if grid[i][j] == direction:
            continue

        tiles.add((i, j))

        if grid[i][j] == "-" and direction in ["^", "V"]:
            beams.append((i, j - 1, "<"))
            beams.append((i, j + 1, ">"))
            continue
        elif grid[i][j] == "|" and direction in [">", "<"]:
            beams.append((i - 1, j, "^"))
            beams.append((i + 1, j, "V"))
            continue
        elif grid[i][j] == "/":
            if direction == "^":
                beams.append((i, j + 1, ">"))
                continue
            elif direction == ">":
                beams.append((i - 1, j, "^"))
                continue
            elif direction == "V":
                beams.append((i, j - 1, "<"))
                continue
            elif direction == "<":
                beams.append((i + 1, j, "V"))
                continue

        elif grid[i][j] == "\\":
            if direction == "^":
                beams.append((i, j - 1, "<"))
                continue
            elif direction == ">":
                beams.append((i + 1, j, "V"))
                continue
            elif direction == "V":
                beams.append((i, j + 1, ">"))
                continue
            elif direction == "<":
                beams.append((i - 1, j, "^"))
                continue

        if direction == "^":
            beams.append((i - 1, j, direction))
        elif direction == ">":
            beams.append((i, j + 1, direction))
        elif direction == "V":
            beams.append((i + 1, j, direction))
        elif direction == "<":
            beams.append((i, j - 1, direction))

        if grid[i][j] not in ["-", "|", "/", "\\"]:
            if grid[i][j] == ".":
                grid[i][j] = direction
            elif grid[i][j] in ["^", ">", "V", "<"]:
                grid[i][j] = "2"
            else:
                grid[i][j] = str(1 + int(grid[i][j]))

    return len(tiles)


grid = [list(line) for line in puzzle_input.splitlines()]
m, n = len(grid), len(grid[0])

beams = []
beams.extend([(0, j, "V") for j in range(n)])
beams.extend([(m - 1, j, "^") for j in range(n)])
beams.extend([(i, 0, "^") for i in range(m)])
beams.extend([(i, n - 1, "^") for i in range(m)])

energized_tiles = []
for beam in beams:
    energized_tiles.append(compute_energized_tiles(beam))

max(energized_tiles)

# %% [markdown] heading_collapsed=true
# # Day 17\*
#

# %% hidden=true
puzzle_input = open("../data/aoc/17.txt", mode="r").read()

puzzle_example1 = """2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533"""

puzzle_example2 = """111111111111
999999999991
999999999991
999999999991
999999999991"""


# %% hidden=true
def dijkstra(block, start_block, direction, min_steps, max_steps):
    m, n = len(block), len(block[0])
    distances = {}
    heap = [(0, start_block, direction)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while heap:
        distance, position, direction = heapq.heappop(heap)

        if (position, direction) in distances:
            continue

        distances[(position, direction)] = distance

        f_i, f_j = direction
        for d_i, d_j in set(directions) - set([(f_i, f_j), (-f_i, -f_j)]):
            i, j = position
            d = distance

            for index in range(1, max_steps + 1):
                i += d_i
                j += d_j

                if 0 <= i < m and 0 <= j < n:
                    d += block[i][j]
                    if index >= min_steps:
                        heapq.heappush(heap, (d, (i, j), (d_i, d_j)))

    return distances


# %% hidden=true
block = [list(map(int, line)) for line in puzzle_input.split()]
distances = dijkstra(block, (0, 0), (0, 0), 1, 3)
m, n = len(block), len(block[0])
{k: v for k, v in distances.items() if k[0] == (m - 1, n - 1)}

# %% hidden=true
block = [list(map(int, line)) for line in puzzle_input.split()]
distances = dijkstra(block, (0, 0), (0, 0), 4, 10)
m, n = len(block), len(block[0])
{k: v for k, v in distances.items() if k[0] == (m - 1, n - 1)}

# %% [markdown] heading_collapsed=true
# # Day 18\*
#

# %% hidden=true
puzzle_input = open("../data/aoc/18.txt", mode="r").read()

puzzle_example = """R 6 (#70c710)
D 5 (#0dc571)
L 2 (#5713f0)
D 2 (#d2c081)
R 2 (#59c680)
D 2 (#411b91)
L 5 (#8ceee2)
U 2 (#caa173)
L 1 (#1b58a2)
U 2 (#caa171)
R 2 (#7807d2)
U 3 (#a77fa3)
L 2 (#015232)
U 2 (#7a21e3)"""

# %% hidden=true
plan = puzzle_input.splitlines()

points = [(0, 0)]
directions = {"D": (1, 0), "L": (0, -1), "R": (0, 1), "U": (-1, 0)}
m, n = -float("inf"), -float("inf")

# Find dimensions of lagoon
for line in plan:
    direction, distance, _ = line.split()

    x, y = points[-1]

    d_x, d_y = directions[direction]
    x += d_x * int(distance)
    y += d_y * int(distance)

    m = max(m, x)
    n = max(n, y)

    points.append((x, y))

xs, ys = zip(*points)
min_x, max_x = min(xs), max(xs)
min_y, max_y = min(ys), max(ys)
m = max_x - min_x + 1
n = max_y - min_y + 1

# Draw lagoon with border 1
lagoon = [["." for y in range(n + 2)] for x in range(m + 2)]

points = [(0, 0)]
for line in plan:
    direction, distance, _ = line.split()

    x, y = points[-1]

    d_x, d_y = directions[direction]

    for i in range(int(distance)):
        lagoon[x - min_x + 1][y - min_y + 1] = "#"
        x += d_x
        y += d_y

    points.append((x, y))

unvisited = set([(0, 0)])
visited = set()
while unvisited:
    x, y = unvisited.pop()
    visited.add((x, y))
    for d_x, d_y in directions.values():
        X = x + d_x
        Y = y + d_y
        if 0 <= X < m + 2 and 0 <= Y < n + 2:
            if (X, Y) not in visited and lagoon[X][Y] == ".":
                unvisited.add((X, Y))

(m + 2) * (n + 2) - len(visited)

# %% hidden=true
plan = puzzle_input.splitlines()

points = [(0, 0)]
directions = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}

m, n = -float("inf"), -float("inf")

horizontals = defaultdict(list)
horizontal_distance = 0
vertical_distance = 0

# Find dimensions of lagoon
for line in plan:
    _, _, color = line.split()
    distance = int(color[2:-2], 16)
    direction = int(color[-2], 16)

    x, y = points[-1]

    if direction == 2:
        horizontals[x].append((y - distance, y + 1))
    elif direction == 0:
        horizontals[x].append((y, y + distance + 1))

    if direction in [0, 2]:
        horizontal_distance += distance
    else:
        vertical_distance += distance

    d_x, d_y = directions[direction]
    x += d_x * int(distance)
    y += d_y * int(distance)

    points.append((x, y))

xs, ys = zip(*points)
xs = sorted(set(xs))
ys = sorted(set(ys))

total = 0

for y1, y2 in zip(ys, ys[1:]):
    flag = False
    subtotal = 0
    for x1, x2 in zip(xs, xs[1:]):
        for line_min, line_max in horizontals[x1]:
            if line_min <= y1 < y2 < line_max:
                flag = not flag

        if flag:
            a = (x2 - x1) * (y2 - y1)
            subtotal += a
    total += subtotal

total + (horizontal_distance + vertical_distance) // 2 + 1

# %% [markdown] heading_collapsed=true
# # Day 19
#

# %% hidden=true
puzzle_input = open("../data/aoc/19.txt", mode="r").read()

puzzle_example = """px{a<2006:qkq,m>2090:A,rfg}
pv{a>1716:R,A}
lnx{m>1548:A,A}
rfg{s<537:gd,x>2440:R,A}
qs{s>3448:A,lnx}
qkq{x<1416:A,crn}
crn{x>2662:A,R}
in{s<1351:px,qqz}
qqz{s>2770:qs,m<1801:hdj,R}
gd{a>3333:R,R}
hdj{m>838:A,pv}

{x=787,m=2655,a=1222,s=2876}
{x=1679,m=44,a=2067,s=496}
{x=2036,m=264,a=79,s=2244}
{x=2461,m=1339,a=466,s=291}
{x=2127,m=1623,a=2188,s=1013}"""

# %% hidden=true
workflows_, parts_ = puzzle_input.split("\n\n")

workflows = {}
for workflow in workflows_.split():
    key, rules = workflow.split("{")
    rules = rules[:-1].split(",")
    rs = []
    for rule in rules[:-1]:
        condition, result = rule.split(":")

        if "<" in condition:
            a, b = condition.split("<")
            rs.append(((1, int(b)), a, result))
        elif ">" in condition:
            a, b = condition.split(">")
            rs.append(((int(b) + 1, 4000), a, result))

    rs.append(((1, 4000), "x", rules[-1]))

    workflows[key] = rs

parts = []
for part_ in parts_.split():
    part = {}
    for rating in part_[1:-1].split(","):
        key, value = rating.split("=")
        part[key] = int(value)

    parts.append(part)

total = 0
for part in parts:
    step = "in"
    while step not in ["A", "R"]:
        for rule in workflows[step]:
            (low, high), prop, next_step = rule
            if low <= part[prop] < high:
                step = next_step
                break

    if step == "A":
        total += sum(part.values())
total

# %% hidden=true
workflows_, parts_ = puzzle_input.split("\n\n")

workflows = {}
for workflow in workflows_.split():
    key, rules = workflow.split("{")
    rules = rules[:-1].split(",")
    rs = []
    for rule in rules[:-1]:
        condition, result = rule.split(":")

        if "<" in condition:
            a, b = condition.split("<")
            rs.append((a, "<", int(b), result))
        elif ">" in condition:
            a, b = condition.split(">")
            rs.append((a, ">", int(b), result))

    rs.append(rules[-1])

    workflows[key] = rs

accepted_parts = []
remaining = [("in", {"x": (1, 4001), "m": (1, 4001), "a": (1, 4001), "s": (1, 4001)})]
while remaining:
    step, part = remaining.pop()
    if 0 in [high - low for low, high in part.values()]:
        continue

    if step == "A":
        accepted_parts.append(part)
        continue
    elif step == "R":
        continue

    for rule in workflows[step][:-1]:
        k, sign, value, next_step = rule
        low, high = part[k]

        if sign == "<":
            new_part = part.copy()
            new_part[k] = min(low, value), min(high, value)
            remaining.append((next_step, new_part))
            part[k] = max(low, value), max(high, value)
        elif sign == ">":
            new_part = part.copy()
            new_part[k] = max(low, value + 1), max(high, value + 1)
            remaining.append((next_step, new_part))
            part[k] = min(low, value + 1), min(high, value + 1)

    final_step = workflows[step][-1]
    remaining.append((final_step, part))

total = 0
for part in accepted_parts:
    total += math.prod([high - low for low, high in part.values()])
total

# %% [markdown] heading_collapsed=true
# # Day 20
#

# %% hidden=true
puzzle_input = open("../data/aoc/20.txt", mode="r").read()

puzzle_example1 = """broadcaster -> a, b, c
%a -> b
%b -> c
%c -> inv
&inv -> a"""

puzzle_example2 = """broadcaster -> a
%a -> inv, con
&inv -> b
%b -> con
&con -> output"""


# %% hidden=true
class Button:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "Button"

    def update_input(self, name):
        pass

    def transmit(self):
        return [(self.name, "low", "broadcaster")]


class Broadcaster:
    def __init__(self, name, outputs):
        self.name = name
        self.outputs = outputs

    def __repr__(self):
        return f"Broadcaster(outputs={self.outputs})"

    def update_input(self, name):
        pass

    def transmit(self, input, pulse):
        return [(self.name, pulse, output) for output in self.outputs]


class Conjunction:
    def __init__(self, name, outputs):
        self.name = name
        self.inputs = {}
        self.outputs = outputs

    def __repr__(self):
        return f"Conjunction(inputs={list(self.inputs.keys())}, outputs={self.outputs})"

    def update_input(self, name):
        self.inputs[name] = "low"

    def transmit(self, input, pulse):
        self.inputs[input] = pulse
        if set(self.inputs.values()) == set(["high"]):
            return [(self.name, "low", output) for output in self.outputs]
        else:
            return [(self.name, "high", output) for output in self.outputs]


class FlipFlop:
    def __init__(self, name, outputs):
        self.name = name
        self.outputs = outputs
        self.state = False

    def __repr__(self):
        return f"FlipFlop(state={self.state}, outputs={self.outputs})"

    def update_input(self, name):
        pass

    def transmit(self, input, pulse):
        if pulse == "low":
            self.state = not self.state
            return [
                (self.name, "high" if self.state else "low", output)
                for output in self.outputs
            ]
        else:
            return []


def generate_config(puzzle):
    config = {"button": Button("button")}

    for line in puzzle.splitlines():
        name, outputs = line.split(" -> ")
        outputs = outputs.split(", ")

        if name == "broadcaster":
            config[name] = Broadcaster(name, outputs)
        elif name[0] == "&":
            name = name[1:]
            config[name] = Conjunction(name, outputs)
        else:
            name = name[1:]
            config[name] = FlipFlop(name, outputs)

    for line in puzzle.splitlines():
        name, outputs = line.split(" -> ")
        outputs = outputs.split(", ")

        if name[0] in ["&", "%"]:
            name = name[1:]

        for output in outputs:
            if output in config:
                config[output].update_input(name)

    return config


# %% hidden=true
config = generate_config(puzzle_example1)

counter = Counter()

for _ in range(1_000):
    pulses = config["button"].transmit()
    while pulses:
        input, pulse, output = pulses.pop(0)
        #         print(input, pulse, output)
        counter[pulse] += 1

        if output in config:
            pulses.extend(config[output].transmit(input, pulse))

counter["low"] * counter["high"]

# %% hidden=true
# Find cycles

config = generate_config(puzzle_input)
connections = ["xc", "th", "pd", "bp"]

dd = defaultdict(list)

for i in range(10_000):
    pulses = config["button"].transmit()
    while pulses:
        input, pulse, output = pulses.pop(0)
        #         print(input, pulse, output)
        if input in connections and pulse == "high":
            print(i, input, pulse, output)

        if output in config:
            pulses.extend(config[output].transmit(input, pulse))

# %% hidden=true
3823 * 3847 * 3877 * 4001

# %% [markdown] heading_collapsed=true
# # Day 21\*
#

# %% hidden=true
puzzle_input = open("../data/aoc/21.txt", mode="r").read()

puzzle_example = """...........
.....###.#.
.###.##..#.
..#.#...#..
....#.#....
.##..S####.
.##..#...#.
.......##..
.##.#.####.
.##..##.##.
..........."""

# %% hidden=true
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

garden = [list(line) for line in puzzle_input.splitlines()]
m, n = len(garden), len(garden[0])

index = puzzle_input.replace("\n", "").index("S")
x, y = index % m, index % n

garden[x][y] = "O"

counter = 0
for _ in range(64):
    new_garden = copy.deepcopy(garden)
    counter = 0

    for x in range(m):
        for y in range(n):
            if garden[x][y] == "O":
                new_garden[x][y] = "."
                for dx, dy in directions:
                    if 0 <= x + dx < m and 0 <= y + dy < n:
                        if garden[x + dx][y + dy] != "#":
                            if new_garden[x + dx][y + dy] != "O":
                                new_garden[x + dx][y + dy] = "O"
                                counter += 1

    garden = new_garden

counter


# %% hidden=true
def process(positions, m, n, i, j):
    return len(
        [
            x
            for x, y in positions
            if i * m <= x < (i + 1) * m and j * n <= y < (j + 1) * n
        ]
    )


directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

garden = [list(line) for line in puzzle_input.splitlines()]
m, n = len(garden), len(garden[0])

index = puzzle_input.replace("\n", "").index("S")
x, y = index % m, index % n

assert garden[x][y] == "S"

positions = set([(x, y)])

num_positions = []
for step in range(1, 460):
    new_positions = set()
    for x, y in positions:
        for dx, dy in directions:
            if garden[(x + dx) % m][(y + dy) % n] != "#":
                new_positions.add((x + dx, y + dy))
    positions = new_positions

    if step % m == m // 2:
        print(step // m, step, len(positions))
        num_positions.append(len(positions))

# %% hidden=true
num_positions = [3778, 33695, 93438, 183007]
print(num_positions)
a = np.diff(num_positions)
print(a)
b = np.diff(a)
print(b)
a = np.concatenate(([29917], [29826] * (202300 - 1))).cumsum()
print(a)
num_positions = np.concatenate(([3778], a)).cumsum()
assert len(num_positions) == 202301
print(num_positions)

# %% [markdown] heading_collapsed=true
# # Day 22
#

# %% hidden=true
puzzle_input = open("../data/aoc/22.txt", mode="r").read()

puzzle_example = """1,0,1~1,2,1
0,0,2~2,0,2
0,2,3~2,2,3
0,0,4~0,2,4
2,0,5~2,2,5
0,1,6~2,1,6
1,1,8~1,1,9"""

# %% hidden=true
bricks = {}
points = []
for id, brick in enumerate(puzzle_input.splitlines()):
    start, end = brick.split("~")
    x1, y1, z1 = map(int, start.split(","))
    x2, y2, z2 = map(int, end.split(","))
    assert x2 >= x1
    assert y2 >= y1
    assert z2 >= z1
    bricks[id + 1] = {
        "x": (x1, x2 + 1),
        "y": (y1, y2 + 1),
        "z": (z1, z2 + 1),
        "z_length": z2 - z1 + 1,
    }
    points.extend([(x1, y1, z1), (x2, y2, z2)])

bricks = OrderedDict(sorted(bricks.items(), key=lambda brick: brick[1]["z"][0]))

xs, ys, zs = zip(*points)
spaces = np.ones((max(xs) + 1, max(ys) + 1), dtype=int)
top_bricks = np.zeros((max(xs) + 1, max(ys) + 1), dtype=int)
supports = {k: set() for k in bricks.keys()}
supports[0] = set()

for id, brick in bricks.items():
    (x1, x2), (y1, y2), (z1, z2), z_length = brick.values()
    brick_points = [(x, y) for x in range(x1, x2) for y in range(y1, y2)]
    max_z = max([spaces[x][y] for x, y in brick_points])
    contact_points = [(x, y) for x, y in brick_points if spaces[x][y] == max_z]

    # Set z range for current brick
    brick["z"] = (max_z, max_z + z_length)

    # Update spaces
    for i in range(x1, x2):
        for j in range(y1, y2):
            spaces[i][j] = max_z + z_length

    # Update supports
    for x, y in contact_points:
        supports[top_bricks[x][y]].add(id)

    # Update top bricks
    for x, y in brick_points:
        top_bricks[x][y] = id

del supports[0]
supports

# %% hidden=true
counter = Counter([i for k, v in supports.items() for i in v])

safe = []
unsafe = []
for k, v in supports.items():
    if all([counter[id] > 1 for id in v]):
        safe.append(k)
    else:
        unsafe.append(k)

len(safe)

# %% hidden=true
fall = 0
for id_ in unsafe:
    c = counter.copy()
    s = supports.copy()
    disintegrating = [id_]
    disintegrated = []
    while disintegrating:
        id = disintegrating.pop()
        disintegrated.append(id)
        for i in s[id]:
            c[i] -= 1
            if c[i] == 0:
                disintegrating.append(i)

    fall += len(disintegrated) - 1

fall

# %% [markdown] heading_collapsed=true
# # Day 23
#

# %% hidden=true
puzzle_input = open("../data/aoc/23.txt", mode="r").read()

puzzle_example = """#.#####################
#.......#########...###
#######.#########.#.###
###.....#.>.>.###.#.###
###v#####.#v#.###.#.###
###.>...#.#.#.....#...#
###v###.#.#.#########.#
###...#.#.#.......#...#
#####.#.#.#######.#.###
#.....#.#.#.......#...#
#.#####.#.#.#########v#
#.#...#...#...###...>.#
#.#.#v#######v###.###v#
#...#.>.#...>.>.#.###.#
#####v#.#.###v#.#.###.#
#.....#...#...#.#.#...#
#.#########.###.#.#.###
#...###...#...#...#.###
###.###.#.###v#####v###
#...#...#.#.>.>.#.>.###
#.###.###.#.###.#.#v###
#.....###...###...#...#
#####################.#"""

# %% hidden=true
trails = [list(line) for line in puzzle_input.splitlines()]
m, n = len(trails), len(trails[0])
start = 0, trails[0].index(".")
end = len(trails) - 1, trails[-1].index(".")

DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

steps = defaultdict(int)
slippery = 0


def process(trails, unvisited):
    while unvisited:
        global slippery, steps
        step, position = unvisited.pop()
        x, y = position

        trails[x][y] = "#"
        steps[position] = max(steps[position], step)

        if position == end:
            continue

        # Filter directions
        directions = []
        for dx, dy in DIRECTIONS:
            if 0 <= (x + dx) < m and 0 <= (y + dy) < n:
                c = trails[x + dx][y + dy]

                if (dx, dy) == (0, -1) and c == ">":
                    slippery += 1
                    continue
                elif (dx, dy) == (-1, 0) and c == "v":
                    slippery += 1
                    continue
                elif c != "#":
                    directions.append((dx, dy))

        if len(directions) == 1:
            dx, dy = directions[0]
            unvisited.append((step + 1, (x + dx, y + dy)))
        else:
            for dx, dy in directions:
                new_unvisited = unvisited.copy() + [(step + 1, (x + dx, y + dy))]
                process(copy.deepcopy(trails), new_unvisited)


process(trails, [(0, (x, y))])
steps[end]

# %% hidden=true
trails = [list(line) for line in puzzle_input.splitlines()]
m, n = len(trails), len(trails[0])
start = 0, trails[0].index(".")
end = len(trails) - 1, trails[-1].index(".")

# Compute junctions
junctions = set([start, end])
for x in range(m):
    for y in range(n):
        if trails[x][y] == "#":
            continue

        branches = 0
        for dx, dy in DIRECTIONS:
            if 0 <= (x + dx) < m and 0 <= (y + dy) < n:
                if trails[x + dx][y + dy] != "#":
                    branches += 1
        if branches > 2:
            junctions.add((x, y))

# Computer segments
segments = defaultdict(set)
for junction in junctions:
    trails_ = copy.deepcopy(trails)
    unvisited = [(0, junction)]
    while unvisited:
        step, position = unvisited.pop()
        x, y = position
        trails_[x][y] = "#"
        if position != junction and position in junctions:
            segments[junction].add((step, position))
            segments[position].add((step, junction))
        else:
            for dx, dy in DIRECTIONS:
                if 0 <= (x + dx) < m and 0 <= (y + dy) < n:
                    if trails_[x + dx][y + dy] != "#":
                        unvisited.append((step + 1, (x + dx, y + dy)))


def process(hike):
    global longest_hike
    steps, segment_start = hike[-1]
    hike_junctions = set([junction for _, junction in hike])
    if segment_start == end:
        longest_hike = max(longest_hike, sum([s for s, _ in hike]))
    else:
        for segment_steps, segment_end in segments[segment_start]:
            if segment_end not in hike_junctions:
                process(hike + [(segment_steps, segment_end)])


longest_hike = 0
process([(0, (0, 1))])
longest_hike

# %% [markdown] heading_collapsed=true
# # Day 24\*
#

# %% hidden=true
puzzle_input = open("../data/aoc/24.txt", mode="r").read()

puzzle_example = """19, 13, 30 @ -2,  1, -2
18, 19, 22 @ -1, -1, -2
20, 25, 34 @ -2, -2, -4
12, 31, 28 @ -1, -2, -1
20, 19, 15 @  1, -5, -3"""

# %% hidden=true
# t_min, t_max = (7, 27)
t_min, t_max = (200000000000000, 400000000000000)

hailstones = []
# for line in puzzle_example.splitlines():
for line in puzzle_input.splitlines():
    position, velocity = line.split(" @ ")
    position = tuple(map(int, position.split(", ")))
    velocity = tuple(map(int, velocity.split(", ")))
    hailstones.append((position, velocity))

intersections = 0

for hailstone1, hailstone2 in combinations(hailstones, r=2):
    (px1, py1, pz1), (vx1, vy1, vz1) = hailstone1
    (px2, py2, pz2), (vx2, vy2, vz2) = hailstone2

    #     print(f"Hailstone A: {px1}, {py1}, {pz1} @ {vx1}, {vy1}, {vz1}")
    #     print(f"Hailstone A: {px2}, {py2}, {pz2} @ {vx2}, {vy2}, {vz2}")

    # y = ax + c
    a = vy1 / vx1
    c = py1 - a * px1
    px1_min, px1_max = (px1, float("inf")) if vx1 > 0 else (float("-inf"), px1)
    py1_min, py1_max = (py1, float("inf")) if vy1 > 0 else (float("-inf"), py1)
    assert px1_min < px1_max

    # y = bx + d
    b = vy2 / vx2
    d = py2 - b * px2
    px2_min, px2_max = (px2, float("inf")) if vx2 > 0 else (float("-inf"), px2)
    py2_min, py2_max = (py2, float("inf")) if vy2 > 0 else (float("-inf"), py2)
    assert px2_min < px2_max

    if a != b:
        ix = (d - c) / (a - b)
        iy = a * ix + c
        condition1 = (t_min <= ix <= t_max) and (t_min <= iy <= t_max)
        condition2 = px1_min <= ix <= px1_max
        condition3 = px2_min <= ix <= px2_max

        if condition1 and condition2 and condition3:
            #             print(f"Hailstones' paths will cross inside the test area (at x={ix}, y={iy}).")
            intersections += 1
        elif not condition2 and not condition3:
            #             print("Hailstones' paths crossed in the past for both hailstones.")
            pass
        elif not condition2:
            #             print("Hailstones' paths crossed in the past for hailstone A.")
            pass
        elif not condition3:
            #             print("Hailstones' paths crossed in the past for hailstone B.")
            pass
        else:
            #             print(f"Hailstones' paths will cross outside the test area (at x={ix}, y={iy}).")
            pass
    else:
        if c == d:
            if vx1 == 0:  # hailstones' path is vertical
                if py1_min < py2_max or py2_min < py1_max:
                    #                     print("Hailstones' paths are the same line; they overlap.")
                    intersections += 1
                else:
                    #                     print("Hailstones' paths are the same line; they don't overlap.")
                    pass
            else:
                if px1_min < px2_max or px2_min < px1_max:
                    #                     print("Hailstones' paths are the same line; they overlap.")
                    intersections += 1
                else:
                    #                     print("Hailstones' paths are the same line; they don't overlap.")
                    pass
        else:
            #             print("Hailstones' paths are parallel; they never intersect.")
            pass

print(intersections)

# %% hidden=true
hailstones = []
# for line in puzzle_example.splitlines():
for line in puzzle_input.splitlines():
    position, velocity = line.split(" @ ")
    position = tuple(map(int, position.split(", ")))
    velocity = tuple(map(int, velocity.split(", ")))
    hailstones.append((position, velocity))

hailstones

# %% hidden=true
from sympy import Eq, solve, symbols

px1, py1, pz1 = symbols("px1, py1, pz1", integer=True)
vx1, vy1, vz1 = symbols("vx1, vy1, vz1", integer=True)

system = []
for (px2, py2, pz2), (vx2, vy2, vz2) in hailstones:
    system.append(Eq((px2 - px1) * (vy1 - vy2), (py2 - py1) * (vx1 - vx2)))
    system.append(Eq((px2 - px1) * (vz1 - vz2), (pz2 - pz1) * (vx1 - vx2)))

px1, py1, pz1, vx1, vy1, vz1 = solve(system, [px1, py1, pz1, vx1, vy1, vz1])[0]
sum([px1, py1, pz1])

# %% [markdown] heading_collapsed=true
# # Day 25
#

# %% hidden=true
puzzle_input = open("../data/aoc/25.txt", mode="r").read()

puzzle_example = """jqt: rhn xhk nvd
rsh: frs pzl lsr
xhk: hfx
cmg: qnr nvd lhk bvb
rhn: xhk bvb hfx
bvb: xhk hfx
pzl: lsr hfx nvd
qnr: nvd
ntq: jqt hfx bvb xhk
nvd: lhk
lsr: lhk
rzs: qnr cmg lsr rsh
frs: qnr lhk lsr"""

# %% hidden=true
nodes = set()
edges = set()
for line in puzzle_input.splitlines():
    source, targets = line.split(": ")
    nodes.add(source)
    for target in targets.split():
        nodes.add(target)
        edges.add((source, target))

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
cut_value, partition = nx.stoer_wagner(G)
assert cut_value == 3
len(partition[0]) * len(partition[1])
