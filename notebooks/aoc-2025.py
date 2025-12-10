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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import functools
import itertools
import operator

# %% [markdown]
# # Day 1
#

# %%
puzzle_input = open("../data/aoc/2025/01.txt", mode="r").read()

puzzle_example = """L68
L30
R48
L5
R60
L55
L1
L99
R14
L82
"""

# %%
i = 50
zeros = 0

for line in puzzle_input.splitlines():
    d = line[0]
    n = int(line[1:])

    if d == "L":
        i = (i - n) % 100
    else:
        i = (i + n) % 100

    if i == 0:
        zeros += 1

print(zeros)


# %%
i = 50
zeros = 0

for line in puzzle_input.splitlines():
    d = line[0]
    n = int(line[1:])

    zeros += n // 100
    n = n % 100

    if d == "L":
        if i - n <= 0 < i:
            zeros += 1
        i = (i - n) % 100
    else:
        if 99 < i + n:
            zeros += 1
        i = (i + n) % 100

    # print(line, i, zeros)

print(zeros)


# %% [markdown]
# # Day 2
#

# %%
puzzle_input = open("../data/aoc/2025/02.txt", mode="r").read()

puzzle_example = "11-22,95-115,998-1012,1188511880-1188511890,222220-222224,1698522-1698528,446443-446449,38593856-38593862,565653-565659,824824821-824824827,2121212118-2121212124"

# %%
total = 0

for line in puzzle_input.split(","):
    start, end = line.split("-")
    for i in range(int(start), int(end) + 1):
        s = str(i)
        idx = len(s) // 2
        if len(s) % 2 == 0 and s[:idx] == s[idx:]:
            total += i

print(total)

# %%
total = 0

for line in puzzle_input.split(","):
    start, end = line.split("-")
    for i in range(int(start), int(end) + 1):
        s = str(i)
        if len(s) > 1 and len(set(s)) == 1:
            total += i
        elif len(s) == 4:
            if s[:2] == s[2:]:
                total += i
        elif len(s) == 6:
            if s[:2] == s[2:4] == s[4:]:
                total += i
            elif s[:3] == s[3:]:
                total += i
        elif len(s) == 8:
            if s[:2] == s[2:4] == s[4:6] == s[6:]:
                total += i
            elif s[:4] == s[4:]:
                total += i
        elif len(s) == 9:
            if s[:3] == s[3:6] == s[6:]:
                total += i
        elif len(s) == 10:
            if s[:2] == s[2:4] == s[4:6] == s[6:8] == s[8:]:
                total += i
            elif s[:5] == s[5:]:
                total += i

print(total)

# %% [markdown]
# # Day 3
#

# %%
puzzle_input = open("../data/aoc/2025/03.txt", mode="r").read()

puzzle_example = """987654321111111
811111111111119
234234234234278
818181911112111
"""

# %%
total = 0

for line in puzzle_input.split("\n"):
    if line.strip():
        bank = [int(c) for c in line]
        digit1 = max(bank[:-1])
        idx = bank.index(digit1)
        digit2 = max(bank[idx + 1 :])
        total += digit1 * 10 + digit2

print(total)

# %%
total = 0

for line in puzzle_input.split("\n"):
    if line.strip():
        bank = [int(c) for c in line]
        d = 0
        i = 0
        for _ in range(12):
            j = len(bank) + d - 11
            subbank = bank[i:j]
            digit = max(subbank)
            d += 1
            i += subbank.index(digit) + 1
            total += digit * 10 ** (12 - d)

print(total)

# %% [markdown]
# # Day 4
#

# %%
puzzle_input = open("../data/aoc/2025/04.txt", mode="r").read()

puzzle_example = """..@@.@@@@.
@@@.@.@.@@
@@@@@.@.@@
@.@@@@..@.
@@.@@@@.@@
.@@@@@@@.@
.@.@.@.@@@
@.@@@.@@@@
.@@@@@@@@.
@.@.@@@.@.
"""

# %%
directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

grid = [list(line) for line in puzzle_input.splitlines()]
m, n = len(grid), len(grid[0])

total = 0
for x in range(m):
    for y in range(n):
        if grid[x][y] == ".":
            continue

        r = 0
        for dx, dy in directions:
            if 0 <= (x + dx) < m and 0 <= (y + dy) < n and grid[x + dx][y + dy] != ".":
                r += 1

        if r < 4:
            total += 1

print(total)


# %%
directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

grid = [list(line) for line in puzzle_input.splitlines()]
m, n = len(grid), len(grid[0])


rolls = set()
for x in range(m):
    for y in range(n):
        if grid[x][y] == ".":
            continue

        r = 0
        for dx, dy in directions:
            if 0 <= (x + dx) < m and 0 <= (y + dy) < n and grid[x + dx][y + dy] != ".":
                r += 1

        if r < 4:
            rolls.add((x, y))

        grid[x][y] = str(r)


total = 0
while rolls:
    x, y = rolls.pop()
    grid[x][y] = "."
    total += 1

    for dx, dy in directions:
        if 0 <= (x + dx) < m and 0 <= (y + dy) < n and grid[x + dx][y + dy] != ".":
            r = int(grid[x + dx][y + dy]) - 1
            grid[x + dx][y + dy] = str(r)

            if r < 4:
                rolls.add((x + dx, y + dy))

print(total)

# %% [markdown]
# # Day 5
#

# %%
puzzle_input = open("../data/aoc/2025/05.txt", mode="r").read()

puzzle_example = """3-5
10-14
16-20
12-18

1
5
8
11
17
32"""

# %%
intervals, ids = puzzle_input.split("\n\n")

intervals = [tuple(map(int, interval.split("-"))) for interval in intervals.split("\n")]
intervals = sorted(intervals, key=lambda x: x[0])

ids = [int(a) for a in ids.split("\n")]


fresh = 0
for id in ids:
    for interval in intervals:
        if interval[0] <= id <= interval[1]:
            fresh += 1
            break

print(fresh)

# %%
intervals, ids = puzzle_input.split("\n\n")

intervals = [tuple(map(int, a.split("-"))) for a in intervals.split("\n")]
intervals = sorted(intervals, key=lambda x: x[0])

merged = []
for interval in intervals:
    if not merged or merged[-1][1] < interval[0]:
        merged.append(interval)
    else:
        merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

fresh = 0
for start, end in merged:
    fresh += end - start + 1

print(fresh)

# %% [markdown]
#

# %% [markdown]
# # Day 6
#

# %%
puzzle_input = open("../data/aoc/2025/06.txt", mode="r").read()

puzzle_example = """123 328  51 64 
 45 64  387 23 
  6 98  215 314
*   +   *   +  """

# %%
operators = {"+": operator.add, "*": operator.mul}

*lines, ops = [line.split() for line in puzzle_input.split("\n")]

total = 0
for i in range(len(ops)):
    op = operators[ops[i]]
    nums = map(int, [line[i] for line in lines])
    total += functools.reduce(op, nums)

print(total)

# %%
operators = {"+": operator.add, "*": operator.mul}

*_, ops = [line.split() for line in puzzle_input.split("\n")]
*lines, _ = [line for line in puzzle_input.split("\n")]

total = 0
nums = []
op_idx = 0
for i in range(len(lines[0])):
    digits = "".join([line[i] for line in lines])

    if digits.strip():  # zero columns
        nums.append(int(digits))
    else:
        total += functools.reduce(operators[ops[op_idx]], nums)
        nums = []
        op_idx += 1

total += functools.reduce(operators[ops[op_idx]], nums)

print(total)

# %% [markdown]
# # Day 7
#

# %%
puzzle_input = open("../data/aoc/2025/07.txt", mode="r").read()

puzzle_example = """.......S.......
...............
.......^.......
...............
......^.^......
...............
.....^.^.^.....
...............
....^.^...^....
...............
...^.^...^.^...
...............
..^...^.....^..
...............
.^.^.^.^.^...^.
..............."""

# %%
lines = [list(line) for line in puzzle_input.split("\n")]
m, n = len(lines), len(lines[0])

splits = 0
for r in range(m - 1):
    l1 = lines[r]
    l2 = lines[r + 1]
    for c in range(n):
        if l2[c] == "." and l1[c] in ["S", "|"]:
            l2[c] = "|"
        elif l2[c] == "^" and l1[c] in ["S", "|"]:
            l2[c - 1] = "|"
            l2[c + 1] = "|"
            splits += 1

print(splits)


# %%
lines = [list(line) for line in puzzle_input.split("\n")]
m, n = len(lines), len(lines[0])

test = [0] * n
test[lines[0].index("S")] = 1

for r in range(m - 1):
    l1 = lines[r]
    l2 = lines[r + 1]
    for c in range(n):
        if l2[c] == "." and l1[c] in ["S", "|"]:
            l2[c] = "|"
        elif l2[c] == "^" and l1[c] in ["S", "|"]:
            l2[c - 1] = "|"
            l2[c + 1] = "|"
            test[c - 1] += test[c]
            test[c + 1] += test[c]
            test[c] = 0

print(sum(test))

# %% [markdown]
# # Day 8
#

# %%
puzzle_input = open("../data/aoc/2025/08.txt", mode="r").read()

puzzle_example = """162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689"""


# %%
class UnionFind:
    def __init__(self, size: int):
        self.count = size
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY
            self.count -= 1
            self.size[rootY] += self.size[rootX]
            self.size[rootX] = 0

    def connected(self, x: int, y: int) -> bool:
        rootX = self.find(x)
        rootY = self.find(y)
        return rootX == rootY


# %%
points = [tuple(map(int, point.split(","))) for point in puzzle_input.split("\n")]
ids = {p: i for i, p in enumerate(points)}

pairs = []
for p1, p2 in itertools.combinations(points, 2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
    pairs.append((distance, (p1, p2)))

pairs.sort()

uf = UnionFind(len(ids))
for _, (p1, p2) in pairs[:1000]:
    uf.union(ids[p1], ids[p2])

print(functools.reduce(operator.mul, sorted(uf.size)[-3:]))

# %%
points = [tuple(map(int, point.split(","))) for point in puzzle_input.split("\n")]
ids = {p: i for i, p in enumerate(points)}

pairs = []
for p1, p2 in itertools.combinations(points, 2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
    pairs.append((distance, (p1, p2)))

pairs.sort()

uf = UnionFind(len(ids))
for _, (p1, p2) in pairs:
    uf.union(ids[p1], ids[p2])
    if max(uf.size) == sum(uf.size):
        print(p1[0] * p2[0])
        break

# %% [markdown]
# # Day 9
#

# %%
puzzle_input = open("../data/aoc/2025/09.txt", mode="r").read()

puzzle_example = """7,1
11,1
11,7
9,7
9,5
2,5
2,3
7,3"""

# %%
points = [tuple(map(int, line.split(","))) for line in puzzle_input.split("\n")]

max_area = 0
for (x1, y1), (x2, y2) in itertools.combinations(points, 2):
    area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
    if area > max_area:
        max_area = area

print(max_area)


# %%
def point_in_polygon(px, py, line_segments):
    crossings = 0

    for (x1, y1), (x2, y2) in line_segments:
        if x1 == x2:  # vertical
            if y1 <= py <= y2 and px == x1:
                return True  # on vertical edge

            # Ray-cast: does horizontal ray to the right from (px, py)
            # intersect this vertical segment?
            if y1 <= py < y2 and x1 > px:
                crossings += 1

        elif y1 == y2:  # horizontal
            if x1 <= px <= x2 and py == y1:
                return True  # on horizontal edge

            # Horizontal edges don't affect crossings
            continue

    return crossings % 2 == 1


def intersects(ls1, ls2):
    p1, p2 = ls1
    p3, p4 = ls2

    # Both vertical
    if p1[0] == p2[0] and p3[0] == p4[0]:
        return False

    # Both horizontal
    if p1[1] == p2[1] and p3[1] == p4[1]:
        return False

    if p1[0] == p2[0]:  # ls1 vertical
        if p3[1] == p4[1]:  # ls2 horizontal
            return p1[1] < p3[1] < p2[1] and p3[0] < p1[0] < p4[0]

    if p3[0] == p4[0]:  # ls2 vertical
        if p1[1] == p2[1]:  # ls1 horizontal
            return p3[0] < p1[0] < p4[0] and p3[1] < p1[1] < p4[1]

    return False


points = [tuple(map(int, line.split(","))) for line in puzzle_input.split("\n")]

line_segments = list(sorted(ls) for ls in itertools.pairwise(points + [points[0]]))


max_area = 0
for (x1, y1), (x2, y2) in itertools.combinations(points, 2):
    area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
    if area > max_area:
        # Check if rectangle sides intersect any line segments
        rect_sides = [
            tuple(sorted([(x1, y1), (x2, y1)])),
            tuple(sorted([(x1, y1), (x1, y2)])),
            tuple(sorted([(x2, y1), (x2, y2)])),
            tuple(sorted([(x1, y2), (x2, y2)])),
        ]

        intersection = any(
            intersects(ls1, ls2) for ls1 in rect_sides for ls2 in line_segments
        )

        if intersection:
            continue

        # Check if rectangle corners are inside the polygon
        rect_corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        if not all(point_in_polygon(px, py, line_segments) for px, py in rect_corners):
            continue

        max_area = area

print(max_area)

