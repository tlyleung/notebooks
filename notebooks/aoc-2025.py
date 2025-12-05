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
