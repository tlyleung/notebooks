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
