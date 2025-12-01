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

