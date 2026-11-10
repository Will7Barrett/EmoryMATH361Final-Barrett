
## There are X primary functions to this code block. They are as follows:


## Take music rating histories from n users, 1j, 2j, ..., nj. Each user will represent one row.

## Each row will contain d ratings (The ratings will likely be on an integer scale). Our goal is
## to build a classifier that determines if a user liked a song or not. 

## The data for this project will most likely be randomly generated, although if I can
## find a real dataset for this issue, it would be fantastic. At the very least, we can sample
## this in small-scale by taking anonymous music ratings from the four members of our group
## on at least four randomly-chosen songs. 

import math
import torch
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=4, sci_mode=False)

n = 5  # size of the matrix
# Generate an n×n matrix from standard normal distribution (mean=0, std=1)
matrix = torch.randn(n, n)

print(matrix)