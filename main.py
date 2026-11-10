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

## There's a number of ways to determine if a user liked a song or not. For example, we could say that if a user rated a song 7 or higher, they liked it. If they rated it 6 or lower, they did not like it. This integer is arbitrary, and as stated in the outline, we assume there is some integer boundary that minimizes the number of misclassifications.
## Alternatively, we could say that if a user repeated a song, they liked it. If they did not repeat a song, they did not like it.
## We could even say that if a user listened up to a certain number of seconds of a song, they liked it. If they listened to less than that number of seconds, they did not like it.
## The constraints of this project are the sample dataset (or lack thereof). Not every application notes all of the data above. As far as I've found, those who may have do not make it easy to access. So, we will likely have to generate our own dataset.


users = 30  # elements of the matrix, or n
songs = 5  # features of the matrix, or d

# TODO: randomize the ratings of the songs first with considerable variance (some songs are widely agreed upon to better than others, such as Habits by osamason). From here, randomize the ratings by-user.
# Generate an n×d matrix from standard normal distribution (mean=0, std=1)
allRatings = torch.randint(low=1, high=11, size=(users, songs))# dataset of n integer ratings of d songs on a scale of 1-10 
theta = torch.rand(low=-0.5,high=0.5, size=(songs, songs))  # random theta for each rating betweeen -0.5 and 0.5, shape (d,d)
b = torch.rand(users, songs)  # random bias for each user

# sanity check
print("allRatings shape:", tuple(allRatings.shape))
print("theta shape:", tuple(theta.shape))
print("b shape:", tuple(b.shape))

y_candidate = allRatings @ theta + b  # compute the linear combination of ratings and theta, plus bias
print("predictions shape:", tuple(y_candidate.shape))

# TODO: this is copied from MATH300's Homework 2 assignment; implement it
def gradient_of_mean_loss(X, y, theta):
    p = torch.sigmoid(X @ theta)
    gradient = X.T @ (p - y) / len(y)  # 2a: gradient of the mean cross-entropy over the n rows, from X, p, and y; shape (3,)
    return gradient


# TODO: replace stand-ins (features, is_bad, theta) with actual data
for step in range(2000):
    theta = theta - 0.5 * gradient_of_mean_loss(allRatings, None, theta)





## This segment computes the probability that a user will like a song, given that similar users liked it.

# this doesn't need to be defined as a function, but it is for the sake of organization at the moment.
def getMeanSongRating(allRatings,i):
    return torch.mean(allRatings[:,i])

allRatings @ theta

## List of sites I've tried to find adequate data from: MusicBrainz, AirBuds, RateYourMusic, Spotify
## Sources I've yet to try: Last.fm