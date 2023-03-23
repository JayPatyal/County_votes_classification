## (Un)Supervised Learning

Using the provided dataset, implementing a multi-layer perceptron for classification via supervised learning, as well as the unsupervised k-means and AGNES clustering algorithms.

## Data

The attached csv file contains all the data. The run file handles importing it and converting it to numpy arrays. A description of the dataset is given below.

## Algorithms

### Multi-Layer Perceptron (MLP)

Much of the MLP is already implemented for you. Please look through the code and try to understand it. What happens if you comment out the line that shuffles the dataset before training? The MLP class calls a fully connected layer ("FCLayer") and a Sigmoid layer. You need to implement the functions implementing the forward and backward passes for each. The forward pass is for prediction and the backward pass is for doing gradient descent. The backward-pass function takes the previous gradient as input, updates the layer weights (for FCLayer) and returns the gradients for the next layer. 

### K-Means:

Use Euclidean distance between the features. Use a maximum number of iterations, t. Choose a k value and use k-means to split data in k clusters. The k value is provided to the k_means class. Please implement the train method, which should return an n-element array (with n the number of data points in X) with the cluster id corresponding to each item.
The distance function is provided for you and you can assume all data is continuous. Ties can be broken arbitrarily.

### AGNES:

Using the Single-Link method (distance between cluster a and b=distance between closest members of clusters a and b) and the dissimilarity matrix.
Using k as the number of clusters. Stop when number of clusters == k. The k value is provided to the AGNES class.

#==========================================================Data==========================================================
# Number of Instances:	
# 653
# Number of Attributes:
# 35 numeric, predictive attributes and the class

# Attribute Information:

# We have 35 variables for 653 counties, including demographics, covid info, previous election 
# results, work related information.
# percentage16_Donald_Trump	
# percentage16_Hillary_Clinton	
# total_votes20	
# latitude	
# longitude	
# Covid Cases/Pop	
# Covid Deads/Cases	
# TotalPop	
# Women/Men
# Hispanic
# White	
# Black	
# Native	
# Asian	
# Pacific	
# VotingAgeCitizen	
# Income	
# ChildPoverty	
# Professional	
# Service	
# Office	
# Construction	
# Production	
# Drive	
# Carpool	
# Transit	
# Walk	
# OtherTransp	
# WorkAtHome	
# MeanCommute	
# Employed	
# PrivateWork	
# SelfEmployed	
# FamilyWork	
# Unemployment


# Class Distribution:
# 328 - Candidate A (1), 325 - Candidate B (0)
#========================================================================================================================