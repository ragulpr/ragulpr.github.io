---
layout: post
title: (2/3) Machine learning for churn prediction
---
In the last post we discussed how to get a churn-definition in place. In this post we'll discuss different  machine-learning modeling choices and their implications. In the next post we will argue for the best, optimal and genial method that I happened to propose in my master-thesis. 

As we argued in the previous post, you should now have a data-structure s.t for each customer you have

$$
\begin{align*}
y_t &= \text{(possibly censored) time to event}\\
u_t &= \begin{cases}
			1\text{ if timestep }t \text{ uncensored}\\
			0\text{ if timestep }t \text{ censored}  
		\end{cases}\\
x_t &= \text{ feature vector}
\end{align*}
$$

## ML recap (focus on the objective)
Remember, an ML-model consists of

1. Objective function 
2. Link function connecting features with target value
3. (learning) tactic for finding the link function

When modeling something you can almost always *focus on the objective* function. This function will tell you all you need to know about how the model will reason. A probabilistic objective function *tells a story* about how errors are weighted.

There's methods like deep learning that you can use to automagically set up a way to connect features to the target value and learn the weights of your network using gradient descent. Models like Random Forests or Gradient boosting handles step 2-3 but to understand the model you need to know 1. SVMs defines step 1 and 3 and is opinionated on the structure of step 2. 

## Binary target: Sliding box model
* Flexible
	* Any machine learning model
* Simple
	* Powerpoint-potential

### Implementations

### Problem

* Lost data
* Hard coded inference question

* Awful iterative modeling-loop
	* Simplicity enters in a late stage
* Sparsity?

## Learn-to-rank methods

### Implementations

### Problem

* Computationally expensive
* Kind of new & novel

## Survival methods

### Implementations

### Problem

* Statisticians don't like generalities
	* Echo-chamber documentation
	* No brutality
* Few models

All of these can later be subdivided into *temporal models* and *static* models. Before you go into the feature-engineering-loop think of what your initial purpose is.  Is it prediction? Is it to see  In the next article we wi

(Figure x3 )
These are all measured on my own subjective scale. We could add another axis and mash them all together, that of how hard it is. 


Failure time prediction done right
As we concluded in previous post we produced 
