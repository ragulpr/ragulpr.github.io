---
layout: post
title: Continuous & Record based WTTE
---
The WTTE-RNN seem to have become popular.

It has some drawbacks. As I formulated it initially it assumes that each day for a customer is an input datapoint leading to *one* prediction valid that day. This is not really how we see our data or how we would think of the problem. This assumes batch-processing all sequences each day, even those unchanged. This is hacky. 

In the best of world we would have do actual real-time prediction for the state of each user, by updating their state whenever they do some action. But wait, we live in the best of worlds and this is actually possible! 

I have some possibly -less hacky- solution for this that I was thinking about previously but I thought it'd be too much to swallow in one sitting and I didn't have the pipeline in place to evaluate it properly. 

Here goes. For now, forget about timelines and recurrent event and imagine that $Y$ is a random waiting time

## Focus on the objective

You want to predict whenever you want to, i.e you want to predict continuously. Your loss should always look like your prediction. 

### Recap: 
#### Less hacky PDF
A continuous positive distribution can always be written in terms of its cumulative hazard function. Check my thesis if you don't believe me.

f(y) = \Lambda'(y) e^{-\Lambda(y)} = \lambda(y) e^{-\Lambda(y)}  = \lambda e^{-\Lambda}

#### Less hacky loss
loglikelihood= -loss = u*log(\lambda)-\Lambda

#### Conditional excess distribution
Pr(Y=t+s|y>=t) = \frac{f(t+s)}{S(t)}=\lambda(t+s)e^{-[\Lambda(t+s)-\Lambda(t)]} =\tilde{\lambda} e^{-\tilde{\Lambda}}

#### Conditional excess loss
"Total loss if we were to get a param at t=0 and use it for prediction with the CED until the actual time t=y"
≈ sum of error at t=0,0.0001, 0.0002,... 
= integrate loss over t [0,y]
