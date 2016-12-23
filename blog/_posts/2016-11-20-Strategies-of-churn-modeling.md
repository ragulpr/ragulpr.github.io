---
layout: post
title: (1/3) Churn-modeling strategies
---

So you've want to build a predictive churn-model. This series of post will walk you through the process. 

It's funny with great articles like [this](https://engineering.shopify.com/17488468-defining-churn-rate-no-really-this-actually-requires-an-entire-blog-post), where the author is almost ashamed by the fact that defining churn-rate isn't an obvious task. I would go further than that. 

As soon as we don't have a simple subscription-based business I would claim that the hardest part is to get a reasonable definition of churn in the first place. This is the focus of this post. 

##Why it's hard
At a first glance this might seem like a no-brainer and your initial googling will strengthen this faulty notion. Then you start working with it and you realize that churn modeling might be mankinds ultimate intellectual challenge. To overcome it you will have to critique your own and others initial intuition, dissect language to uncover impractical linguistical temporal assumptions and engrained circular reasoning that impeds your understanding and eventually get yourself together, be an engineer by drawing some arbitrary line in the sand to decide a target value and package it nicely so that your models output is useful and adequately corresponds to the (possibly faulty) intuition of those who will consume your model. 

Before we start thinking about it we imagine that the problem looks like this:

###Fantasy
Customer $$n$$ has a feature vector $$x^n$$ and a binary target value $$y^n$$:

$$
y^n=
\begin{cases}
1\text{ if customer } n\text{ will churn}\\
0\text{ if customer } n\text{ won't churn}
\end{cases}
$$

The machine learning model we want to build then uses features $$x^n$$ for customer $$n$$ to estimate the probability of churning, i.e the churn score™:

$$
\hat{\theta}^n = \hat{\Pr}(Y^n = 1) = f(x^n)
$$

The model is defined by its loss function, the bernoulli log-loss (cross-entropy):

$$
\text{minimize} \sum_n \log\left(\hat{\theta}^{y^n}\cdot(1-\hat{\theta})^{1-y^n}\right)
$$

The companys current churnrate is separately calculated as some [aggregate measure](https://engineering.shopify.com/17488468-defining-churn-rate-no-really-this-actually-requires-an-entire-blog-post) of churners and has little to do with this model.

###Reality

You are very lucky if you have a business model that's easy enough to fit into the above model. IOne could argue that *churn* is only a viable concept in the most well-defined subscription-based service if then. Typically

* churn itself is undefined. 
	- Think regular e-commerce. How long without repeat purchase before being considered *churned*? Your initution tells you *forever* but you soon realize that it takes forever to observe that that was the case. The *will* and *won't* doesn't make practical sense.
	- Even with a subscription based business model resurrection might happen if they renew their plan. Subscribed but not paying? Paying but not active? Even here the binary states might not be carved in stone. 
* We have *time series* for each customer. 
	- It will probably require extensive feature-engineering to get flat dataset of customers $$\times$$ features and we might not even want this datastructure in the first place.

What it almost always ends up is that we make up some quite arbitrary definition of what constitutes a churned customer. Losely posed as an optimization problem we want to choose a definition that:

* Minimize probability of resurrection 
	- Permanence. Stakeholders will assume that a churned customer is permanent and/or won't bring in any future value.
* Maximize the probability of detection
	- Measurability. Your churn-definition is useless if the reality corresponding to your churn-definition is neither measurable or predictable. Note that this depends on your downstream ML-modeling strategy.
* Maximize interpretability of your definition
	- Interpretability. A good & safe churn-definition corresponds to what people assume that it means. A bug in natural language is that you can't change this assumption so if you can't model this concept, **don't call it churn**. Call it something else. [Confusion can be expensive](http://www.globenewswire.com/news-release/2004/08/10/314172/62086/en/Investor-Notice-Murray-Frank-Sailer-LLP-Announces-Shareholder-Lawsuit-Against-Netflix-Inc-NFLX.html)

As we'll see, the churn-definition might influence your whole ETL-pipeline for the model. As you need that pipeline to start datamunging to figure out a proper definition this might lead to a whole lot of expensive and timeconsuming iterative work. There's some modeling-tricks that might help you speed up the process.

## Think active/inactive before churned/not churned  
We need to squeeze the highly asymmetric reality into the very square churn-concept. A mental model that I've found useful is to break the problem into pieces and initially drop the idea of being *churned* or not (don't worry, we'll get there) and initially focus on what a customer needs to do in order to be defined as *active* or *inactive* or some similar less permanent state. This is more easily communicated and might even interface with your KPI's like DAUs and MAUs.

Your group of datascientist almost surely will have an informal definition of what customers being *active* entails otherwise this is a good talking point to get different stakeholders to sit down and define it, leaving you to sneak out and use its negation - being *inactive* - and later *very inactive* for your churn-definition. 


## Separate *event*, *status* and *state*  

| **Event**  | *Purchased today?* 					| 
| **Status** | *Purchased in past 30 days?* 		|  Time since event $$ \leq 30$$
| **State**  | *In a period of 30 days with a purchase?*	| (time since event $$\geq 30/2$$) XOR (time to event $$\leq 30/2$$) 

## Going from event to state
This is the preferred method. If *purchases* are the thing that we want customers to do everything else follows. A state (active/inactive) is defined by setting some arbitrary threshold $$\tau$$, say no purchases for 30 days. *Status* is whether a customer has made a purchase in the last 30 days and *state* is if it'll be at least 30 days gap between purchases. 

state = active/inactive

status = observed active/inactive

event = d/dt status

churned = state or status of long inactiveness

resurrected = came back after being churned

Note that by you can only know the *state* **between** events while status and events are known by definition. Your definition of what being churn*ed* entails is a business-specific decision disconnected from the future modeling-choices. What is the time of death of Schrödingers cat? 
  
Taking this perspective will also open the golden gates to a loosely defined but really interested ML-field. Welcome to the world of time to event prediction!

![](/assets/intro_event_feature_pairs_v2.gif					){:class="img-responsive"}

Minimize resurrection while maximizing available data through minimizing $$\tau$$ while also maximizing the interpratibility of the model. 


As with many problems, the less you think about the problem the easier it becomes. In the beginning 

'Event' -> 'thresholding on time' -> Binary model -> prediction
							      -> Reporting
'Event' -> Model -> Threshold -> Prediction/reporting


* Managers dream

* Real problem : timelines of events
 * Discretize to binary target
 * Time to event
 * Konstiga metoder

* Binary target : Sliding box

* Time to event solutions
* Flowchart
* You can read out all the nice metrics: active, inactive, dormant, churned

Takeaway should be that you should build a what you thought was a churn model but only call it a churn model if you can actually model churn. If you can't, call it something else. 
