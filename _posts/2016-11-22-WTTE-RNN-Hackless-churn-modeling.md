---
layout: post
title: WTTE-RNN - Less hacky churn prediction
---
(How to model and predict churn using deep learning)

Churn prediction is one of the most common machine-learning problems in industry. The task is to predict whether customers are about to leave, i.e *churn*. You can't imagine how many complex and hacky ways there are to do this. The topic of this post is how to avoid these pitfalls by choosing a smart modeling strategy and by using what I think is a pretty neat machine-learning model, what I call the WTTE-RNN. 

The trick is really to define the problem in a way that makes the solution obvious. In doing so it becomes identical to that of trying to predict when patients will die, when machines will fail or when an earthquake will hit. It turns out that the model that I'll tell you about might be a good solution for those problems too. And yes, of course it involves deep learning but I won't talk much about it here. The focus is on the objective (function) that you train your neural networks with. 

At the end of this article I hope that you've gotten familiar with at least some of the words from the title of my thesis (link available soon).

> *WTTE-RNN : Weibull Time To Event Recurrent Neural Network*
>
> *A model for sequential prediction of time-to-event in the case of discrete or continuous censored data, recurrent events or time-varying covariates*


Table of contents:

* TOC
{:toc}

## Churn prediction is hard
There's been a couple of great [articles](https://engineering.shopify.com/17488468-defining-churn-rate-no-really-this-actually-requires-an-entire-blog-post) where the author is almost ashamed to admit how hard it is to define an aggregate churn metric. I think they could have gone further than that. Even here they assumed that the definition of a churned customer is written in stone which is typically not the case. You only need to google 'churn prediction' to realize that a bunch of stakeholders have this scary idealized view of the problem:

> Customer $$n$$ has a feature vector $$x^n$$ and a binary target value $$y^n$$:
>
> $$
y^n=
\begin{cases}
1\text{ if customer } n\text{ will churn}\\
0\text{ if customer } n\text{ won't churn}
\end{cases}
$$
> 
> The *churn-rate* is the mean change in the number of churned customers. 
> The machine learning model we want to build uses features $$x^n$$ for customer $$n$$ to estimate the probability of churning, i.e the churn score™. 

Reality rarely fits into this box. Even though we often know a churned customer when we see them,  operationalizing this fuzzy concept can be hard. The problem stems from some type of linguistical bug - our intuition around churn seems to be riddled with hidden temporal assumptions and engrained circular reasoning. As you dig into the details you discover an abyss of vagueness.

* What does ***will*** mean? We all 'churn' at some point :( 
* What does ***a customer*** mean? A customer at a given point in time? A subscription plan? A non-'churned' period of a given customer-id?
* What's the shape of the feature vector? Fixed width with static features or aggregated over time implying different measurement errors? Isn't it really time series for each customer?
* What does ***churn*** mean? Million dollar question 
* You probably can't know the current churn rate, you need your churn-model to predict/estimate it. 

After realizing this and snapping out of the dream, data scientists will often end up defining it by drawing some arbitrary line in the sand like 'no purchase for 30 days'. This is nothing to be ashamed of and there are a few tricks that helps you speed up this modeling process. The first is to frame the problem in a good way. 

### Churn prediction = non-event prediction
Don't predict churn, predict non-churn. My philosophy is losely that *if something happens in the future* that can be used to define the customer as non-churned we can define this *something* as an *event*. If there's an event happening in the future we can define the *time* to that event from any prior point in time. If a customer has a longer time to a future event that customer is *more churned*. 

The raw data that we have to work with are a series of records for each customer. You can think of each customer as a timeline starting from when we first saw them until today. We can stack these timelines on top of eachother to get some overview on who had data when:

![stacked_timelines.png](http://i.imgur.com/JT60TnL.png)

Divide your dataset so that you have the *events* (like purchases or logins) that matters for your churn definition and *features* (clicks, purchases, logins) that can be used to predict them. 
 
![intro_event_feature_pairs_v2.gif](http://i.imgur.com/pdpFqb8.gif)

We want to use these features to sequentially predict the future using historic data:
![intro_sequential_prediction.gif](http://i.imgur.com/fD9iQJd.gif)

The next trick is to define **what** you want to predict which can be done in multiple ways. I think the most natural thing is to predict the **time to the next event** $$y_t$$ at each timestep $$t$$. I will call this **TTE** for short. We can visualize this as a kind of sawtooth-wave:

![intro_tte_nofuzz.gif](http://i.imgur.com/CjQoNFg.gif)
If some user has a longer time to the next purchase it's reasonable to say that they are *more churned*. If the user never purchases anything again $$y_t=\infty$$. Problem is that we need to wait infinitely long to know that this was the case. This leads us to the fundamental problem of this type of data: **censoring**.

### Censored data
We don't know how old we'll be but we know that we'll get older than our current age. This is an example of a **censored datapoint**. If you haven't heard about this useful concept before it's probably because the frequentist statisticians that work on it only likes to explain it using particular jargon in impenetrable 1000-page books on how it can be applied using a particular model to a particular dataset for some particular type of cancer. I'll try to be more general.

In our world we only have event-data from the **observed past** i.e from when we first saw the customer up until now. This means that we don't know what the actual time to the next (unseen) event was after the last seen event. This gives us a lower bound that we can use for training, $$\hat{y}_t\leq y_t$$. This partial observation is called **right censored data** and is shown in dotted red:

![intro_tte.gif](http://i.imgur.com/hkpM09J.gif )

A censored observation $$\tilde{y}_t$$ is interpreted as *"at time $$t$$ there was at least $$\hat{y}_t$$ timesteps until an event"*. How can we use this data for training models? 

## Models for censored data
The hacky way that 99.9% of all churn-models use is to do a binary workaround using fixed windows. I call this the **sliding box model**:

### Sliding box model
Instead of trying to predict the TTE directly we predict whether an event happened within a preset timeframe $$\tau$$. We define the observed target value $$b_t$$ at timestep $$t$$ as

$$
	b_t = 
	\begin{cases}
		1& \text{if event in }    &[t,t+\tau) \\ 
		0& \text{if no event in } &[t,t+\tau) \\ 
		\text{unknown}& \text{else}&
	\end{cases}
$$

This can be seen as sliding a box in front of you and see if it covers any events:
![previous_work_sliding_box_gendata_tau_2.gif](http://i.imgur.com/osjA1uZ.gif)
Here the unknowns/$$NA$$'s appear in the last $$\tau$$ steps of the observations when there's no events (shown as blank in plot).

To construct a probabilistic objective function, think of the $$b_t$$'s as independently drawn from a Bernoulli distribution that has a time varying parameter $$\theta_t$$ denoting the probability of event within $$\tau$$ time from timestep $$t$$:

$$
B_t \sim \text{Bernoulli}(\theta_t)
$$

$$
\Pr(B_t=b_t) = \theta_t^{b_t}\cdot(1-\theta_t)^{1-b_t}
$$

Set $$\theta_t$$ as some machine learning model $$g$$ taking data $$x_{0:t}$$ available at time $$\theta_t=g(x_{0:t})$$. The objective function then becomes:

$$
\begin{align*}
	\underset{\substack{g}}{\text{maximize}}\, 
\log\left(\mathfrak{L}(\theta_t)\right) : = \log\left(\theta_t^{b_t}\cdot(1-\theta_t)^{1-b_t}
\right)
\end{align*}
$$

Predicting with the model can be visualized as letting the box-height be the predicted probability $$\theta_t$$:

![previous_work_sliding_box_pred_tau_2.gif](http://i.imgur.com/eUlGAVZ.gif)

#### Use as a churn-model
There's some obvious benefits of this model. 

* Simplicity and explicity. It's pretty easy to explain how it works. 
* Flexibility. We can use any binary prediction algorithm. Think Xgboost or Random Forests or char-level RNNs.

If you've built one you also know the downsides: 

* Your predictions are not very informative.

Is a churn definition saying '30 days without event' really nuanced enough to be actionable? Does it fit all your customers? Instead of binary you could predict multiple timeintervals ahead but leads to more hacking and hyperparameters.
In any case you probably want $$\tau$$ to be as big as possible since you want to predict the *lack* of events (churn) in a foreseable future. Problem with that is that:

* We can't use the last $$\tau$$ timesteps for training

We can observe $$b_t=1$$ here if there are events but we can't exclude that there's no event just beyond the boundary. Unless we explicitly model this dropout we don't know what sorts of biases and class-imbalances we introduce by using the positive observations. It's therefore safest to drop all observations happening in the last $$\tau$$ steps. This means that the higher $$\tau$$ is, the less recent data we have to train on. 
Raising $$\tau$$ also induce a kind of sparsity such as all being 1's:

![previous_work_sliding_box_gendata_tau_4.gif](http://i.imgur.com/vDgwNNo.gif)

Giving very blunt signals for your model to train on. 

So if $$\tau$$ is too small the model output becomes meaningless. If $$\tau$$ is too big we can't train it. In summary, the sliding box model is

* Hackish and horrible to work with

The parameter $$\tau$$ rigidly defines everything about your model. Changing $$\tau$$ changes your data pipeline. It also changes the meaning of your predicted output and the model performance. Striking a balance is an awful and time-consuming iterative modeling loop.

In some settings binary targets/$$\tau$$'s are more easily defined and this model is great in certain applications. [Moz developer blog](https://moz.com/devblog/deep-learning-for-customer-churn-prediction/) wrote a nice piece about how they successfully used RNNs in what looks like a similar framework. 

### Making it a *learning to rank*-problem.

The usual application for a churn-model is to use the score to rank customers according to their riskset. Ranking is its own [machine learning topic](https://en.wikipedia.org/wiki/Learning_to_rank), and we can incorporate censored data in these models. If we know that there was at least say $$7$$ days until an event, we can compare this to when we know that there was $$3$$ days to an event since $$3<7$$. Noticing that *ranking* is really defined by all such pairwise comparisons we can formulate a binary target: 

$$
r_{ij} = 
	\begin{cases}
		1&  \text{If }\hat{y}_i \leq \hat{y}_j \text{ and obs }i\text{ not censored}\\ 
		0&  \text{If }\hat{y}_i > \hat{y}_j    \text{ and both not censored} \\
		\text{unknown}& \text{Else}
	\end{cases}
$$

The machine-learning model $$g$$ is then to use features from observation $$i$$ and $$j$$ to predict the ordering:

$$
\hat{r}_{ij} = g(x_{0:i},x_{0:j})
$$

If we don't have any censored data and fix the target to be discrete $$\hat{r}_{ij} \in \{0,1\}$$ and minimize $$\|\hat{r}_{ij}-r_{ij}\|$$ one can show that this is identical to optimizing the [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve). (I'm sure weaker assumptions exists) I haven't thought of any probabilistic models for this but if we relax it to continuously be $$\hat{r}_{ij} \in [0,1]$$ we could get some results from any binary prediction algorithm.

As the dataset are the pairwise combinations of all the observations the training dataset hence training time grows quadratically. We also have the problem of not actually being able to say whether an individual customer is predicted as churned or not, only whether they are predicted as more churned than someone else.

There's some really cool current research on learning to rank and optimization for the AUC that deserves its own blogpost. The fact that this is a cool and fairly unknown field is a problem in terms of lack of implementations & documentation. 

## Objective : What we want in a churn-model

### A good churn-definition is predictable

When drawing this line it's good to keep a few things in mind. The objective when defining 'churn' is to:

* Minimize probability of resurrection 
	- Permanence. Stakeholders will assume that a churned customer is in a permanent state and/or won't bring in any future value.
* Maximize the probability of detection
	- Measurability. Your churn-definition is useless if the reality corresponding to your churn-definition is neither measurable or predictable. Note that this is linked to your downstream ML-modeling strategy.
* Maximize interpretability of your definition
	- A good & safe churn-definition corresponds to what people assume that it means. A bug in natural language is that you can't change this assumption so if you can't model this concept, **don't call it churn**. Call it something else like *active* vs *inactive* customers. 

I think the last point is the most important one. One can argue that 'churn'-modeling is something you should be doing only if you're in a clear-cut subscription based service. Even then it might be hard. The Netflix-class action lawsuit, where shareholders felt mislead over reported churn-rates, showed that [all stakeholders seems to be confused](http://www.globenewswire.com/news-release/2004/08/10/314172/62086/en/Investor-Notice-Murray-Frank-Sailer-LLP-Announces-Shareholder-Lawsuit-Against-Netflix-Inc-NFLX.html) about how to define churn. The court even dropped the case concluding that there's no official definition of it. 

To do this you'll probably end up with a shoppinglist like this one

### A good churn-model is flexible
As we've seen the big bottleneck in churn-modeling is feature-engineering and a cumbersome modeling loop, the final dream-model looks something like this:

1. For recurrent events
2. Can handle time varying covariates 
3. Can learn temporal patterns
4. Handle sequences of varying length 
5. Learn with censored data
6. Flexible predictions 

We can cross out 1-4 by using RNNs as the machine-learning algorithm. In short, Neural Networks feels less hacky than many other models as you can spend less time engineering features. Recurrent neural networks for time series prediction are less hacky than non-temporal alternatives because you don't have to hand-engineer features using windowfunctions such as 'mean number of purchases last x days'. 
Regression-puritans will hang me for this, but **neural networks are not black box models**. If you want insights analyze the patterns your network learned, not the other way around. 

The final points, 5-6 can be crossed of by choosing a smart objective function. Meet the WTTE-RNN:

## WTTE-RNN 
The recipe for this model is embarassingly simple. In the most general framework you want to:

1. Assume that the time to event $$Y_t$$ in each step follows some distribution governed by some parameter set $$\theta_t$$ s.t $$Pr(Y_t\leq y_t\|\theta_t)$$
2. Let $$\theta_t$$ be the output of some machine learning model (like an RNN) taking feature history at timestep $$t$$ as input, $$\theta_t = g(x_{0:t})$$ 
3. Train the machine learning model using a special log-likelihood loss for censored data.
4. In each step you can now predict the *distribution* over the time to the next event

Any [survival regression model](https://en.wikipedia.org/wiki/Survival_analysis#Cox_proportional_hazards_.28PH.29_regression_analysis) using time-varying feature data can be said to fall in this category.

What I call the WTTE-RNN is when we 

* Assume $$Y\sim$$Weibull with parameters $$\alpha_t$$ and $$\beta_t$$
* Let $$\theta_t=\begin{pmatrix}\alpha_t\\\beta_t\end{pmatrix}=g(x_{0:t})$$ to be the output of an RNN

The sequential prediction over the timeline can then be visualized as:
![solution_beta_2.gif](http://i.imgur.com/BUIvufH.gif)
(Here only varying $$\alpha_t$$ but you get the picture)

During training we push the pdf up around known TTE for uncensored data and minimize the area under the PDF until the censoring point for censored data. More on this later. The point is that in order to learn from what we didn't see, we make assumptions on how the TTE behaves beyond the observation boundary. 

So let's talk about the distribution that I think is an excellent choice for this problem.

### Embrace the Weibull Euphoria
In the 60's and 70s the Weibull-distribution was trending. It was said to be a universal PDF for scientists and engineers. This even went so far that it warranted the famous call to [beware of the Weibull euphoria](https://www.researchgate.net/publication/224582459_Beware_of_the_Weibull_Euphoria). There's many reasons not to. 

The Weibull distribution has a

* Continuous and discrete variant
* Expressive. Can take many shapes by adjusting its two parameters. 
* Closed form PDF, CDF, PMF, expected value, Median, Mode, Quantile function (inverse CDF)
* Used everywhere for predicting things that will brake since it magically appears in nature just like the normal distribution
* Weakest link property : If a system breaks with the failure of any of its independent identical components then the time to failure is approximately Weibull distributed. 
* Built in regularization mechanisms. By controlling the size of $$\beta$$ we control the peakedness hence the confidence of predicted location.

The continuous [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution) and its [discretized variant](https://en.wikipedia.org/wiki/Discrete_Weibull_distribution) has 

CHF | $$\Lambda(w) = \int_0^w \lambda(t)dt$$   | $$= \left(\frac{w}{\alpha}\right)^{\beta}$$
 HF |  $$\lambda(w) = \Lambda'(w)$$ | $$= \left(\frac{\beta}{\alpha}\right)\left(\frac{w}{\alpha}\right)^{\beta-1}$$
CDF |  $$F(w) = 1-e^{-\Lambda(w)} $$ | $$= 1-e^{-\left(\frac{w}{\alpha}\right)^{\beta}}$$
SF  |  $$S(w) = e^{-\Lambda(w)} $$ | $$= e^{-\left(\frac{w}{\alpha}\right)^{\beta}}$$
PDF | $$f(w) = \Lambda'(w)e^{-\Lambda(w)} $$ | $$=\left(\frac{\beta}{\alpha}\right)\left(\frac{w}{\alpha}\right)^{\beta-1}e^{-\left(\frac{w}{\alpha}\right)^{\beta}}$$
CDF | $$F(w) = 1-e^{-\left(\frac{w+1}{\alpha}\right)^{\beta}}$$ | $$= 1-e^{-\Lambda(w)} $$
PMF | $$p(w) =e^{-\Lambda(w)}-e^{-\Lambda(w+1)} $$ | $$ = e^{-\left(\frac{w}{\alpha}\right)^{\beta}}-e^{-\left(\frac{w+1}{\alpha}\right)^{\beta}}$$ 


We can use it to approximate alot of shapes of distributions. We can let it become infinitely flat or infinetely spiky and it can model hazard rates (also called [failure rate](https://en.wikipedia.org/wiki/Failure_rate)) that are decreasing ($$\beta<1$$), constant ($$\beta=1$$), or increasing ($$\beta>1$$). 

![weibull_distribution_animated.gif](http://i.imgur.com/maUHyto.gif)

The [exponential](https://en.wikipedia.org/wiki/Exponential_distribution) and the discrete [geometric](https://en.wikipedia.org/wiki/Geometric_distribution) distribution is the special case when $$\beta=1$$. This means that the exponential- and Weibull [Accelerated Failure Time model](https://en.wikipedia.org/wiki/Accelerated_failure_time_model) and the [Proportional Hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model) models are special cases of the WTTE-RNN. When $$\beta=2$$ it coincides with the [Rayleigh-distribution](https://en.wikipedia.org/wiki/Rayleigh_distribution). 

### Training with censored data
All you need is a special loss-function, some assumptions and some imagination. It seems like magic, how can we train on something that we haven't even observed? The idea comes from [survival analysis](https://en.wikipedia.org/wiki/Survival_analysis) (you could argue that the idea *is* survival analysis). The trick is that given some assumptions regarding how the censoring happens it turns out that the likelihood for the joint distribution of the censoring variable and the TTE can be written:

$$
\begin{equation*}
\mathfrak{L}(\theta) \propto 
	\begin{cases}
	 Pr(Y=y  |\theta) 		& \mbox{if uncensored}\\
	 Pr(Y> \tilde{y} |\theta)	& \mbox{if right censored }  \\
	\end{cases}
\end{equation*}
$$

There's some mathy assumptions and proofs to justify it probabilistically (read the thesis) but the intuition why it works is pretty clear:
![optimizing_censored.png](http://i.imgur.com/2AnTHaR.png)

With $$u=1$$ if we have uncensored data, after some manipulations we can see that the loglikelihood (**objective functions**) becomes:

continuous | $$\log(f(t)^u \cdot S(t)^{1-u})$$ |$$=$$| $$ u\cdot \log\left(\lambda(t)\right) - \Lambda(t)$$
discrete   | $$\log(p(t)^u \cdot S(t+1)^{1-u}) $$ |$$=$$| $$ u\cdot \log\left(e^{\Lambda(t+1)-\Lambda(t)}-1\right) - \Lambda(t+1)$$

To satisfy the assumptions to warrant its use we need *uninformative censoring*. With $$C$$ the censoring variable we need 

$$ Y \perp C \| \theta$$

$$ C \perp \theta$$

I.e you get no information about the parameter or the distribution of the TTE by knowing the censoring time.

A weaker assumption exists but its kind of complicated. A good indicator of whether your assumptions are met from the original problems point of view is to see if you can predict the time to the censoring point using your feature data. If you can, the algo will learn this artifact so try to mask your features so that they don't contain too much of this information. 

Be cautious but don't worry too much. If the entry time of your customers is somewhat random you can see the observation window over your timelines as a kind of slot machine making the censoring point random.

#### Gradient descent over Weibull surfaces
Let's forget about features for now and check out this simple example. I simulated some Weibull data and censored it at different thresholds. Below I show how the [RMSPROP gradient descent algorithm](http://sebastianruder.com/optimizing-gradient-descent/) (used to train Neural Networks) tries to find its way to the correct parameters (black dotted line) from four different initializations.

When we have discrete data of low resolution it's pretty clear how training works with different levels of censoring. Check the GIF below, $$\infty$$, 2 and 1 uncensored 'bins' which leads to 0%, 36.8%, and 77.9% of observations being censored:

![all_discrete_a2_censoring.gif](http://i.imgur.com/0JyrJKz.gif)
The vertical red dotted line in the rightmost graph marks where censoring occurs so TTE falling on the right of it are censored. See what happens at $$77.9$$% censoring. Here the training data only has two different values to see: $$y=0$$ or $$\tilde{y}=1$$. 
All initializations reach the correct conclusion that $$\Pr(Y=0)\approx 0.21$$ but yellow and green gets stuck in local minimas leading to erronous conclusions about the right tail of the distribution.

With higher resolution we can get away with more censoring. Here showing training with $$\infty$$, 10 and 5 uncensored 'bins' which leads to 0%, 73.9% and 91.4% of observations being censored:

![all_discrete_a20_censoring.gif](http://i.imgur.com/KfE4eKz.gif)

With truly continuous data it seven ends up figuring it out with more than $$99.9$$% censoring:
![all_contin_a2_censoring.gif](http://i.imgur.com/zUgJdpd.gif)


The takeaway is that if the way your data is being censored is *random enough* (doesn't effect the TTE and can't be predicted using your features) and you don't have too coarse TTE-data, censoring isn't much of a problem.

In the real world you *assume* that your TTE is Weibull given your data. Even if this is not true, by being such an expressive distribution it's hopefully a good enough assumption/approximation of the true distribution. Let's show it works in practice:

# Implementation & Experiments
All you need is your favorite step-to-step RNN-architecture (also called char-RNN) with a 2-dimensional positive output layer. I recommend using SoftPlus to output $$\beta$$ and exponential activation to output $$\alpha$$. 

![fig_rnn_weibull.png](http://i.imgur.com/OFKXx3C.png)

After some smart initialization you then train the network using discrete or continuous weibull-loss, here implemented in tensorflow:

{% highlight python %}
def weibull_loglikelihood_continuous(a_, b_, y_, u_,name=None):
    ya = tf.div(y_+1e-35,a_)
    return(
        tf.mul(u_,
               tf.log(b_)+tf.mul(b_,tf.log(ya))
              )- 
        tf.pow(ya,b_)
    )

def weibull_loglikelihood_discrete(a_, b_, y_, u_, name=None):
    with tf.name_scope(name):
        hazard0 = tf.pow(tf.div(y_+1e-35,a_),b_) 
        hazard1 = tf.pow(tf.div(y_+1,a_),b_)
    return(tf.mul(u_,tf.log(tf.exp(hazard1-hazard0)-1.0))-hazard1)
{% endhighlight %}

It's not pretty but it works like a charm. It even has an inbuilt regularization mechanism. We can add a penalty function that blows up when $$\beta$$ becomes large. This prevents large values of $$\beta$$ hence controls the peakedness of the predicted distribution. 

{% highlight python %}
def weibull_beta_penalty(b_,location = 10.0, growth=20.0, name=None):
    # Regularization term to keep beta below location
    with tf.name_scope(name):
        scale = growth/location
        penalty_ = tf.exp(scale*(b_-location))
    return(penalty_)
{% endhighlight %}

There's other simple extensions. Cumulative Hazard function-space is closed under summation so you can easily create new distributions like multimodal predictions. We could also extend it to multivariate TTE's by just widening the output layer together with some covariance structure but that's for another blogpost. 

## Predicting evenly spaced points
Let's try a simple example. Each training sequence is a randomly shifted sequence of evenly spaced points. The goal is to sequentially predict the number of steps to the next point. Each sequence consists of 100 timesteps. The feature data is a lagged event indicator:

$$
x_t=
\begin{cases}
1 & \text{If last step had an event}\\
0 & \text{else}\\
\end{cases}
$$

The network is a tiny LSTM with a recurrent state size 10 so $$1\times 10 \times 2$$ neurons altogether. I still had to penalize it as discussed above to avoid perfect fit and numerical instability. During training the network only got to see the censored target value in the last (rightmost) steps. I tried this using 100, 75, 50, 25 and 5 steps between the points. 

Check the results (true TTE superimposed in black, censored dotted black), it worked really well except at spacing 100 (first pic) where it goes nuts :
![evenly_spaced_100.gif](http://i.imgur.com/Cjz1yfm.gif)
![evenly_spaced_75.gif ](http://i.imgur.com/6s17rOU.gif)
![evenly_spaced_50.gif ](http://i.imgur.com/xTzd7KT.gif)
![evenly_spaced_25.gif ](http://i.imgur.com/klNuiZe.gif)
![evenly_spaced_5.gif  ](http://i.imgur.com/WAkmO5t.gif)

In the first pic the spacing was so wide that it only trained on one event at a time, meaning it never got to train on uncensored TTE's after the first event. This incentivices it to always push predicted location up after seeing an event. In doing so it gets it wrong, but it's honest about not being sure!

In my thesis I compared this with methods like disregarding the censored datapoints or treating them like uncensored. When not explicitly modeling censoring I ended up getting extremely confident *and* completely wrong predictions. Taking censoring into account won on test set every time.

## Predicting the destruction of jet-engines
There's a pretty cool dataset called the C-MAPSSS, or the [Turbofan Engine Degradation Simulation Data Set
](https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/). The subset of data used here consists of 418 sequences of 26-dimensional jet-engine sensor-readings which is used to sequentially predict the time to failure (also called Remaining Useful Life/RUL). Here we don't have censored data but it's an interesting example anyway.

I used a vanilla LSTM with width 100 of the recurrent state and a 10-node hidden layer ($$26 \times 100 \times 10 \times 2$$). With little to no hyperparameter-search I managed to get competitive results. The predicted output for some sequence that failed after 130 cycles looks something like this:

![it_61786_pmf_stack_151.png](http://i.imgur.com/kapWXMD.png)
It's pretty mesmerizing how the distribution becomes tighter and tighter as the engine starts to break down. Both the predicted expected value and the MAP (mode) gets closer and closer to the target. How does this translate to churn?

## How to use WTTE-RNN as a churn-model
In churn-prediction I've argued that you're really interested in *non events*. Instead of focusing on the events we focus on the void, i.e the *time between events*. As you know the time since an event in real time you only need to predict the time *to* events. 

Ground-truth-world | actual churners| customer with TTE $$y_t>\tau$$ **is** churned
Prediction world   | predicted churners| $$\Pr(Y_t\leq \tau)<\theta^*$$ the customer is **predicted** as churned
training world | observed churners | If $$y_t<\tau$$ customer **was** active. If $$\hat{y}_t\geq\tau$$ customer **was** churned

Just like with the sliding-box model's definition of churn we can estimate the probability of event within some fixed time window such as *if probability of purchase within $$\tau=100$$ days is less than $$\theta^*=0.01$$ customer is churned*. The big difference is that we can decide which threshold to use *after* the model is trained and we can extrapolate to set *any* threshold, like $$\tau=200$$ even if your company only existed for 100 days.

The WTTE-RNN does involve a leap of faith as we're making assumptions, arguably a more intricate assumption than with the sliding box model. Unlike with the Sliding Box model you may use the latest data for training. Not doing so is also an assumption. 

With a distribution we get much richer predictions giving us room to define a more sensitive and interpretable churn-definition.
We can also use it to derive and predict a bunch of other interesting metrics that might interface with your KPI's like DAUs and MAUs. It should also be a good start for predicting the dreadedly hard concept of Customer Lifetime Value (lifetime or next years payments is a censored datapoint).

But wait! There's more..

### WTTE-RNN produces risk-embeddings
Your predicted parameters creates a 2-d embedding (vector representation) for each of your customers current predicted activity level. 
The parameters of the Weibull-distribution has a nice interpretation. $$\alpha$$ is a location parameter (like $$\mu$$ for the normal distribution) so it gives us a prediction about *when*. $$1/\beta$$ is a scale parameter (like $$\sigma$$) so $$\beta$$ is roughly a prediction about *dispersion* or how sure we are about the location. 

This means that plotting them gives you a neat tool to monitor your whole customer base. You might even find yourself with *valuable business insights&trade;*!

From the C-MAPSS example, when plotting the predicted parameters for all the jet engines and their timesteps some weird pattern emerges. The alpha-baseline is the raw target value mean and the beta-baseline separates decreasing ($$\beta<1$$) or increasing ($$\beta>1$$) risk. 

![61786_alphabetaplot.png](http://i.imgur.com/8o0bHLu.png)
Adding time as a third axis it looks like each jet engine takes a walk on this graph. I superimposed the predicted parameters from the individual jet engine shown previously:

![61786_scatter3dplot_151.png](http://i.imgur.com/dEUj9Qe.png)
This engine took a semi-random walk from the far right corner to the bottom left corner. All engines seemed to traverse this graph in a similar fashion. 

We could name each region in some smart way. The peak ($$\alpha\approx 50$$, $$\beta> 2$$) could for example be called 'known failure onset'. Think what this plot could mean for analysing your customer onboarding process and where each customer is in it *now*.

So despite giving us a bunch of different alternatives to create a fixed churn-definition such as deciding threshold on what the predicted expected value, quantile or something else we can choose regions in the Weibull-plot and name them. 
As an example, if $$\beta$$ is really high we're pretty sure about when the next event will be. Maybe these are christmas-shoppers?

We could get a richer, but less intuitive, embedding by choosing to store the whole hidden RNN-state and use it for clustering or as features for other models. Think Word2Vec for churn. 

There's other ways to visualize a a whole customer base. Take individual timelines and stack them on top of eachother. This gives us a graph showing the prediction and how it varied through time. The jet-engine test set. 

By coloring the timelines with their predicted **alpha**:
![it_61786_alpha.png](http://i.imgur.com/s7u6t2O.png)
It tells tell us about the predicted time to *future events* from each point in time. Every horizontal line is a jet-engine and its predicted value. Here the prediction goes from healthy *far away*-red to worrying *anytime soon!*-blue

By coloring them by their predicted **beta**:
![it_61786_beta.png](http://i.imgur.com/M0uSsdp.png)
It tells us how confident the predictions where. It seems like the confidence was the highest (red) at what looks like the onset of degradating health. 

Here the xlab *time* means survival time but for a customer database it would a timeline of cohorts. This could give you a realtime prediction of current churn (by threshold in the right tail say) and DAU (thresholding in the left tail). By taking mean over the prediction you would get a predicted *rate* for all your customers. 

# Summary
Too summarise, the WTTE-RNN can:

* Handle discrete or continuous time
* Train on censored data
* use temporal features/time varying covariates
* Learn long-term temporal patterns

And it's less hacky than the sliding box model as you don't need to set some arbitrary window size before training your model. The whole modeling cycle just gets smoother. You can use all your available data and you get output that's interpretable. 

There are assumptions (it's a *model*, not reality), but they are explicit - no hidden or dirty tricks:

* Assume the time to event to be Weibull distributed given features
* Assume uninformative censoring

I'm currently working on cleaning up some research-grade dirty code to put it on github. PM me if you can't wait.

ps. I'm sure alot of people have built this model or variants of it previously but I haven't been able to find any papers on it. If you have and want to discuss it/be cited please get in touch!


