---
layout: post
title: How to evaluate multiclass classification 
---

It's surprisingly quiet in the machine-learning world about the subtleties of analyzing how well your model performs when you're predicting more than two classes i.e multiclass classification. 
It doesn't even seem like a problem; isn't it just enough to report total classification accuracy after classifying into the most 'likely' class? The short answer is *no*. The longer answer is

> Each ROC-point is a confusion-matrix resulting from classification at a given threshold.

Which you might know from regular binary classification. But what does this mean in the multiclass world? I will give you the basic introduction to evaluating classifiers when there's 2 *or more* classes.

## Binary and multiclass classification
Your machine learning model will output a *score*. You may even think it has probabilistic meaning but you shouldn't assume it does before you test it. 

$$k=1,2,\ldots,K$$ classes 

$$i=1,2,\ldots,n$$ observations 

$$y_{ik}=1$$ if observation $$i$$ had class $$k$$

$$s_{ik}=1$$ the *score*/output of your machine-learning model. This could be a probability or some other value

Assuming that the model will be used to take decisions, you will have to find a threshold/decision rule. You should not always assume that the argmax-rule is good. 

* *Separation*: How well can we separate classes? 
	* Tool : Multiclass ROC
	* Metric : 1vsA-AUC, AvsA-AUC 
* *Calibration*: Does the outputted probabilities say what I think they say?
	* Tools : Binning probabilities
	* Metric : Brier score, mean categorical log-likelihood (cross entropy), Hosmer-Lemeshew test
* *Classification*: How well does my model perform at the set threshold?
	* Tool : Confusion matrix
	* Metric : accuracy, mean sensitivities

There's two very obvious properties of classification:

* Being in one class means that you're not in another. $$\sum_k x_{ik} =1$$
* We only need to know $$K-1$$ values to figure out the last.

I find it helpful to disregard the second property as it results in nicer code and formulas. 

In general there's two approaches: 1 vs all and all vs all. In the first case we see multiclass classification as $$K$$ binary prediction problems. 1 class is compared to the rest



## Evaluating classification
I have a cancer-diagnosis model that will work in $$99.99$$% of the cases: you don't have it. This is the reason why we don't report accuracy. 

* Classification is conditionally independent

### Confusion matrix

## Evaluating calibration
There's a $$

### Binning

### Log-likelihood (cross-entropy)

### Brier score

## Evaluating separation

### AUC-matrix


### MAUC


### Artistic philosophy

* Sens vs Spec
* colorcode your classes and/or present them in some natural order
	* Adds *lots* of complexity to all your plotting-pipeline.