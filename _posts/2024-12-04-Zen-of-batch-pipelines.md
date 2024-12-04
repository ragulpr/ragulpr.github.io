---
layout: post
title: Zen of Batch Pipelines - A recipe to reduce cognitive load
---
[![](https://victorianweb.org/art/illustration/dore/bible/8.jpg)](https://victorianweb.org/art/illustration/dore/bible/8.html)

In this post, I'll share some hard-earned tricks for building and maintaining many many batch data pipelines. It's all about coercing yourself to keep the mental overhead to a minimum.

For the past 5 years I worked at Schibsted, a multinational Marketplace & Media giant in tiny Nordics. It was the deep learning ambitions and the massive streams of event data that lured me in and the great people and engineering culture and that kept me there.

My team ran the machine learning pipelines for personalization and advertising (roughly _behavioural events  in $$\rightarrow$$ predicted user traits out_). We maintained some 25 pipelines × 10s stages × 3 countries × dev/pre/pro under strict privacy, quality and downtime constraints.


Over the years I had many reasons to think about scaling a data team and its limiting factors. This matters because my humble opinion is that the only way an ML/Data team can create value is by _running_ their own jobs & services. Center of expertise? Internal consultants? No. More jobs $$\propto$$ more value and the written-in-flesh limit is the shared context of 12 people.

To be able to debug & develop on the already vast error surface, cognitive load - hence variation - had to be kept in check. This was collectively learnt from thousands of hours sifting through confs, logs, stack-traces, s3, code, monitors etc etc.

I wrote this because my team needed a shared mental model of our battle-tested design principles. It would end up serving as a portal piece for many design discussions.

Maybe it's of interest to the internet even if you're not using an analogous stack to what we had (Scala/Spark/Python/K8s/Luigi/S3). You can read this with one finger solemnly raised if it helps.

# The Zen of batch pipelines

Purpose of this document is to keep track and align programming principles for our batch pipelines.
We should update this doc and revisit it to keep track of style/design decisions. If possible; keep it short/readable/declarative with the intended readers in mind:

* New team members trying to understand why things are like they are.
* Old team member referencing this doc in a design decision/PR/review.

## Guiding Principles / Desired state
Our data pipelines should follow functional programming principles:
**reproducibility, atomicity, and readability**

### Infra
```
travis         {on commit}        =>
Spinnaker      {on merge}         =>
k8s cronjob    {every 20 min}     =>
k8s Job        repeat             =>
luigi worker   {if not completed} =>
batch stage => {s3://../_SUCCESS}
```

* We automate deployments to pre/pro.
* Kubernetes Cronjobs schedules luigi pods
* luigi launches batch job _stages_
* A batch job stage is uniquely defined by its luigi input arguments
* Batch job is dockerized
* The `main` branch intends to reflect _production_ code. Whatever is in `pro`-_environment_ can be assumed to be _production_ - it's depended on.

### A _stage_:

* is a transformation `Inputs => Output` launched by luigi. Typically `s3 => s3`
  * Example: `s3://../events-raw/.. => s3://../events-preprocessed/..`
* Is **idempotent**: Same luigi input params produces the same output.
    * We don't like side effects. If unavoidable - all of them are easily understood from luigi task.
    * Output can be traced to a commit.
    * We strive for deterministic programs. Fixing seeds or maintaining sort order helps.
* Is [**atomic**](https://luigi.readthedocs.io/en/stable/tasks.html#task-output). Has completeness defined by its single `_SUCCESS` file.
* Is [versioned](https://www.researchgate.net/publication/316651123_Versioning_for_End-to-End_Machine_Learning_Pipelines). Its path `/version=./` is always bumped for backwards incompatible changes.
* _Work_ of stage is done in either Spark/Scala or Python. We avoid inlining it in luigi.
* has easily found and enforced output format/schema. The output of a Spark stage is a `Dataset[Type]`.
* A single stage is better than more stages.
* Pipelines should be self-contained:
    * Keep the interdependencies between pipelines to a minimum. Each pipeline produces its own derived datasets (aggregations, features, models) rather than depend on other pipelines. Redundant compute is cheaper than blocked developers.
    * _External systems_ are accessed through standardized datasets or shared libraries. Ex: Don't call an API to fetch data into your stage. Read data from a stage that persisted data from an API call.

### Naming

* A luigi task is named after its output _dataset_ (sideeffect). Think "what's a good sql _table_ name"
* Dataset names explain its _content_, rather  than transformations to get there.
* Transformation code (eg. Scala `main`) can be named after what it does if it's used more than once.
* It's best if luigi, s3, and scala names are identical enough for a dumb IDE to help us with search-replace-refactoring.
* Ideally we want to be able to infer from the data path who:
  * Wrote it (Pipeline)
  * Who triggered it (luigi task)
  * What code was involved (scala/python main class)
  * What's in the dataset and what output schema it has (name of output case class)
  * Infer from column names where data comes from

<!-- <details> -->
  <!-- <summary>Examples</summary> -->

#### Examples
##### Good:

* `EventsWithLocations` luigi task
* `s3://.../events-with-locations/../`
* Scala main class under `com.schibsted.ate.pipelinename.jobs`:
  * `EventsWithLocationsJob.scala`
* Output type is `Dataset[EventWithLocation]`

A full path could look like
```
s3://bucket-name/retention=5/pipeline-name/events-with-locations/version=1/lookback=25/year=2024/month=1/day=1/hour=1/_SUCCESS
```
This is long indeed but everything serves its purpose.

##### Bad:

* `InterestAttributeFilterPredictor` luigi task
* `s3://bucket-name/pipeline-name/ecosystem/attribute-filter-predictions/version=0.0.3/..`
* `FilterPredictor.scala`
* Output type is an untyped `DataFrame`

<!-- </details> -->


<!-- <details> -->
  <!-- <summary> Implications</summary> -->
#### Implications

* Luigi does all the path checking/manipulations.
* If your stage is not easily named such that its luigi task explains its sideffect - you're likely trying to do something we shouldn't do.
* If you don't know exactly what the output format of a stage is - a consumer won't either.
* The mental model described here may not always fit to how we want to _write_ code, but if we try to conform to any kind consistent mental model it will help _reading_ and reasoning about the system (and mainly) the data afterwards.

</details>

## Testing

* Goal: if travis says ✅ then the k8s job runs without failure.
* _Luigi tests_ should explicitly and verbosely verify expected sideffects - i.e the exact function call like input/output path of stage.
* Scala `main` should only contain IO and calls to a function which is tested.
* We prefer explicit and dumb unit tests.
* If it's hard to test - you probably shouldn't do it.

## Style

* We try to minimize library dependencies.
* Formatting is automatically enforced via `pre-commit` and `scalafmt`.
* We enforce mypy typing. Every function should be using types.
* We gravitate towards spark `Dataset` api and shunn the `org.apache.spark.ml.Transformer` Api.
* We [avoid Spark udf's](https://www.google.com/search?q=avoid+udf+spark)
* We keep luigi task parameters  to  a minimum.
    * Parameters are passed explicitly. Luigi tasks do not `.clone()`.
    * We avoid default parameters.
in the _Pipeline Differences_ Google doc
* (_Some Pipeline Name_) is the gold standard. Changes should be propagated to it and from it. We try to keep other pipelines in sync with this pipeline.