# classifiers-monorepo

- [classifiers-monorepo](#classifiers-monorepo)
  - [Outline](#outline)
  - [Dependancies](#dependancies)
  - [Supervised Classification](#supervised-classification)
  - [Overview](#overview)
  - [Lazy learner](#lazy-learner)
    - [K-Nearest Neighbor](#k-nearest-neighbor)
      - [K-Nearest Neighbor : Overview](#k-nearest-neighbor--overview)
      - [K-Nearest Neighbor : Sample git repos](#k-nearest-neighbor--sample-git-repos)
    - [Case-based Reasoning](#case-based-reasoning)
      - [Case-based Reasoning : Overview](#case-based-reasoning--overview)
      - [Case-based Reasoning : Terminologies](#case-based-reasoning--terminologies)
        - [Case-based Reasoning : Case Definition](#case-based-reasoning--case-definition)
        - [Case-based Reasoning : Case-base Definition](#case-based-reasoning--case-base-definition)
      - [Case-based Reasoning : Advantages/Disadvantages](#case-based-reasoning--advantagesdisadvantages)
      - [Case-based Reasoning : Use Cases](#case-based-reasoning--use-cases)
      - [Case-based Reasoning : Sample Git Repos](#case-based-reasoning--sample-git-repos)
  - [Eager Learner](#eager-learner)
    - [Decision Tree](#decision-tree)
      - [Decision Tree: Overview](#decision-tree-overview)
      - [Decision Tree : Terminologies](#decision-tree--terminologies)
        - [Decision Tree : Node Related Definitions](#decision-tree--node-related-definitions)
        - [Decision Tree : Impurity](#decision-tree--impurity)
        - [Decision Tree : Entropy](#decision-tree--entropy)
        - [Decision Tree : Information Gain](#decision-tree--information-gain)
      - [Decision Tree : Classification Type](#decision-tree--classification-type)
      - [Decision Tree : Regression Type](#decision-tree--regression-type)
      - [Decision Tree : Advantages/Disadvantages](#decision-tree--advantagesdisadvantages)
      - [Decision Tree : Use Cases](#decision-tree--use-cases)
      - [Decision Tree : Interesting Papers](#decision-tree--interesting-papers)
      - [Decision Tree : Sample Git Repos](#decision-tree--sample-git-repos)
    - [Random Forest](#random-forest)
      - [Random Forest : Overview](#random-forest--overview)
      - [Random Forest : Advantages/Disadvantages](#random-forest--advantagesdisadvantages)
      - [Random Forest : Sample Git Repos](#random-forest--sample-git-repos)
    - [Support Vector Machine](#support-vector-machine)
      - [Support Vector Machine : Overview](#support-vector-machine--overview)
      - [Support Vector Machine : Terminologies](#support-vector-machine--terminologies)
        - [Support Vector Machine : Kernel](#support-vector-machine--kernel)
        - [Support Vector Machine : Regularization](#support-vector-machine--regularization)
        - [Support Vector Machine : Gamma](#support-vector-machine--gamma)
        - [Support Vector Machine : Margin](#support-vector-machine--margin)
      - [Support Vector Machine : Advantages/Disadvantages](#support-vector-machine--advantagesdisadvantages)
      - [Support Vector Machine : Sample Git Repos](#support-vector-machine--sample-git-repos)
    - [Naive Bayes](#naive-bayes)
      - [Naive Bayes : Overview](#naive-bayes--overview)
      - [Naive Bayes : Advantages/Disadvantages](#naive-bayes--advantagesdisadvantages)
      - [Naive Bayes : Use Cases](#naive-bayes--use-cases)
      - [Naive Bayes : Sample Git Repos](#naive-bayes--sample-git-repos)
    - [Artificial Neural Networks](#artificial-neural-networks)
  - [References](#references)

## Outline

This repo is used to host material on machine learning classification methodologies and algorithms.

## Dependancies

To offer consistent workflow, all files and dependencies are compiled in isolated docker container by default when using `gnu make`.
If you want to build and run make target on your own machine , out of docker containers , edit Makefile and set `DOCKER_ENV` to `false`. Make sure to run `make dep` in case you choose to not use the dockerized build pipeline.

All diagrams and flowcharts are generated with [`Mermaid.JS`](https://mermaidjs.github.io) . Source codes for mermaid files are stored under `mermaid` directory and generated pdf files can be found at `fixtures/mermaid`. You can generate diagrams by running `make mermaid` in terminal.

An script for setting up development environment for machine learning is provided at `scripts/environment-setup`. This script has been tested on has been tested on google compute engine and is designed to set up a debian development environment with `python`, `conda`,`numpy`,`tensorflow(GPU)` ,`keras` to name a few and also tries to setup `CUDA`. A docker file will be provided in the future that uses these script to create a container for running all deep learning tasks. If you are planning on using this script to setup your environment ( or build the docker file ), Due to licensisng issues, you must download `cuDNN` from `nVidia` website before running the script and put it in the same directory as the rest of the files (`scripts/environment-setup`). Follow the guide in this [`Medium Article`](https://medium.com/@jayden.chua/quick-install-cuda-on-google-cloud-compute-6c85447f86a1).

## Supervised Classification

![Supervised Classifiers Based On Learner Type][supervised-root]

## Overview

`Lazy learner` simply store the training data and wait up until testing data appears ; they classify at that point based on the most related data in the stored training data. To put it in simpler terms, A lazy learner does not have a training phase. As an example, lazy learners are used by online recommendation systems.

`Eager learner` on the other hand before receiving data for classification setup a classification model which is based on the data passed to it during the training phase. Eager learning algorithms must be able to commit to a single hypothesis that covers the entire instance space. to put it in simpler terms, An eager learner has a model fitting or training step.

|   Classifier  | Time To Train | Time To Predict |
|:-------------:|:-------------:|:---------------:|
|  Lazy Learner |      Low      |       High      |
| Eager Learner |      High     |       Low       |

## Lazy learner

### K-Nearest Neighbor

#### K-Nearest Neighbor : Overview

`KNN` Is one of the more basic algorithms and is oftenly used to benchmark more complex classifiers like Artificial Neural Networks and Support Vector Machines.

It operates by storing all instances correspond to training data points in n-dimensional space. When an unknown discrete data is received, it analyzes the closest k number of instances saved and returns the most common class as the prediction result .In case of real-valued data it returns the mean of k nearest neighbors.

There is also other variation of KNN such as:

- [Distance-weighted KNN](https://www.geeksforgeeks.org/weighted-k-nn/)
- [dual distance-weighted KNN](https://pdfs.semanticscholar.org/a128/62972be0e7e6e901825723e703117a6d8128.pdf)
- [Modified K-Nearest Neighbor (MKNN)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.545&rep=rep1&type=pdf)
- [Dynamic K-Nearest-Neighbor](https://ieeexplore.ieee.org/abstract/document/5559858)

I also found an inseresting [paper](https://ieeexplore.ieee.org/abstract/document/4406010) that offers different methodologies to pmproving K-Nearest-Neighbor for Classification.

#### K-Nearest Neighbor : Sample git repos

The following are some real-world sample implementations and/or utilization of KNN:

- [Simple Python Implementation of KNN algorithm](https://github.com/iiapache/KNN)

### Case-based Reasoning

#### Case-based Reasoning : Overview

![Case-Based Reasoning Flowchart][case-based-reasoning]

In a nutshell, CBR is reasoning by remembering: previously solved problems (cases) are used to suggest solutions for novel but similar problems. Kolodner lists four assumptions about the world around us that represent the basis of the CBR approach:

- `Regularity`: the same actions executed under the same conditions will tend to have the same or similar outcomes.
- `Typicality`: experiences tend to repeat themselves.
- `Consistency`: small changes in the situation require merely small changes in the interpretation and in the solution.
- `Adaptability`: when things repeat, the differences tend to be small, and the small differences are easy to compensate for.

#### Case-based Reasoning : Terminologies

##### Case-based Reasoning : Case Definition

`case` is explained as several features describing a problem plus an outcome or a solution.They are records of real events and are excellent for justifying decisions.cases can be very rich and have elements consisting of texts, numbers, symbols, plans, multimedia. Keep in mind that they are not usually distilled knowledge. A case consists of a problem, its solution, and, typically, annotations about how the solution was derived. There are two major types of case features:

- `unindexed features`: They provide background information to users and are not predictive & not used for retrieval.
- `indexed features` : Predictive and used for retrieval.

##### Case-based Reasoning : Case-base Definition

`case-base` is a set of cases.Usually, they just flat files or relational databases. a robust case-base, containing a representative and well distributed set of cases, is the foundation for a good CBR system.

Case-based reasoning process is ca be describe as :[1]

- `Retrieve`: retrieve from memory cases relevant to solving a given target problem.
- `Reuse`: Map the solution from the previous case to the target problem. This may involve adapting the solution as needed to fit the new situation.
- `Revise`: Having mapped the previous solution to the target situation, test the new solution in the real world (or a simulation) and, if necessary, revise.
- `Retain`: After the solution has been successfully adapted to the target problem, store the resulting experience as a new case in memory.

#### Case-based Reasoning : Advantages/Disadvantages

Advantages/Disadvantages of CBR are summarised in the following table :

| Advantages                                          | Disadvantages                                                                                                         |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| CBR is intuitive since it mimics it’s how we work.  | Can take large storage space for all the cases                                                                        |
| CBR Development easier                              | Can take large processing time to find similar cases in case-base                                                     |
| No knowledge elicitation to create rules or methods | Cases may need to be created by hand                                                                                  |
| Systems learn by acquiring new cases through use    | Adaptation may be difficult                                                                                           |
| Maintenance is easy                                 | Needs case-base, case selection algorithm, and possibly case-adaptation algorithm.                                    |
| Justification through precedent                     | CBR may not be for you if you require the best solution or the optimum solution.                                      |
|                                                     | CBR systems generally give good or reasonable solutions. This is because the retrieved case often requires adaptation |

#### Case-based Reasoning : Use Cases

It is advised to use CBR in the following cases :

- when a domain model is difficult or impossible to elicit
- when the system will require constant maintenance
- when records of previously successful solutions exist
  
The following table consists of a list of projects illustrating recent themes in case-based reasoning and knowledge discovery :

| Citation                                | Themes                                 | Phase                                | Application Domain                     |
|-----------------------------------------|----------------------------------------|--------------------------------------|----------------------------------------|
| Adedoyin et al. 2016                    | Big Data Explanation Application       | Retrieval                            | Fraud Detection                        |
| Barua et al. 2014                       | Signal Processing  Application         | Case Acquisition Retrieval           | Classifying Ocular Artifacts in EEGs   |
| Canensi et al. 2014 Canensi et al. 2016 | Big Data Explanation Interactivity     | Retrieval                            | Medical Processes                      |
| Dileep and Chakraborti 2014             | Big Data Knowledge Rich CBR            | Retrieval Similarity                 | Textual CBR                            |
| Eyorokon et al. 2016                    | Explanation Interactivity              | Retrieval                            | Conversational CBR Dialogue            |
| Guo, Jerbi, and O’Mahony 2014           | Big Data Application                   | Retrieval Similarity Weight Learning | Job Recommendation                     |
| Hromic and Hayes 2014                   | Big Data Signal Processing Application | Case Mining                          | Twitter Datasets for Event Detection   |
| Olsson et al. 2014                      | Signal Processing Application          | Retrieval Similarity                 | Fault Diagnosis in Heavy Duty Machines |
| Sekar and Chakraborti 2016              | Big Data Interactivity Application     | Retrieval                            | Product Recommendation                 |
| Tomasic and Funk 2014                   | Regression Analysis Application        | Reuse Revise                         | Quality Control in Manufacturing       |
| Zhang, Zhang, and Leake 2016            | Big Data                               | Retain                               | Streaming Data                         |

#### Case-based Reasoning : Sample Git Repos

The following are some repos consisting of real-world sample implementations and/or utilization of CBR:

- [CBR usage in finding missing values in data](https://github.com/SanjinKurelic/CaseBasedReasoning)

## Eager Learner

### Decision Tree

#### Decision Tree: Overview

Decision Tree Classifier, repetitively divides the working area(plot) into sub part by identifying lines.
the division is terminated in one of the following two cases  :

- it has divided into classes that are pure (only containing members of single class )
- Some criteria of classifier attributes are met.

there are two main types of Decision tree : `Classification Trees` and `Regression Trees`.

#### Decision Tree : Terminologies

##### Decision Tree : Node Related Definitions

- `Nodes` : Test for the value of a certain attribute.
- `Edges/ Branch` : Correspond to the outcome of a test and connect to the next node or leaf.
- `Splitting`: It is a process of dividing a node into two or more sub-nodes.
- `Pruning`: When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting.
- `Root Node`: It represents entire population or sample and this further gets divided into two or more homogeneous sets.
- `Leaf nodes` : Terminal nodes that predict the outcome (represent class labels or class distribution).
- `Decision Node` : When a sub-node splits into further sub-nodes, then it is called decision node.
- `Parent and Child Node`: A node, which is divided into sub-nodes is called parent node of sub-nodes whereas sub-nodes are the child of parent node.

##### Decision Tree : Impurity

Impurity is when we have a traces of one class division into other. This can arise due to following reason

- We run out of available features to divide the class upon.
- We tolerate some percentage of impurity (we stop further division) for faster performance.

##### Decision Tree : Entropy

Entropy is degree of randomness of elements or in other words it is measure of impurity. Mathematically, it can be calculated with the help of probability of the items as:

```formula
H = -sum[P(x)*log(P(x))]
```

##### Decision Tree : Information Gain

In case of multiple features to divide the current working set, we select the gives us less impurity for division, In that case the information gain at any node is defined as

```formula
Information Gain (n) = Entropy(x) — ([weighted average] * entropy(children for feature))
```

Decision tree at every stage selects the one that gives best information gain. When information gain is 0 means the feature does not divide the working set at all.

#### Decision Tree : Classification Type

`Classificaion` trees are binary (Yes/No data types) trees. In this case , the decision variable is **Discrete** (Categorical).
Such a tree is built through a process known as binary recursive partitioning. This is an iterative process of splitting the data into partitions, and then splitting it up further on each of the branches.

#### Decision Tree : Regression Type

`Regression` trees are used when data type is **Continuous** . In many cases the target variables are real numbers.  

#### Decision Tree : Advantages/Disadvantages

Advantages/Disadvantages of Decision Trees are summarised in the following table :

| Advantages                                                                        | Disadvantages                                                                                    |
|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Inexpensive to construct                                                          | Easy to overfit                                                                                  |
| Extremely fast at classifying unknown records                                     | Decision Boundary restricted to being parallel to attribute axes                                 |
| Easy to interpret for small-sized trees                                           | Decision tree models are often biased toward splits on features having a large number of levels  |
| Accuracy comparable to other classification techniques for many simple data sets. | Small changes in the training data can result in large changes to decision logic                 |
| Excludes unimportant features                                                     | Large trees can be difficult to interpret and the decisions they make may seem counter intuitive |

#### Decision Tree : Use Cases

Decision trees can be used in the following cases :

- there is an objective the the user is tying to achieve , such as maximizing profit.
- there are several courses of action
- there are events beyond control of the decision maker , eg. environmental factors
- there is a calculable,quantifiable measureof benefit of the various alternatives
- there is uncertainty surronding the outcome,eg: which will actually happen.

#### Decision Tree : Interesting Papers

- [Al Hamad, M., & Zeki, A. M. (2018, November). Accuracy vs. Cost in Decision Trees: A Survey. In 2018 International Conference on Innovation and Intelligence for Informatics, Computing, and Technologies (3ICT) (pp. 1-4). IEEE.](https://ieeexplore.ieee.org/abstract/document/8855780)

#### Decision Tree : Sample Git Repos

- [Extremely Fast Decision Tree implementation](https://github.com/doubleplusplus/incremental_decision_tree-CART-Random_Forest_python)
  - `paper` : [Manapragada, C., Webb, G. I., & Salehi, M. (2018, July). Extremely fast decision tree. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 1953-1962). ACM.](https://dl.acm.org/citation.cfm?id=3220005)

### Random Forest

#### Random Forest : Overview

Random Forest is an ensemble-learning model. An ensemble-learning model aggregates multiple Machine Learning models to improve performance. Each of the models, when used on their own, is weak. However, when used together in an ensemble, the models are strong—and therefore generate more accurate results.Decision Tree Classifier are the underlying concept for Random Forest Classifier.
The objective behind random forests is to take a set of high-variance, low-bias decision trees and transform them into a model that has both low variance and low bias. By aggregating the various outputs of individual decision trees, random forests reduce the variance that can cause errors in decision trees. Random forest classifier creates a set of decision trees from randomly selected subset of training set. It then aggregates the votes from different decision trees to decide the final class of the test object. Keep in mind that the subsets in different decision trees created may overlap.
A variation of Random forest is the case that weight concept is applied for considering the impact of result from any decision tree. Tree with high error rate are given low weight value and vise versa. This would increase the decision impact of trees with low error rate.

#### Random Forest : Advantages/Disadvantages

Advantages/Disadvantages of Random Forest are summarised in the following table :

| Advantages                                                                                    | Disadvantages                                                                 |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| There is no need for feature normalization.Robust to Outliers, Non-linear and unbalanced Data | They’re not easily interpretable                                              |
| Individual decision trees can be trained in parallel                                          | They’re not a state-of-the-art algorithm                                      |
| Versatility. They can be used in regression or classification task                            | For very large data sets, the size of the trees can take up a lot of memory.  |
| Great with High dimensionality                                                                | It can tend to overfit, which can be dealt with by tuning the hyperparameters |
| Quick Prediction/Training Speed                                                               |                                                                               |

#### Random Forest : Sample Git Repos

- [Robust Random Cut Forest implementation](https://github.com/kLabUM/rrcf)
  - `paper` : [Guha, S., Mishra, N., Roy, G., & Schrijvers, O. (2016, June). Robust random cut forest based anomaly detection on streams. In International conference on machine learning (pp. 2712-2721).](http://proceedings.mlr.press/v48/guha16.pdf)
- [Random Survival Forest implementation](https://github.com/julianspaeth/random-survival-forest)
  - `paper` : [Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008). Random survival forests. The annals of applied statistics, 2(3), 841-860.](https://projecteuclid.org/euclid.aoas/1223908043)
- [Quantile Random Forest Regression implementation](https://github.com/dfagnan/QuantileRandomForestRegressor)
  - `paper` : [Meinshausen, N. (2006). Quantile regression forests. Journal of Machine Learning Research, 7(Jun), 983-999.](http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf)
- [Random Bits Forest binary wrapper compatible with sklearn](https://github.com/tmadl/sklearn-random-bits-forest)
  - `paper` : [Wang, Y., Li, Y., Pu, W., Wen, K., Shugart, Y. Y., Xiong, M., & Jin, L. (2016). Random bits forest: a strong classifier/regressor for big data. Scientific reports, 6, 30086.](https://www.nature.com/articles/srep30086.pdf)
- [Deep Random Forests implementation](https://github.com/matejklemen/deep-rf)
  - `paper` : [Zhou, Z. H., & Feng, J. (2017). Deep forest: Towards an alternative to deep neural networks. arXiv: 1702.08835 v1.](https://arxiv.org/pdf/1702.08835.pdf)
- [Random Forest model using Hellinger Distance as split criterion](https://github.com/EvgeniDubov/hellinger-distance-criterion)
- [CART regression tree and random forests implementation](https://github.com/chandarb/Python-Regression-Tree-Forest)
  - `paper` : [Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). Classification and regression trees. Wadsworth Int. Group, 37(15), 237-251.](https://www.sciencedirect.com/science/article/abs/pii/0377221785903212)
- [Unsupervised Clustering using Random Forests](https://github.com/joshloyal/RandomForestClustering)

### Support Vector Machine

#### Support Vector Machine : Overview

A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.

#### Support Vector Machine : Terminologies

##### Support Vector Machine : Kernel

The learning of the hyperplane in linear SVM is done by transforming the problem using some linear algebra. This is where the **Kernel** plays role.
For linear kernel the equation for prediction for a new input using the dot product between the input (x) and each support vector (xi) is calculated as follows:

```formula
f(x) = B(0) + sum(ai * (x,xi))
```

The polynomial kernel can be written as:

```formula
f(x,xi) = 1 + sum(x * xi)^d
```

The exponential kernel can be written as:

```formula
f(x,xi) = exp(-gamma * sum((x — xi^2))
```

##### Support Vector Machine : Regularization

The **Regularization** parameter (often termed as C parameter in python’s sklearn library) tells the SVM optimization how much you want to avoid misclassifying each training example.
For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points.

##### Support Vector Machine : Gamma

The **Gamma** parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’. In other words, with low gamma, points far away from plausible seperation line are considered in calculation for the seperation line. Where as high gamma means the points close to plausible line are considered in calculation.

##### Support Vector Machine : Margin

**Margin** is a separation of line to the closest class points.
A good margin is one where this separation is larger for both the classes. A good margin allows the points to be in their respective classes without crossing to other class.

#### Support Vector Machine : Advantages/Disadvantages

Some advantages/disadvantages of SVMs are summarised in the following table :

| Advantages                                                                                  | Disadvantages                                                                                                                                                            |
|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SVM works relatively well when there is clear margin of separation between classes.         | SVM algorithm is not suitable for large data sets.                                                                                                                       |
| SVM is more effective in high dimensional spaces.                                           | SVM does not perform very well, when the data set has more noise i.e. target classes are overlapping.                                                                    |
| SVM is effective in cases where number of dimensions is greater than the number of samples. | In cases where number of features for each data point exceeds the number of training data sample , the SVM will under perform.                                           |
| SVM is relatively memory efficient                                                          | As the support vector classifier works by putting data points, above and below the classifying hyper plane there is no probabilistic explanation for the classification. |

#### Support Vector Machine : Sample Git Repos

- [A Neural Network Architecture Combining Gated Recurrent Unit (GRU) and Support Vector Machine (SVM) for Intrusion Detection](https://github.com/AFAgarap/gru-svm)
  - `paper` : [Agarap, A. F. M. (2018, February). A neural network architecture combining gated recurrent unit (GRU) and support vector machine (SVM) for intrusion detection in network traffic data. In Proceedings of the 2018 10th International Conference on Machine Learning and Computing (pp. 26-30). ACM.](https://arxiv.org/pdf/1709.03082.pdf)
- [An Architecture Combining Convolutional Neural Network (CNN) and Linear Support Vector Machine (SVM) for Image Classification](https://github.com/AFAgarap/cnn-svm)
  - `paper` : [Agarap, A. F. (2017). An architecture combining convolutional neural network (CNN) and support vector machine (SVM) for image classification. arXiv preprint arXiv:1712.03541.](https://arxiv.org/abs/1712.03541)
- [Quasi-Newton Semi-Supervised Support Vector Machine Algorithm](https://github.com/LorenzoNorcini/Quasi-Newton-S3VM)
  - `paper` : [Gieseke, F., Airola, A., Pahikkala, T., & Kramer, O. (2012). Sparse Quasi-Newton Optimization for Semi-supervised Support Vector Machines. In ICPRAM (1) (pp. 45-54).](http://www.fabiangieseke.de/pdfs/icpram2012.pdf)
  - `paper` : [Gieseke, F., Airola, A., Pahikkala, T., & Kramer, O. (2014). Fast and simple gradient-based optimization for semi-supervised support vector machines. Neurocomputing, 123, 23-32.](http://www.fabiangieseke.de/pdfs/neucom2013_draft.pdf)
- [implementation of a Support Vector Machine using the Sequential Minimal Optimization (SMO) algorithm for training](https://github.com/LasseRegin/SVM-w-SMO)
- [Cost-Sensitive Support Vector Machines](https://github.com/airanmehr/cssvm)
- [Classification and Clustering using Support Vector Machine and Enhanced Fuzzy C-Means](https://github.com/febrianimanda/SVM-FCM)
- [genetic algorithm used for optimizing svm hyperparameters](https://github.com/senolakkas/sklearn-optimize)

### Naive Bayes

#### Naive Bayes : Overview

Naive Bayes classifier calculates the probabilities for every factor and Then it selects the outcome with highest probability.This classifier assumes the features are independent,Hence the word naive.

```formula
              P(B | A) * P(A)
 P(A | B) = ------------------
                   P(B)
```

#### Naive Bayes : Advantages/Disadvantages

| Advantages                                                                                                                                        | Disadvantages                                                                                                                              |
|---------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| Low propensity to overfit : For problems with a small amount of training data, it can achieve better results than other classifiers               | Cannot incorporate feature interactions                                                                                                    |
| Relatively quick training and prediction                                                                                                          | Performance is sensitive to skewed data. when the training data is not representative of the class distributions in the overall population |
| Modest memory footprint : the operations do not require the whole data set to be held in RAM at once                                              | Needs sizable amount of data for regression problems, i.e. continuous real-valued data to calculate calculate  likelihoods correctly       |
| Modest CPU usage : there are no gradients or iterative parameter updates to compute, since prediction and training employ only analytic formulae. |                                                                                                                                            |
| Linear scaling with number of features and number of data points, and is easy to update with new training data.                                   |                                                                                                                                            |
| handles missing feature values by re-training and predicting without that feature                                                                 |                                                                                                                                            |

#### Naive Bayes : Use Cases

it is powerful algorithm mostly used for:

- Real time Prediction
- Text classification/ Spam Filtering
- Recommendation System

#### Naive Bayes : Sample Git Repos

- [Multiclass Naive Bayes Support Vector Machine](https://github.com/lrei/nbsvm)
  - `paper` : [Wang, S., & Manning, C. D. (2012, July). Baselines and bigrams: Simple, good sentiment and topic classification. In Proceedings of the 50th annual meeting of the association for computational linguistics: Short papers-volume 2 (pp. 90-94). Association for Computational Linguistics.](https://dl.acm.org/citation.cfm?id=2390688)
- [Gibbs sampler for for a Naive Bayes document classifier implementation](https://github.com/wpm/Naive-Bayes-Gibbs-Sampler)
  - `paper` : [Resnik, P., & Hardisty, E. (2010). Gibbs sampling for the uninitiated (No. CS-TR-4956). Maryland Univ College Park Inst for Advanced Computer Studies.](https://drum.lib.umd.edu/handle/1903/10058)
- [`Tree Augmented Naive Bayes classifier implementation`](https://github.com/Anaphory/augmented_bayes)
  - `paper` : [Zheng, F., & Webb, G. I. (2010). Tree augmented naive Bayes. Encyclopedia of Machine Learning, 990-991.](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_850)
- [Gaussian Naive Bayes classifier implementation](https://github.com/amallia/GaussianNB)
- [Hybrid Naive Bayes classifier implementation](https://github.com/ashkonf/HybridNaiveBayes)

### Artificial Neural Networks

## References

Papers used as refrence are kept under `classifiers-monorepo/fixtures/papers` directory.

- Agnar Aamodt and Enric Plaza, "Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches," Artificial Intelligence Communications 7 (1994): 1, 39-52.
- Bichindaritz, I., Marling, C.R., & Montani, S. (2017). Recent Themes in Case-Based Reasoning and Knowledge Discovery. FLAIRS Conference.
- R.C. Schank, Dynamic memory: A theory of reminding and learning in computers and people. Cambridge, UK: Cambridge University Press, 1982.
- R.C. Schank, Memory-based expert systems. Technical Report (# AFOSR. TR. 84-0814), Yale University, New Haven, USA, 1984.
- J. Kolodner, “Making the implicit explicit: Clarifying the principles of case-based reasoning”, Case-Based Reasoning: Experiences, Lessons & Future Directions, D.B.
- Auria, Laura and Rouslan A. Moro. “Support Vector Machines (SVM) as a Technique for Solvency Analysis.” (2008).
- [Machine Learning Algorithms: Introduction to Random Forests](https://www.dataversity.net/machine-learning-algorithms-introduction-random-forests)
- [Random Forest Analysis in ML and when to use it](https://www.newgenapps.com/blog/random-forest-analysis-in-ml-and-when-to-use-it)
- [Chapter 1 : Supervised Learning and Naive Bayes Classification — Part 1](https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-1-theory-8b9e361897d5)
- [Chapter 2 : SVM (Support Vector Machine) — Theory](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)
- [Chapter 3 : Decision Tree Classifier — Theory](https://medium.com/machine-learning-101/chapter-3-decision-trees-theory-e7398adac567)
- [Chapter 4: K Nearest Neighbors Classifier](https://medium.com/machine-learning-101/k-nearest-neighbors-classifier-1c1ff404d265)
- [Chapter 5: Random Forest Classifier](https://medium.com/machine-learning-101/chapter-5-random-forest-classifier-56dc7425c3e1)
- [Decision Trees : A Simple Way To Visualize A Decision](https://medium.com/greyatom/decision-trees-a-simple-way-to-visualize-a-decision-dc506a403aeb)
- [Decision Tree Classification : An introduction to Decision Tree Classifier](https://towardsdatascience.com/decision-tree-classification-de64fc4d5aac)
- [An Implementation and Explanation of the Random Forest in Python](https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)
- [Why Random Forest is My Favorite Machine Learning Model](https://towardsdatascience.com/why-random-forest-is-my-favorite-machine-learning-model-b97651fa3706)
- [The Naïve Bayes Classifier](https://towardsdatascience.com/the-naive-bayes-classifier-e92ea9f47523)
- [Top 4 advantages and disadvantages of Support Vector Machine or SVM](https://medium.com/@dhiraj8899/top-4-advantages-and-disadvantages-of-support-vector-machine-or-svm-a3c06a2b107)
- [Understanding Random Forests Classifiers in Python](https://www.datacamp.com/community/tutorials/random-forests-classifier-python)

[supervised-root]: fixtures/mermaid/supervised-root.png "Supervised Classifiers Based On Learner Type"
[case-based-reasoning]: fixtures/mermaid/case-based-reasoning.png "Case-Based Reasoning Flowchart"
