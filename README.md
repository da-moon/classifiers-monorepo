# classifiers-monorepo

- [classifiers-monorepo](#classifiers-monorepo)
  - [Outline](#outline)
  - [Dependancies](#dependancies)
  - [Supervised Classification](#supervised-classification)
  - [Overview](#overview)
  - [Lazy learner](#lazy-learner)
    - [K-Nearest Neighbor](#k-nearest-neighbor)
    - [Case-based Reasoning](#case-based-reasoning)
  - [Eager Learner](#eager-learner)
    - [Decision Tree](#decision-tree)
    - [Logistic Regression](#logistic-regression)
    - [Naive Bayes](#naive-bayes)
    - [Random Forest](#random-forest)
    - [Support Vector Machine](#support-vector-machine)
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

`KNN` Is one of the more basic algorithms and is oftenly used to benchmark more complex classifiers like Artificial Neural Networks and Support Vector Machines.

It operates by storing all instances correspond to training data points in n-dimensional space. When an unknown discrete data is received, it analyzes the closest k number of instances saved and returns the most common class as the prediction result .In case of real-valued data it returns the mean of k nearest neighbors.

There is also other variation of KNN such as:

- [Distance-weighted KNN](https://www.geeksforgeeks.org/weighted-k-nn/)
- [dual distance-weighted KNN](https://pdfs.semanticscholar.org/a128/62972be0e7e6e901825723e703117a6d8128.pdf)
- [Modified K-Nearest Neighbor (MKNN)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.545&rep=rep1&type=pdf)
- [Dynamic K-Nearest-Neighbor](https://ieeexplore.ieee.org/abstract/document/5559858)

I also found an inseresting [paper](https://ieeexplore.ieee.org/abstract/document/4406010) that offers different methodologies to pmproving K-Nearest-Neighbor for Classification.

The following are some real-world sample implementations and/or utilization of KNN:

- `[GO]` [Implementation In Pure GO](https://github.com/mervin0502/knnAlg)
- `[GO]` [Webserver For classifying images with KNN](https://github.com/arnaucube/galdric)
- `[RUST]` [KNN Node.js module written in Rust for high perfromance and parallelism](https://github.com/houtanf/Unsupervised-KNN-JS)
- `[RUST]` [KNN algorithm with GPU calculation](https://github.com/stoand/rust-gpu_kNN)
- `[PYTHON]` [Simple Python Implementation of KNN algorithm](https://github.com/iiapache/KNN)

### Case-based Reasoning

![Case-Based Reasoning Flowchart][case-based-reasoning]

In a nutshell, CBR is reasoning by remembering: previously solved problems (cases) are used to suggest solutions for novel but similar problems. Kolodner lists four assumptions about the world around us that represent the basis of the CBR approach:

- `Regularity`: the same actions executed under the same conditions will tend to have the same or similar outcomes.
- `Typicality`: experiences tend to repeat themselves.
- `Consistency`: small changes in the situation require merely small changes in the interpretation and in the solution.
- `Adaptability`: when things repeat, the differences tend to be small, and the small differences are easy to compensate for.

`case` is explained as several features describing a problem plus an outcome or a solution.They are records of real events and are excellent for justifying decisions.cases can be very rich and have elements consisting of texts, numbers, symbols, plans, multimedia. Keep in mind that they are not usually distilled knowledge. A case consists of a problem, its solution, and, typically, annotations about how the solution was derived. There are two major types of case features:

- `unindexed features`: They provide background information to users and are not predictive & not used for retrieval.
- `indexed features` : Predictive and used for retrieval.

`case-base` is a set of cases.Usually, they just flat files or relational databases. a robust case-base, containing a representative and well distributed set of cases, is the foundation for a good CBR system.

Case-based reasoning process is ca be describe as :[1]

- `Retrieve`: retrieve from memory cases relevant to solving a given target problem.
- `Reuse`: Map the solution from the previous case to the target problem. This may involve adapting the solution as needed to fit the new situation.
- `Revise`: Having mapped the previous solution to the target situation, test the new solution in the real world (or a simulation) and, if necessary, revise.
- `Retain`: After the solution has been successfully adapted to the target problem, store the resulting experience as a new case in memory.

`Advantages/Disadvantages` of CBR are summarised in the following table :

| Advantages                                          | Disadvantages                                                                                                         |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| CBR is intuitive since it mimics it’s how we work.  | Can take large storage space for all the cases                                                                        |
| CBR Development easier                              | Can take large processing time to find similar cases in case-base                                                     |
| No knowledge elicitation to create rules or methods | Cases may need to be created by hand                                                                                  |
| Systems learn by acquiring new cases through use    | Adaptation may be difficult                                                                                           |
| Maintenance is easy                                 | Needs case-base, case selection algorithm, and possibly case-adaptation algorithm.                                    |
| Justification through precedent                     | CBR may not be for you if you require the best solution or the optimum solution.                                      |
|                                                     | CBR systems generally give good or reasonable solutions. This is because the retrieved case often requires adaptation |

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

The following are some repos consisting of real-world sample implementations and/or utilization of CBR:

- `[GO]` [Implementation In GO](https://github.com/tfgordon/cbr-tool)
- `[PYTHON]` [CBR usage in finding missing values in data](https://github.com/SanjinKurelic/CaseBasedReasoning)

## Eager Learner

### Decision Tree

### Logistic Regression

### Naive Bayes

### Random Forest

### Support Vector Machine

### Artificial Neural Networks

## References

Papers used as refrence are kept under `classifiers-monorepo/fixtures/papers` directory.

- Agnar Aamodt and Enric Plaza, "Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches," Artificial Intelligence Communications 7 (1994): 1, 39-52.
- Bichindaritz, I., Marling, C.R., & Montani, S. (2017). Recent Themes in Case-Based Reasoning and Knowledge Discovery. FLAIRS Conference.
- R.C. Schank, Dynamic memory: A theory of reminding and learning in computers and people. Cambridge, UK: Cambridge University Press, 1982.
- R.C. Schank, Memory-based expert systems. Technical Report (# AFOSR. TR. 84-0814), Yale University, New Haven, USA, 1984.
- J. Kolodner, “Making the implicit explicit: Clarifying the principles of case-based reasoning”, Case-Based Reasoning: Experiences, Lessons & Future Directions, D.B.

[supervised-root]: fixtures/mermaid/supervised-root.png "Supervised Classifiers Based On Learner Type"
[case-based-reasoning]: fixtures/mermaid/case-based-reasoning.png "Case-Based Reasoning Flowchart"
