# Simulation-Based Grasping with Efficient Adaptability of Gripper Properties


![Overview](/images/fig1.jpg)


**The Proposed GAAGNet Task.** First, we collect a dataset of gripper attributes. Second, we construct a grasp learning network. Finally, we generate a grasping policy adapted to the gripper attributes.

## Introduction
Grasping detection based on machine vision is a critical yet challenging task due to the uncertainties in object and gripper attributes. While prior methods primarily focus on grippers with fixed attributes, they often fail in real-world scenarios due to mismatches between object and gripper characteristics. In this paper, we propose an innovative approach that leverages an attribute-adaptive dataset and deep neural networks to predict grasps for diverse grippers. Our key contributions include:
1) A generalizable data collection method and hybrid sampling strategy that enables the construction of grasping datasets with arbitrary gripper attributes, based on the contact dynamics between grippers and objects.
2) An attribute-adaptive robotic grasping framework trained on point clouds and annotated grasp labels, which incorporates gripper attributes and outputs the top-scoring grasp in an end-to-end manner.



## About the paper

GAAGNet is being submitted to the IEEE Transactions on Instrumentation & Measurement journal.

## About this repository

This repository provides data and code as follows.


```
    data/                   # contains grippers of different attribution
    datasets/               # contains datasets of different attribution
    scripts/                # running script of code
    logs/                   # the results of model training
    models/                 # the Network Structure of GAAGNet
```

##  Experiment results
We visualize the grasp scores predicted by the pre-trained grasp scoring network in Fig. 2. The figure clearly demonstrates that, given grippers with different attributes and grasping orientations, the network has learned to extract geometric features that are action-specific and attribution-aware. 

![Overview](/images/fig5.jpg)
Fig. 2. Visualization of the per-pixel grasp scoring network predictions for a given gripper grasp direction


We illustrate the estimated graspability scores of our method on four grippers with different attributes and present the top 5 grasps with the highest scores in Fig. 3. 
![Overview](/images/fig6.png)
Fig. 3. Visualization of per-pixel graspability scores and grasp proposals predictions on example objects