# Partially-Observable Non-Obervable Working Memory Toolkit

An integral function of fully autonomous robots and humans is the ability to focus attention on a few relevant percepts to reach a certain goal while disregarding irrelevant percepts. Humans and animals rely on the interactions between the Pre-Frontal Cortex and the Basal Ganglia to achieve this focus, which is known as working memory. The working memory toolkit (WMtk) was developed based on a computational neuroscience model of this phenomenon with the use of temporal difference learning for autonomous systems. Recent adaptations of the toolkit either utilize abstract task representations to solve non-observable tasks or storage of past input features to solve partially-observable tasks, but not both. We propose a new model, which combines both approaches to solve complex tasks with both Partially-Observable (PO) and Non-Observable (NO) components called PONOWMtk. The model learns when to store relevant cues in working memory as well as when to switch from one task representation to another based on external feedback. The results of our experiments show that PONOWMtk performs effectively for tasks that exhibit PO properties or NO properties or both.

# Folders
- practice/: This folder contains the first few programs that were meant for learning. These programs work using Reinforcement Learning and Temporal Difference Learning
- nTL/: Our recreation of n-task learning for the use of our combined model
- working_memory/: Our recreation of the Working Memory Toolkit for the use of our combined model
- combined_mode/: Main folder that contains the PONOWMtk. Current work including dynamic threshold is being done in this folder
