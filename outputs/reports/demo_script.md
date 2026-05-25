# Live Demo Script: League of Legends Macro Playstyle Clustering

**Estimated Time:** 3-5 Minutes  
**Target Audience:** Recruiters, Technical Interviewers, Data Science Managers

---

### 1. Introduction (0:00 - 0:45)
**"Hi everyone, thanks for joining. Today I want to walk you through a project where I applied computer vision and unsupervised machine learning to solve a complex sports analytics problem in esports."**

**"The problem is this:** In *League of Legends*, teams use different overarching strategies—like grouping up for fights, split-pushing, or controlling specific neutral objectives like Baron or Dragon. But these macro playstyles are incredibly hard to quantify because they're based on spatial positioning over time, not just simple stats."

**"My goal was to automatically discover and categorize these playstyles from raw minimap data and link them to actual match outcomes."**

### 2. The Data & Pipeline (0:45 - 1:45)
**"To do this, I built an end-to-end pipeline."**

*(Show architecture diagram or briefly mention the steps)*

**"First, I used a YOLO object detection model on thousands of minimap screenshots to extract the exact (x, y) coordinates of every player on the map. But raw coordinates aren't useful on their own, so I engineered domain-specific spatial features—things like how many players are grouped in a lane, who has control of the river, and the spread of the team across the map."**

**"With these features, I ran multi-algorithm clustering—testing KMeans, Gaussian Mixture Models, and HDBSCAN—and evaluated them using Silhouette and Davies-Bouldin scores to find the most distinct map-control states."**

### 3. The Desktop Application & Results (1:45 - 3:30)
**"Instead of just presenting a static report, I built a custom PyQt desktop application to make these findings fully interactive for stakeholders like coaches or analysts. Let's take a look."**

*(Open the Desktop App)*

**"On the left, you can see the different spatial states the model discovered. Let's click on Cluster 2."** 
*(Click Cluster 2)*

**"Notice the plain-language summary here. The pipeline automatically generated this explainability report by comparing the cluster's centroid to the global average. It tells us immediately that this state is characterized by heavy top-side and Baron zone control."**

**"If we look at the Representative Samples grid below, you can see actual minimap thumbnails from real matches that perfectly match this pattern. We've successfully bridged the gap from abstract ML math back to tangible game footage."**

*(Switch to Macro Tab)*

**"But what does this mean for winning? In the Macro Tab, I linked these static map states back to high-level ranked game data. For this cluster, we can see it heavily correlates with high objective pacing—meaning teams in this state are actively accelerating the game to secure Baron."**

*(Switch to Projection Tab)*

**"Finally, the Projection tab uses t-SNE (or UMAP) to plot the entire feature space in 2D. This visually proves that our clusters are distinct and form clear behavioral pockets, giving us confidence in the model's output."**

### 4. Conclusion (3:30 - 4:00)
**"To summarize, this project demonstrates a complete, applied machine learning lifecycle. I took unstructured image data, extracted semantic features, applied unsupervised learning to find hidden patterns, and built a software tool to deliver those insights in a robust, explainable way."**

**"Everything is fully reproducible with a suite of unit tests, automated pipelines, and packaging scripts. I'd love to answer any questions you have about the architecture, the models, or the UI."**
