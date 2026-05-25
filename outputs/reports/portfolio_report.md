# League of Legends Macro Playstyle Clustering
Generated on: 2026-05-25

## 1. Executive Summary
This report details the unsupervised learning approach used to categorize map-control patterns from League of Legends minimap snapshots. By translating spatial YOLOv3 detections into structured features, we segmented game states into distinct macro playstyles.

## 2. Clustering Evaluation
Algorithm performance based on quantitative metrics:

| algorithm   |   silhouette_score |   davies_bouldin_score |   calinski_harabasz_score |
|:------------|-------------------:|-----------------------:|--------------------------:|
| kmeans      |           0.936981 |              0.0956014 |                   74039.6 |
| gmm         |           0.936981 |              0.0956014 |                   74039.6 |


## 3. Discovered Playstyles (Clusters)

### KMEANS - Cluster 0
**Tendency**: Distributed or balanced map presence

Cluster 0 represents 5529 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 0 represents 76.8% of the data (5529 samples). It is characterized by abnormally high values in: bbox_area_norm_mean (+0.17). The most distinguishing features overall compared to the global average are: image_width, image_height, mean_y, n_detections, grouping_score. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

### KMEANS - Cluster 1
**Tendency**: Distributed or balanced map presence

Cluster 1 represents 666 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 1 represents 9.2% of the data (666 samples). It is characterized by abnormally high values in: blue_base_count (+3.04), base_count (+3.04), mean_y (+1.40). The most distinguishing features overall compared to the global average are: image_width, image_height, base_count, blue_base_count, river_count. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

### KMEANS - Cluster 2
**Tendency**: Distributed or balanced map presence

Cluster 2 represents 399 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 2 represents 5.5% of the data (399 samples). It is characterized by abnormally high values in: red_side_count (+3.02), bot_lane_count (+2.81), lane_count_bot (+2.81), mean_y (+1.62). The most distinguishing features overall compared to the global average are: image_width, image_height, center_count, red_side_count, bot_lane_count. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

### KMEANS - Cluster 3
**Tendency**: Distributed or balanced map presence

Cluster 3 represents 357 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 3 represents 5.0% of the data (357 samples). It is characterized by abnormally high values in: bot_lane_count (+2.81), lane_count_bot (+2.81), mean_y (+0.39). The most distinguishing features overall compared to the global average are: image_width, image_height, bot_lane_count, lane_count_bot, river_count. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

### KMEANS - Cluster 4
**Tendency**: Distributed or balanced map presence

Cluster 4 represents 249 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 4 represents 3.5% of the data (249 samples). It is characterized by abnormally high values in: red_side_count (+3.02), mean_x (+1.15), bbox_area_norm_mean (+0.35). The most distinguishing features overall compared to the global average are: image_width, image_height, center_count, red_side_count, mean_x. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

### GMM - Cluster 0
**Tendency**: Distributed or balanced map presence

Cluster 0 represents 5529 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 0 represents 76.8% of the data (5529 samples). It is characterized by abnormally high values in: bbox_area_norm_mean (+0.17). The most distinguishing features overall compared to the global average are: image_width, image_height, mean_y, n_detections, grouping_score. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

### GMM - Cluster 1
**Tendency**: Distributed or balanced map presence

Cluster 1 represents 666 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 1 represents 9.2% of the data (666 samples). It is characterized by abnormally high values in: blue_base_count (+3.04), base_count (+3.04), mean_y (+1.40). The most distinguishing features overall compared to the global average are: image_width, image_height, base_count, blue_base_count, river_count. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

### GMM - Cluster 2
**Tendency**: Distributed or balanced map presence

Cluster 2 represents 399 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 2 represents 5.5% of the data (399 samples). It is characterized by abnormally high values in: red_side_count (+3.02), bot_lane_count (+2.81), lane_count_bot (+2.81), mean_y (+1.62). The most distinguishing features overall compared to the global average are: image_width, image_height, center_count, red_side_count, bot_lane_count. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

### GMM - Cluster 3
**Tendency**: Distributed or balanced map presence

Cluster 3 represents 357 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 3 represents 5.0% of the data (357 samples). It is characterized by abnormally high values in: bot_lane_count (+2.81), lane_count_bot (+2.81), mean_y (+0.39). The most distinguishing features overall compared to the global average are: image_width, image_height, bot_lane_count, lane_count_bot, river_count. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

### GMM - Cluster 4
**Tendency**: Distributed or balanced map presence

Cluster 4 represents 249 similar game states. Based on the prominent features, the likely macro tendency is 'Distributed or balanced map presence'. Context: Cluster 4 represents 3.5% of the data (249 samples). It is characterized by abnormally high values in: red_side_count (+3.02), mean_x (+1.15), bbox_area_norm_mean (+0.35). The most distinguishing features overall compared to the global average are: image_width, image_height, center_count, red_side_count, mean_x. The within-cluster feature variance is 0.00. 

_Note: This is based on qualitative feature analysis and spatial clustering. Direct causal impact on game outcomes requires further controlled studies._

## 4. Visualizations & Projections
*(Include saved 2D projection images and feature distribution charts here)*

---
*End of Report*