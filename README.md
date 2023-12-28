# Leaf Disease Detection and Classification with Machine Learning



### Introduction
- ROI extraction for leaf images It was accomplished by masking the green color in the lab space.Feature extraction was made through masked areas,Disease classification was performed using 5-class SVM. 


### Data Acquisition
- The images of diseased leaves were obtained from the Kaggle website: https://www.kaggle.com/datasets/emmarex/plantdisease. 
In this project, a set of images was utilized, encompassing four classes of diseased leaves and one class of healthy leaves. 25 photos were used in each class



### Accuracy with Best Parametres

- SVM was examined through K-fold analysis. C parameters were individually analyzed as 0.1, 1, 10, and 100, and Kernel types 'Linear', 'RBF' (Radial Basis Function), and 'Poly' (Polynomial) were investigated. The highest average accuracy was achieved when the C parameter was set to 100, and the Kernel type was set to 'Polynomial.' The corresponding values are as follows: Mean ROC-AUC: 0.96, Mean Precision: 0.87, Mean Recall: 0.87, Mean F1-Score: 0.87.





