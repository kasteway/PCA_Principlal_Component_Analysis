# PCA_Principlal_Component_Analysis

### Summary:

Principal Component Analysis (PCA) in the context of machine learning is a powerful technique used for dimensionality reduction. It's particularly useful when dealing with high-dimensional data. PCA is like a filter that takes in a lot of detailed and complex information and simplifies it into the key points that matter most, making it easier to understand and make decisions. PCA in machine learning is a technique for simplifying datasets with many features into a more manageable form, without losing significant information. This simplification helps in efficient data processing, analysis, and visualization, and is particularly valuable in scenarios where the dataset has a large number of features.

Imagine a tech website that reviews smartphones. Each phone is rated on multiple features: battery life, camera quality, screen size, and price. Now, suppose you're trying to decide which smartphone to buy, but you find it overwhelming to compare all these features across different models.

Here's where PCA comes in handy. PCA is like a smart assistant that helps simplify this decision-making process. It does this in two main steps:

1. Combining Features: Instead of looking at each feature separately, PCA combines them into new categories that are easier to compare. For example, it might combine battery life and price into one category (let's call it "Value for Money"), and camera quality and screen size into another (let's call it "User Experience"). These new categories are the 'principal components.'

2. Highlighting What's Important: PCA then figures out which of these new categories (principal components) are the most important in distinguishing between the smartphones. Maybe "User Experience" is more important than "Value for Money" for high-end smartphones, while for budget phones, it's the opposite.

In our smartphone example, PCA helps you focus on what's really important by simplifying the complex information. Instead of juggling four different features across multiple phones, you now have maybe just two principal components to consider, making your decision easier.

#### PCA In Action:

1. Dimensionality Reduction:

Imagine you have a dataset with many features (variables). In machine learning, each feature is a dimension. High-dimensional datasets can be complex to work with and can suffer from the "curse of dimensionality" (where models become less effective as the number of features increases).
PCA reduces the number of dimensions without losing much information. It transforms the original features into a new set of features, which are fewer in number and are called principal components.

2. Principal Components:

These are new, synthetic variables created by PCA. Each principal component is a linear combination of the original variables.
The first principal component captures the most variance (information) in the dataset, the second captures the second most, and so on.
These components are uncorrelated, meaning they don't overlap in the information they convey.

3. Application in Machine Learning:

In machine learning, PCA is often used in the preprocessing stage. Before building a model, you might use PCA to simplify your dataset, making it easier to process and analyze.
It's especially useful for visualization. For example, high-dimensional data can be projected onto two or three principal components, making it possible to visualize and understand complex patterns in the data.

Example: Image Recognition

Imagine a machine learning model designed to recognize faces in images. Each image is composed of thousands of pixels, each pixel being a feature. Processing all these pixels directly would require immense computational resources.
PCA can be used to reduce the number of pixels that the model needs to consider. It does this by identifying the pixels that carry the most information about the differences between images. The model then only needs to focus on these informative pixels, significantly simplifying the task.



---

### Steps used in PCA: Through these steps, PCA allows you to simplify complex datasets while retaining the most meaningful aspects of your original data.

- Get original data - Collect and assemble your data in a structured format. This dataset should consist of various features (variables) that you want to analyze and reduce in dimensionality.
  
- Calculatue Covariance Matrix (np.cov(scaled_X, rowvar=False)) - Compute the covariance matrix to understand how each pair of features in the dataset varies with each other. The covariance matrix helps in identifying the relationships and dependencies among the features.
  
- Calculate Eigen Vectors (eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix) )- Find the eigen vectors of the covariance matrix. These eigen vectors are essentially directions along which the data varies the most. They are crucial for understanding the underlying structure of the data.
  
- Sort EigenVectors by Eigen Values (eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix) )- Arrange the eigen vectors in descending order according to their corresponding eigen values. Eigen values give the magnitude of variance captured by each eigen vector, so sorting them helps in understanding the relative importance of each direction of variance.
  
- Choose N largest Eigen Values - Select the top N eigen vectors corresponding to the largest N eigen values. This step is about deciding how many principal components you want to keep based on how much variance you want to retain in the data.
  num_components = 2 -> sorted_key = np.argsort(eigen_values)[::-1][:num_components] -> eigen_values, eigen_vectors = eigen_values[sorted_key], eigen_vectors[:, sorted_key]
  
- Project original data onto Eigen Vectors (principal_components = np.dot(scaled_X,eigen_vectors))- Transform the original data onto the new axes defined by the selected eigen vectors. This step effectively reduces the dimensions of your data to those that capture the most variance, as per the chosen eigen vectors.
  
- Visualize the Principal Components -> plt.scatter(principal_components[:,0],principal_components[:,1])



---


### Advantages & Disadvantages:

#### Advantages:
- Reduces Overfitting: By lowering the number of features, PCA can help reduce the chances of overfitting in a model.

- Improves Visualization: For high-dimensional data, PCA can reduce dimensions to 2 or 3 principal components, making it possible to visualize and understand complex data sets.

- Simplifies Data: PCA simplifies the complexity in high-dimensional data by transforming it into fewer dimensions, which are easier to analyze and work with.

- Removes Correlated Features: In many datasets, features are correlated; PCA helps in removing this multicollinearity by transforming the original data into uncorrelated principal components.

- Enhances Algorithm Performance: Many machine learning algorithms perform better or compute faster when the number of input features is reduced.

- Data Compression: PCA can act as a tool for data compression, reducing storage and memory requirements.



#### Disadvantages:
- Loss of Information: While reducing dimensions, some information is inevitably lost. This might affect the performance of some machine learning models, especially if the removed components contained important information.

- Interpretability: The principal components are linear combinations of the original variables and often do not have a meaningful interpretation. This makes it hard to understand the role of the original features.

- Standardization Required: PCA is affected by the scale of the features, so it requires feature scaling (like standardization) before applying it. This could be a drawback if the scales of the original features carry important information.

- Not Effective for Non-Linear Relationships: PCA is a linear method. It doesn't work well if the data has complex, non-linear relationships.

- Sensitive to Outliers: PCA is sensitive to outliers in the data. Outliers can skew the results significantly, leading to incorrect conclusions.

- Choosing the Number of Components: Deciding the number of principal components to keep can be subjective and may not always be straightforward.


---



 

---

### Data:

The data set used for this analysis comes from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom). This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525).  Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended.  This latter class was combined with the poisonous one.  The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.

- Dataset Characteristics -> Multivariate

- Subject Area -> Biology

- Associated Tasks -> Classification

- Feature Type -> Categorical

- Instances -> 8124

- Features -> 22



---

### Tips:

- Must scale data -> Standardize to shift data mean to 0 (from sklearn.preprocessing import StandardScaler)
- 



---

### Tips:

- AdaBoostClassifier [Scikit Learn AdaBoostClassifier]([https://archive.ics.uci.edu/dataset/73/mushroom](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).
- base_estimator -> default is none meaning it will use "DecisionTreeClassifier" initialized with max_depth = 1 as the stump (if desired, another estimator can be manually selected)
- 
---
