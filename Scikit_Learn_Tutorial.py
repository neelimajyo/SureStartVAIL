#!/usr/bin/env python
# coding: utf-8

# In[7]:


from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import graphviz


# In[ ]:


from sklearn.datasets import load_iris


# In[ ]:


from sklearn import tree


# In[ ]:


# Load in our dataset
iris_data = load_iris()


# In[ ]:


# Initialize our decision tree object
classification_tree = tree.DecisionTreeClassifier()


# In[ ]:


# Train our decision tree (tree induction + pruning)
model = classification_tree.fit(iris_data.data, iris_data.target)


# In[ ]:


sudo apt-get install graphviz


# In[ ]:


dot_data = tree.export_graphviz(model, out_file=None, 
                     feature_names=iris_data.feature_names,  
                     class_names=iris_data.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  


# In[ ]:


graph = graphviz.Source(dot_data)  


# In[ ]:


graph.render("iris")


# In[ ]:




