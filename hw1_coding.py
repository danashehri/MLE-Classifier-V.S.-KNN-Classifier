# **COMS HW1 4771 (Spring 2022)**


import scipy.io
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
data = scipy.io.loadmat('digits.mat')

X = data['X']
Y = data['Y']
X = X.astype(float)
Y = Y.astype(float)
thresholder = VarianceThreshold(12000)
X = thresholder.fit_transform(X)

print("Loading Data Completed")
class ProbabilisticClassifer:

  def train(self, x_train, y_train):
    
    #class priors
    self.p0, self.p1, self.p2, self.p3, self.p4, self.p5, self.p6, self.p7, self.p8, self.p9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    for y in y_train:
      if y == 0:
        self.p0 += 1
      elif y == 1:
        self.p1 += 1
      elif y == 2:
        self.p2 += 1
      elif y == 3:
        self.p3 += 1
      elif y == 4:
        self.p4 += 1
      elif y == 5:
        self.p5 += 1
      elif y == 6:
        self.p6 += 1
      elif y == 7:
        self.p7 += 1
      elif y == 8:
        self.p8 += 1
      elif y == 9:
        self.p9 += 1

    self.samples = y_train.size 
    
    #class conditionals
    xy_list = list(zip(x_train, y_train))

    #find data through which y = 0
  
    xdata_y0 = []
    for item in xy_list:
      if item[1] == 0: #labelled as class 0
        xdata_y0.append(np.expand_dims(item[0], axis=1))
    
    #calc MLE for mean
  
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y0:
      sum_mu += xi
    mu0 = sum_mu/self.p0 

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y0:
      sum_cov += (xi - mu0)*((xi - mu0).transpose()) 
    
    self.sigma0 = sum_cov/self.p0 
    self.mu0 = mu0
    
    #################################################################################
    #find data through which y = 1
    xdata_y1 = []
    for item in xy_list:
      if item[1] == 1:
        xdata_y1.append(np.expand_dims(item[0], axis=1))

    #calc MLE for mean
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y1:
      sum_mu += xi
    mu1 = sum_mu/self.p1

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y1:
      sum_cov += (xi - mu1)*(xi - mu1).transpose()
    self.sigma1 = sum_cov/self.p1
    self.mu1 = mu1
    
    #################################################################################
    #find data through which y = 2
    xdata_y2 = []
    for item in xy_list:
      if item[1] == 2:
        xdata_y2.append(np.expand_dims(item[0], axis=1))

    #calc MLE for mean
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y2:
      sum_mu += xi
    mu2 = sum_mu/self.p2

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y2:
      sum_cov += (xi - mu2)*(xi - mu2).transpose()
    self.sigma2 = sum_cov/self.p2
    self.mu2 = mu2

    #################################################################################
    #find data through which y = 3
    xdata_y3 = []
    for item in xy_list:
      if item[1] == 3:
        xdata_y3.append(np.expand_dims(item[0], axis=1))

    #calc MLE for mean
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y3:
      sum_mu += xi
    mu3 = sum_mu/self.p3

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y3:
      sum_cov += (xi - mu3)*(xi - mu3).transpose()
    self.sigma3 = sum_cov/self.p3
    self.mu3 = mu3
     
    #################################################################################
    #find data through which y = 4
    xdata_y4 = []
    for item in xy_list:
      if item[1] == 4:
        xdata_y4.append(np.expand_dims(item[0], axis=1))

    #calc MLE for mean
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y4:
      sum_mu += xi
    mu4 = sum_mu/self.p4

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y4:
      sum_cov += (xi - mu4)*(xi - mu4).transpose()
    self.sigma4 = sum_cov/self.p4
    self.mu4 = mu4
    
    #################################################################################
    #find data through which y = 5
    xdata_y5 = []
    for item in xy_list:
      if item[1] == 5:
        xdata_y5.append(np.expand_dims(item[0], axis=1))

    #calc MLE for mean
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y5:
      sum_mu += xi
    mu5 = sum_mu/self.p5

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y5:
      sum_cov += (xi - mu5)*(xi - mu5).transpose()
    self.sigma5 = sum_cov/self.p5
    self.mu5 = mu5
    
    #################################################################################
    #find data through which y = 6
    xdata_y6 = []
    for item in xy_list:
      if item[1] == 6:
        xdata_y6.append(np.expand_dims(item[0], axis=1))

    #calc MLE for mean
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y6:
      sum_mu += xi
    mu6 = sum_mu/self.p6

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y6:
      sum_cov += (xi - mu6)*(xi - mu6).transpose()
    self.sigma6 = sum_cov/self.p6
    self.mu6 = mu6
    
    #################################################################################
    #find data through which y = 7
    xdata_y7 = []
    for item in xy_list:
      if item[1] == 7:
        xdata_y7.append(np.expand_dims(item[0], axis=1))

    #calc MLE for mean
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y7:
      sum_mu += xi
    mu7 = sum_mu/self.p7

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y7:
      sum_cov += (xi - mu7)*(xi - mu7).transpose()
    self.sigma7 = sum_cov/self.p7
    self.mu7 = mu7
    
    #################################################################################
    #find data through which y = 8
    xdata_y8 = []
    for item in xy_list:
      if item[1] == 8:
        xdata_y8.append(np.expand_dims(item[0], axis=1))

    #calc MLE for mean
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y8:
      sum_mu += xi
    mu8 = sum_mu/self.p8

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y8:
      sum_cov += (xi - mu8)*(xi - mu8).transpose()
    self.sigma8 = sum_cov/self.p8
    self.mu8 = mu8

    #################################################################################
    #find data through which y = 9
    xdata_y9 = []
    for item in xy_list:
      if item[1] == 9:
        xdata_y9.append(np.expand_dims(item[0], axis=1))

    #calc MLE for mean
    sum_mu = np.zeros([x_train.shape[1], 1])
    for xi in xdata_y9:
      sum_mu += xi
    mu9 = sum_mu/self.p9

    #calc MLE for cov
    sum_cov = np.zeros([x_train.shape[1], x_train.shape[1]])
    for xi in xdata_y9:
      sum_cov += (xi - mu9)*(xi - mu9).transpose()
    self.sigma9 = sum_cov/self.p9
    self.mu9 = mu9
     
    #################################################################################
  
  def predict(self, X_test):
    predictions = []
    for x in X_test:
    #find the class with the highest probability
      y_0 = multivariate_normal.pdf(x,mean=self.mu0.transpose()[0], cov=self.sigma0)*(self.p0/self.samples)
      y_1 = multivariate_normal.pdf(x,mean=self.mu1.transpose()[0], cov=self.sigma1)*(self.p1/self.samples)
      y_2 = multivariate_normal.pdf(x,mean=self.mu2.transpose()[0], cov=self.sigma2)*(self.p2/self.samples)
      y_3 = multivariate_normal.pdf(x,mean=self.mu3.transpose()[0], cov=self.sigma3)*(self.p3/self.samples)
      y_4 = multivariate_normal.pdf(x,mean=self.mu4.transpose()[0], cov=self.sigma4)*(self.p4/self.samples)
      y_5 = multivariate_normal.pdf(x,mean=self.mu5.transpose()[0], cov=self.sigma5)*(self.p5/self.samples)
      y_6 = multivariate_normal.pdf(x,mean=self.mu6.transpose()[0], cov=self.sigma6)*(self.p6/self.samples)
      y_7 = multivariate_normal.pdf(x,mean=self.mu7.transpose()[0], cov=self.sigma7)*(self.p7/self.samples)
      y_8 = multivariate_normal.pdf(x,mean=self.mu8.transpose()[0], cov=self.sigma8)*(self.p8/self.samples)
      y_9 = multivariate_normal.pdf(x,mean=self.mu9.transpose()[0], cov=self.sigma9)*(self.p9/self.samples)
      
      probs = [y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9]
      v = max(probs,key=lambda x:float(x))      

      if v == y_0:
        predictions.append(0)
      elif v == y_1:
        predictions.append(1)
      elif v == y_2:
        predictions.append(2)
      elif v == y_3:
        predictions.append(3)
      elif v == y_4:
        predictions.append(4)
      elif v == y_5:
        predictions.append(5)
      elif v == y_6:
        predictions.append(6)
      elif v == y_7:
        predictions.append(7)
      elif v == y_8:
        predictions.append(8)
      else:
        predictions.append(9)
      
    return np.array(predictions)

"""## Q4ii"""

class KNNClassifier:
  def __init__(self, n_neighbors):  
        self.n_neighbors = n_neighbors
        self.x_train = None
        self.y_train = None

  def euclid_dist(self, a, b):
    a = np.expand_dims(a, axis=0)
    b = np.expand_dims(b, axis=0)
    d = distance.euclidean(b, a)
    return d

  def L1_norm(self, a, b):
    d = np.linalg.norm((b-a), ord=1)
    return d

  def inf_norm(self, a, b):
    d = np.linalg.norm((b-a), ord=math.inf)
    return d

  def fit(self, x_train, y_train):
    self.x_train = x_train
    self.y_train = y_train
    self.zipped = list(zip(self.x_train, self.y_train))

  def comp(self, test_data, norm):
  
    distances = []
    
    for i,x in enumerate(self.x_train):
      x = np.expand_dims(x, axis=0)
      if norm == 2:
        dist = self.euclid_dist(x, test_data)
      elif norm == 1:
        dist = self.L1_norm(x, test_data)
      elif norm == math.inf:
        dist = self.inf_norm(x, test_data)
      
      distances.append((x, dist, y_train[i][0])) 

    distances.sort(key=lambda item: item[1]) 

    neighbors = [] #labels of the data points

    for k in range(self.n_neighbors):
      neighbors.append(distances[k][2])
    
    votes = Counter(neighbors)

    if len(list(set(votes.values()))) == 1:
      #tie
      majority_vote = neighbors[0]
    else:
      majority_vote = max(set(neighbors), key=neighbors.count)

    return majority_vote

  def predict(self, x_test, norm=2):
    y_pred = []
    for xt in x_test:
      xt = np.expand_dims(xt, axis=0)
      vote = self.comp(xt, norm)
      y_pred.append(vote)

    return np.array(y_pred)

"""
K-Fold Cross Validation Analysis With K = 5
"""
answer = input("Run plot for Q4(iii) and Q4(iv)? Please Type Y/N")
if answer == 'Y':
  print("Running K-Fold Cross Validation (K=5) on five classifiers at once:-")
  print("----> Q4(iii): KNNClassifeier(3) and ProbabilisticClassifer()...")
  print("----> Q4(iv) : on three KNNClassifeier(3) each using L2, L1, and L_∞ respectively...")

  kf = KFold(n_splits=5)
  kf.get_n_splits(X)
  KNN_scores = []
  BPC_scores = []
  L2_scores = []
  L1_scores = []
  Linf_scores = []

  counter = 0
  for train_index, test_index in kf.split(X):   
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
      
      knn = KNNClassifier(3)
      knn.fit(X_train, y_train)
      y_pred = knn.predict(X_test)
      knn_score = accuracy_score(y_test,y_pred)
      KNN_scores.append(knn_score)

      model = ProbabilisticClassifer()
      model.train(X_train, y_train)
      y_pred = model.predict(X_test)
      model_score = accuracy_score(y_test,y_pred)
      BPC_scores.append(model_score)

      #Evaluating KNN Classifers with different distance metric
      model1 = KNNClassifier(3)
      model1.fit(X_train, y_train)
      y_pred = model1.predict(X_test, norm=2)
      score = accuracy_score(y_test,y_pred)
      L2_scores.append(score)

      model2 = KNNClassifier(3)
      model2.fit(X_train, y_train)
      y_pred = model2.predict(X_test, norm=1)
      score = accuracy_score(y_test,y_pred)
      L1_scores.append(score)

      model3 = KNNClassifier(3)
      model3.fit(X_train, y_train)
      y_pred = model3.predict(X_test, norm=math.inf)
      score = accuracy_score(y_test,y_pred)
      Linf_scores.append(score)

      counter += 1
      print("Fold ", counter, "done.")


  print("Figure 1: KNN Classifier(3) V.S. Probabilistic Classifier")
  #Ploting 5-Fold Cross Validation Results
  folds = [1,2,3,4,5]
  plt.plot(folds, KNN_scores)
  plt.plot(folds, BPC_scores)
  plt.xlabel("K-Folds")
  plt.ylabel("Accuracy Score")
  plt.legend(["KNN Classifier", "Probabilistic Classifier"])
  plt.show()

  print("Figure 2: (KNN Classifiers) L2 norm V.S. L1 norm V.S. L_∞ norm ----")
  #Plotting aacuracy scores against number of folds
  plt.plot(folds, L2_scores)
  plt.plot(folds, L1_scores)
  plt.plot(folds, Linf_scores)

  plt.xlabel("K-Folds")
  plt.ylabel("Accuracy Score")
  plt.legend(["L2 Norm", "L1 Norm", "L_inf Norm"])
  plt.show()
elif answer == 'N':
    pass
else:
  print("Please type either Y or N")
  exit()

answer2 = input("Run code for plotting KNN Classifiers with different K nearest neighbors? Y/N")
if answer2 == 'Y':
  print("---- Analysis on why I chose K nearest neighbors to be 3 for my KNN classifier ----")
  print("Running KNNClassifier(1), KNNClassifier(3), KNNClassifier(5) three times each using a different distance metrics...")
  x_train = X[:8000]
  y_train = Y[:8000]
  x_test = X[8000:]
  y_test = Y[8000:]

  """## $L_2$ Norm"""
  print("Computing L2 norm accuracy scores...")
  L2_scores = []
  knn = KNNClassifier(1)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test, norm=2)
  score = accuracy_score(y_test,y_pred)
  L2_scores.append(score)

  knn = KNNClassifier(3)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test, norm=2)
  score = accuracy_score(y_test,y_pred)
  L2_scores.append(score)

  knn = KNNClassifier(5)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test, norm=2)
  score = accuracy_score(y_test,y_pred)
  L2_scores.append(score)
  print("Done with L2 norm")

  """## $L_1$ norm"""
  print("Computing L1 norm accuracy scores..")
  L1_scores = []
  knn = KNNClassifier(1)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test, norm=1)
  score = accuracy_score(y_test,y_pred)
  L1_scores.append(score)

  knn = KNNClassifier(3)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test, norm=1)
  score = accuracy_score(y_test,y_pred)
  L1_scores.append(score)

  knn = KNNClassifier(5)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test, norm=1)
  score = accuracy_score(y_test,y_pred)
  L1_scores.append(score)
  print("Done with L1 norm")

  """## $L_∞$ norm"""
  print("Computing L_∞ norm accuracy scores..")
  Linf_scores = []
  knn = KNNClassifier(1)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test, norm=math.inf)
  score = accuracy_score(y_test,y_pred)
  Linf_scores.append(score)

  knn = KNNClassifier(3)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test, norm=math.inf)
  score = accuracy_score(y_test,y_pred)
  Linf_scores.append(score)

  knn = KNNClassifier(5)
  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test, norm=math.inf)
  score = accuracy_score(y_test,y_pred)
  Linf_scores.append(score)
  print("Done with L_∞ norm")
  print("Figure 3: Plotting number of K nearest neighbors against distance metrics accuracy scores")
  knn = [1, 3, 5]

  plt.plot(knn, L2_scores)
  plt.plot(knn, L1_scores)
  plt.plot(knn, Linf_scores)
  plt.xlabel("K Values for KNN Classifier")
  plt.ylabel("Accuracy Score")
  plt.legend(["L2 Norm", "L1 Norm", "L_inf Norm"])
  plt.show()
elif answer2 == 'N':
  print("end of code")
  exit()
else:
  print("Please type either Y or N")
  exit()





