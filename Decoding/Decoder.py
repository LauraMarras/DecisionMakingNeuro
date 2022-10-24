import numpy as np
import os
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score

def getFeaturesAndLabels(reref, pp, epoch, target, repetitions):
   # Define paths 
   feature_path = 'PreprocessedData/Features/{}/'.format(pp)
   label_path = 'PreprocessedData/Labels/'
   
   # Load data   
   features = np.load(feature_path + '{}_{}_features_{}.npy'.format(pp, reref, epoch))
   labelsDF = pd.read_pickle(label_path + '{}_labels.pkl'.format(pp))

   # Deal with NaN values
   if target == 'decision':
      labelsDF.dropna(inplace=True)

   elif target == 'accuracy':
      labelsDF.dropna(inplace=True)
      labelsDF.reset_index(drop=True, inplace=True)

   elif target == 'stimulus_category':
      if labelsDF[target].sum() < labelsDF.shape[0]/2:
         labelsDF[target].fillna(1, inplace=True)
      else:
         labelsDF[target].fillna(0, inplace=True)

   # Extract features of interest (based on repetition selected)
   indices = labelsDF.index[labelsDF.repetition.isin(repetitions)].to_numpy()
   features = features[indices,:,:]

   # Define Labels of interest
   labels = labelsDF[target][labelsDF.repetition.isin(repetitions)].astype(int).to_numpy()
   r_name = ','.join([str(x) for x in repetitions])

   return features, labels, r_name

def permTest(features, labels, n_permutations=1000, cv_method='LeaveOneOut'):
   score_perms = np.zeros((n_permutations))
   error_perms = np.zeros((n_permutations))

   for perm in range(n_permutations):
   # Permute labels
      permuted_labels = np.random.permutation(labels)
      
      dim = features.shape[1]
      random_chan = np.random.choice(dim)
      X = features[:,random_chan,:]

      # Define Classifier, run Crossvalidation for permuted labels dataset
      clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

      # Define cross validation method to use
      if cv_method == 'LeaveOneOut':
         cv = LeaveOneOut()
         CV_pred = cross_val_predict(clf, X, permuted_labels, cv=cv)
         score_perms[perm] = roc_auc_score(permuted_labels, CV_pred)
      
      elif cv_method == 'KFold':
         cv = StratifiedKFold(n_splits=5)
         perm_scores_env = cross_val_score(clf, X, permuted_labels, cv=cv, scoring="roc_auc", error_score='raise')
         
         # Store permuted scores, stds
         score_perms[perm] = perm_scores_env.mean()
         error_perms[perm] = perm_scores_env.std()

   if cv_method == 'LeaveOneOut':
      return score_perms

   elif cv_method == 'KFold':
      return score_perms, error_perms   

def classifier(features, labels, cv_method='LeaveOneOut'):
   dim = features.shape[1]

   score_means = np.zeros((dim))
   errors = np.zeros((dim))
      
   # Run Classifier for each channel
   for i in range(dim):     
      X = features[:,i,:]
      clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
   
   # Define cross validation method to use
      if cv_method == 'LeaveOneOut':
         cv = LeaveOneOut()
         CV_pred = cross_val_predict(clf, X, labels, cv=cv)
         score_means[i] = roc_auc_score(labels, CV_pred)
      
      elif cv_method == 'KFold':
         cv = StratifiedKFold(n_splits=5)
         CV_scores = cross_val_score(clf, X, labels, cv=cv, scoring='roc_auc')

         score_means[i] = CV_scores.mean()
         errors[i] = CV_scores.std()
   
   
   if cv_method == 'LeaveOneOut':
      return score_means

   elif cv_method == 'KFold':
      return score_means, errors 
   
def estimatePVals(score_perms, score_means):
   dim = features.shape[1]
   
   # Get p-values
   p_vals = np.zeros((dim))

   for i in range(dim):
      p_vals[i] = np.count_nonzero(score_perms > score_means[i]) / len(score_perms)

   # Estimate significant threshold and significant channels
   score_perms_sorted = np.sort(score_perms)
   threshold = np.percentile(score_perms_sorted, q=[0.1, 1, 5, 95, 99, 99.9])
   significant_indices = np.arange(dim)[score_means > threshold[3]]

   return p_vals, threshold, significant_indices

if __name__=="__main__":
   reref='ESR'
   epoch = 'stimulus'
   target = 'decision'
   repetitions = [1]
   PPs = ['p01', 'p02', 'p03', 'p04', 'p05']
   cv_method = 'LeaveOneOut'  #'KFold'
   permutation = True
   n_permutations = 1000
   
   for pp in PPs:
   # Define OutPath
      out_path = 'DecodingResults/{}/{}/{}/{}/'.format(reref, epoch, target, pp)
      if not os.path.exists(out_path):
         os.makedirs(out_path)

   # Get features and labels 
      features, labels, r_name = getFeaturesAndLabels(reref, pp, epoch, target, repetitions)

   # Run classifier
      if cv_method == 'LeaveOneOut':
         score_means = classifier(features, labels, cv_method=cv_method)
      elif cv_method == 'KFold':
         score_means, errors = classifier(features, labels, cv_method=cv_method)

   # Save classifier scores
      if cv_method == 'LeaveOneOut':
         np.savez(out_path + '{}_decoder_{}_{}'.format(pp, r_name, cv_method), 
            score_means=score_means
            )
      elif cv_method == 'KFold':
         np.savez(out_path + '{}_decoder_{}_{}'.format(pp, r_name, cv_method), 
            score_means=score_means,
            errors=errors
            )
      
      if permutation == True:
      # Run permutation test
         if cv_method == 'LeaveOneOut':
            score_perms = permTest(features, labels, n_permutations=n_permutations, cv_method=cv_method)
         elif cv_method == 'KFold':
            score_perms, error_perms = permTest(features, labels, n_permutations=n_permutations, cv_method=cv_method)

      # Estimate significance threshold and p_vals
         p_vals, threshold, significant_indices = estimatePVals(score_perms, score_means)

      # Save permTest data
         if cv_method == 'LeaveOneOut':
            np.savez(out_path + '{}_decoder_{}_{}_permTest'.format(pp, r_name, cv_method), 
               score_perms=score_perms,
               p_vals=p_vals, 
               threshold=threshold, 
               significant_indices=significant_indices
               )
         elif cv_method == 'KFold':
            np.savez(out_path + '{}_decoder_{}_{}_permTest'.format(pp, r_name, cv_method), 
               score_perms=score_perms,
               error_perms=error_perms,
               p_vals=p_vals, 
               threshold=threshold, 
               significant_indices=significant_indices
               )