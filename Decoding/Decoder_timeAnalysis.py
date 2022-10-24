import numpy as np
import os
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score

def getFeaturesAndLabels(reref, pp, epoch, target, repetitions, win_length, step):
   # Define paths 
   feature_path = 'PreprocessedData/Features/{}/'.format(pp)
   label_path = 'PreprocessedData/Labels/'
   
   # Load data   
   features = np.load(feature_path + '{}_{}_features_{}_w{}_st{}.npy'.format(pp, reref, epoch, win_length, step))
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
   dim = features.shape[-2]
   n_wind = features.shape[1]
         
   score_perms = np.zeros((dim,n_permutations))
   error_perms = np.zeros((dim,n_permutations))

   for i in range(dim):
      for perm in range(n_permutations):
      # Permute labels
         permuted_labels = np.random.permutation(labels)
         random_wind = np.random.choice(n_wind) # select a random window 
         
         X = features[:,random_wind,i,:]

         # Define Classifier, run Crossvalidation for permuted labels dataset
         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

         # Define cross validation method to use
         if cv_method == 'LeaveOneOut':
            cv = LeaveOneOut()
            CV_pred = cross_val_predict(clf, X, permuted_labels, cv=cv)
            score_perms[i,perm] = roc_auc_score(permuted_labels, CV_pred)
         
         elif cv_method == 'KFold':
            cv = StratifiedKFold(n_splits=5)
            perm_scores_env = cross_val_score(clf, X, permuted_labels, cv=cv, scoring="roc_auc", error_score='raise')
            
            # Store permuted scores, stds
            score_perms[i,perm] = perm_scores_env.mean()
            error_perms[i,perm] = perm_scores_env.std()

   if cv_method == 'LeaveOneOut':
      return score_perms

   elif cv_method == 'KFold':
      return score_perms, error_perms   

def classifier(features, labels, cv_method='LeaveOneOut'):
   dim = features.shape[-2]
   n_wind = features.shape[1]

   score_means = np.zeros((dim,n_wind))
   errors = np.zeros((dim,n_wind))
      
   # Run Classifier for each channel and for each window
   clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
   for i in range(dim):
      for w in range(n_wind):     
         X = features[:,w,i,:]
   
   # Define cross validation method to use
         if cv_method == 'LeaveOneOut':
            cv = LeaveOneOut()
            CV_pred = cross_val_predict(clf, X, labels, cv=cv)
            score_means[i,w] = roc_auc_score(labels, CV_pred)
         
         elif cv_method == 'KFold':
            cv = StratifiedKFold(n_splits=5)
            CV_scores = cross_val_score(clf, X, labels, cv=cv, scoring='roc_auc')

            score_means[i,w] = CV_scores.mean()
            errors[i,w] = CV_scores.std()
      
   if cv_method == 'LeaveOneOut':
      return score_means

   elif cv_method == 'KFold':
      return score_means, errors 
   
def estimatePVals(score_perms, score_means):
   dim = features.shape[-2]
   n_wind = features.shape[1]

   # Get p-values
   p_vals = np.zeros((dim,n_wind))

   for i in range(dim):
      for w in range(n_wind):
         p_vals[i,w] = np.count_nonzero(score_perms[i] > score_means[i,w]) / len(score_perms[i])

   # Estimate significant threshold and significant channels
   score_perms_sorted = np.sort(score_perms)
   threshold = np.zeros((dim,6))
   significant_indices = []
   for i in range(dim):
      threshold[i] = np.percentile(score_perms_sorted[i], q=[0.1, 1, 5, 95, 99, 99.9])
      significant_indices.append(np.arange(n_wind)[score_means[i] > threshold[i][3]])
   significant_indices = np.array(significant_indices, dtype=object)
   return p_vals, threshold, significant_indices

if __name__=="__main__":
   reref='ESR'
   epoch = 'long_stim'
   target = 'decision'
   repetitions = [1]
   PPs = ['p01', 'p02', 'p03', 'p04', 'p05']
   cv_method = 'LeaveOneOut'  #'KFold'
   significant_indices_epoch = 'stimulus'
   only_significant_indices = True
   permutation = True
   n_permutations = 1000
   win_length = 0.1
   step = 0.05
   
   for pp in PPs:
   # Define OutPath
      out_path = 'DecodingResults/{}/{}/{}/{}/'.format(reref, epoch, target, pp)
      if not os.path.exists(out_path):
         os.makedirs(out_path)
   
   # Get features and labels 
      features, labels, r_name = getFeaturesAndLabels(reref, pp, epoch, target, repetitions, win_length, step)
   
      if only_significant_indices == True:
      # Get significant indices
         significant_indices = np.load('DecodingResults/{}/{}/{}/{}/{}_decoder_{}_{}_permTest.npz'.format(reref, significant_indices_epoch, target, pp, pp, r_name, cv_method))['significant_indices']

      # Subselect only feautures from significant channels
         features = features[:,:,significant_indices,:]

   # Run classifier
      if cv_method == 'LeaveOneOut':
         score_means = classifier(features, labels, cv_method=cv_method)
      elif cv_method == 'KFold':
         score_means, errors = classifier(features, labels, cv_method=cv_method)

   # Save classifier scores
      if cv_method == 'LeaveOneOut':
         np.savez(out_path + '{}_decoder_{}_w{}_st{}_{}_onlySign'.format(pp, r_name, win_length, step, cv_method), 
               score_means=score_means
               )
      elif cv_method == 'KFold':
         np.savez(out_path + '{}_decoder_{}_w{}_st{}_{}_onlySign'.format(pp, r_name, win_length, step, cv_method), 
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
            np.savez(out_path + '{}_decoder_{}_w{}_st{}_{}_permTest_onlySign'.format(pp, r_name, win_length, step, cv_method), 
               score_perms=score_perms,
               p_vals=p_vals, 
               threshold=threshold, 
               significant_indices=significant_indices
               )
         elif cv_method == 'KFold':
            np.savez(out_path + '{}_decoder_{}_w{}_st{}_{}_permTest_onlySign'.format(pp, r_name, win_length, step, cv_method), 
               score_perms=score_perms,
               error_perms=error_perms,
               p_vals=p_vals, 
               threshold=threshold, 
               significant_indices=significant_indices
               )