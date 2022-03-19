# exp_on_iRNA5hmCdata

We performed the i5hmCVec on the dataset from WeakRM. 

- ```'data'``` contains the dataset from iRNA5hmC.
- ```'Feature'``` contains the digital features of dataset from iRNA5hmC with feature encoding method proposed in this study.
- ```'ger_feature.py'``` is used to generated the feature proposed in i5hmCvec on the dataset from iRNA5hmC.
- ```'model_5_fold.py'``` is used to evaluted the performance of i5hmCvec with different c, gamma in SVM. This Script is used as a parameter tuning Script.
- ```'model_10times_5fold.py'``` is used to performed 10 times 5-fold cross-validation with optimized parameters from ```'model_5_fold.py'```.
- ```'res'``` is stored as a results of ```'train_model.py'``` and ```'model_10times_5fold.py'```.

