# i5hmCVec

In this work, we propose a computational method for identifying the 5hmC modified regions using machine learning algorithms. We applied a sequence feature embedding method based on the dna2vec algorithm to represent the RNA sequence. The ```'data'``` folder stores the data needed in the experiment. The ```'src'``` folder contains the code used in the method. The ```'feature'``` folder contains the feature vector used in the method. The ```'exp_on_weakRMdata'```folder stores the program applied for comparing i5hmCVec with WeakRM. The ```'exp_on_iRNA5hmCdata'```folder stores the program applied for comparing i5hmCVec with iRNA5hmC.

# Run

To run our supplied program, you need to configure the python 3.8 environment.

Run our code with the following command:

+ Generate the feature vector 

```
python ./src/ger_feature.py --s
```

```
--s: species used for generating feature. dm represents for drosophila melanogaster. mouse represents for mouse
```

+ Generate the performace on the SVM

```
python ./src/train_svm.py --s --K --c --g 
```

```
--s: species used for generating feature. dm represents for drosophila melanogaster. mouse represents for mouse
--K: the k-mer embeddings used for assemblying feature vector
--c: the cost parameter in SVM
--g: the parameter in the RBF kernel function
```

+ Generate the performace on the MLP

```
python ./src/train_mlp.py --s --K --lr --epoch 
```

```
--s: species used for generating feature. dm represents for drosophila melanogaster. mouse represents for mouse
--K: the k-mer embeddings used for assemblying feature vector
--lr: the learning rate of the SGD optimizer
--epoch: the number of epochs to train the model
```

- Generate the performace on the C4.5

C4.5 is implemented by weka.

```
train_c45.java is used for train c4.5.
'species' variable in 'main' denotes for species. dm represents for drosophila melanogaster. mouse represents for mouse
'ks' list in 'main' denotes for the k-mer embeddings used for assemblying feature vector
'C' variable in 'main' denotes for the confidence threshold for pruning
```

  # Citation
**Hang-Yu Liu**, Pu-Feng Du*, i5hmCVec: Identifying 5-Hydroxymethylcytosine sites of *Drosophila* RNA using sequence feature embeddings. _Frontiers in Genetics_ (2022) (Accepted)
