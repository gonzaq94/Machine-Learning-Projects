# UE_ML_project

  <img src="https://www.imt-atlantique.fr/sites/default/files/logo_mt_0_0.png" WIDTH=200 HEIGHT=140 >

This is the mini project for the UE Machine Learning 2019.

### Authors:

<ul>
  <li>Tales Marra</li>
  <li>Santiago Agudelo</li>
  <li>Gonzalo Quintana</li>
  <li>Wang Lei</li>
</ul>

### Environement 

In order to be able to execute the following steps, you will need to create a Python 3 Environement.
This can be done by:

```bash
pip install requirements.txt
```

### Directory Structure

```
UE_ML_project
│   README.md 
│   requirements.txt
|   main.py   <- main python file to execute
├── data_classification
│   │   data_banknote_authentication.txt          <- banknote data
│   │   data_banknote_authentication_info.txt     <- link to raw data
│   │   kidney_disease.csv                        <- kidney_disease data   
├── images                                        <- images confusion matrices
│   models.py                                     <- contains the models
│   prepro.py                                     <- contains the preprocessing
│   outils.py                                     <- contains useful functions
```

### Execution 

```bash
python3 main.py --dataset <DATASET_NAME> --models <LIST OF MODELS>
```
Where dataset can be either _kidney-disease_ or _bank-note_.

An additional parameter is _--do_pca_, if you want to have a PCA analysis performed. 

The list of available models are: _svm_model_, _knn_model_, _neural_network_, _decision_tree_model_, _bayes_model_. They have to be passed separated by comma.
