# An Entity Alignment Model for Echinococcosis Knowledge Graph

## Dependencies
- python: 3.7
- pytorch: 1.10.0
- transformers: 4.2.2
- tqdm: 4.56.0


### Datasets

- ECHI: The data is sourced from the echinococcosis molecule and anti echinococcosis drug website EchiDB (https://echidb.ahu-bioinf-lab.com/?page=home) built in our laboratory 

```
cd src
python ECHIPreprocess.py
```

### Pre-trained Models

The pre-trained models of _transformers_ library can be downloaded from https://huggingface.co/models. 
We use [biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2) in our experiments. 

Please put the downloaded pre-trained models into _"EKGEA/pre_trained_models"_. 


##  Run

```shell
cd src
python ECHIPreprocess.py
python EKGEAPreprocess.py
python EKGEATrain.py
```

