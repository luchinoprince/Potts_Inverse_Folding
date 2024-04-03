# Potts_Inverse_Folding
Repository reproducing some of the codes of the paper "Uncovering sequence diversity from a known protein structure".

The repository has different folders which allow for the different steps necessary to train or test the model as shown in the paper. We first begin by the Dataset creation. 

## Dataset creation

This section will require roughly $120Gbs$ of memory while the execution time depends on the CPU resources avaiable to [MMseqs2](https://github.com/soedinglab/MMseqs2). We ran it on _intel I9-13900K/KF 5.8 Ghz_, and this step took roughly a day to complete. The codes can be found in the folder *data_creation*. To run the codes one needs to have downloaded both the _Uniref50_ dataset which can be found on the [UniProt website](https://www.uniprot.org/help/downloads) and the [CATH](http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/) 4.2 40% non redundant dataset. Also to create the dataset one needs to download also the  [MMseqs2](https://github.com/soedinglab/MMseqs2) and [esm](https://github.com/facebookresearch/esm) library.

To generate the MSA's for the different structures one needs to run firstly the code the *create_msas.sh* file, followed by the code *train_test_split* to get the different dataset outlined in the paper. Finally to get the *esm1f* pretrained encodings for our different potts decoder one needs to run the code *get_encodings.py*. When running the above codes one has to be carefull about the locations of the different necessary folders/packages.

## Training of models

To train the two different models one needs to run the respective .py files in the *train* folder; the codes should be commented enough to be readable, hence one can set the parameters manually inside the code. These dodes call the codes from the auxiliary folders *model* and *utils*; the former have the implementations of the different models implemented in the paper, as well as the code to compute the psuedolikelihood, while the latter have auxiliary files necessary for the different trainings/testing for the different models. 

## Testing

