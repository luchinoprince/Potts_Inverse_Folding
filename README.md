# Potts_Inverse_Folding
Repository reproducing some of the codes of the paper "Uncovering sequence diversity from a known protein structure".

The repository has different folders which allow for the different steps necessary to train or test the model as shown in the paper. We first begin by the Dataset creation. 

## Dataset creation

This section will require roughly $120Gbs$ of memory while the execution time depends on the CPU resources avaiable to [MMseqs2](https://github.com/soedinglab/MMseqs2). We ran it on _intel I9-13900K/KF 5.8 Ghz_, and this step took roughly a day to complete. The codes can be found in the folder *data_creation*. To run the codes one needs to have downloaded both the _Uniref50_ dataset which can be found on the [UniProt website](https://www.uniprot.org/help/downloads) and the [CATH](http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/) 4.2 40% non redundant dataset. Also one needs to download the [MMseqs2](https://github.com/soedinglab/MMseqs2) and [esm](https://github.com/facebookresearch/esm) libraries.

Once this is done can first run the bash file _create_msas.sh_ to create all the necessary MSA, then one can split them into the train test split outlined in the paper running the python code _train_test_split_. Also to use the current dataloader one has to run  the notebook _get_numerical_MSA.ipynb_ to get all the MSA's in numerical format, so that they are ready to use by the model and don't have to be converted at every update step of the training. To get the **esm1f** pretrained encodings to feed to one of our Potts decoders, one need to run the code _get_encodings_. When running these codes the location of the different input and output directories have to be properly specified. 
