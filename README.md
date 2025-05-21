# Potts_Inverse_Folding
Repository reproducing some of the codes of the paper "Uncovering sequence diversity from a known protein structure".
## Main packages required and versions used
1. torch                     2.1.0
2. torch-geometric           2.3.1
3. torch-scatter             2.1.2+pt21cu121
4. torch-sparse              0.6.18+pt21cu121
5. torchaudio                2.1.0
6. torchvision               0.16.0
7. tensorboard               2.14.1
8. seaborn                   0.13.0
9. scipy                     1.11.3
10. pyhmmer                   0.10.4
11. pandas                    1.5.3
12. optuna                    3.4.0
13. numpy                     1.24.4
14. biotite                   0.38.0
15. bio                       1.5.9
16. python                    3.10.12


The repository has different folders which allow for the different steps necessary to train or test the model as shown in the paper.
## Dataset creation (Currently resolving a reported bug)

The below instructions have been reported to raise some dimension errors in the code. The bugs are on the process of being resolved, in the mean time anyone can download the full zipped data at the following link: https://www.dropbox.com/scl/fi/bac5zrwjnlskdqo7eg9pm/IF.tar?rlkey=gtusz8kxhghgndd6jw57j49wa&st=9qrs6boc&dl=0.

This section will require roughly $120Gbs$ of memory while the execution time depends on the CPU resources avaiable to [MMseqs2](https://github.com/soedinglab/MMseqs2). We ran it on _intel I9-13900K/KF 5.8 Ghz_, and this step took roughly a day to complete. The codes can be found in the folder *data_creation*. The requirements to run the code are:
- Download the _Uniref50_ dataset which can be found on the [UniProt website](https://www.uniprot.org/help/downloads).
- Download the [CATH](http://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/) 4.2 40% non redundant dataset. 
- Install the the [MMseqs2](https://github.com/soedinglab/MMseqs2) library and [esm](https://github.com/facebookresearch/esm) repository.

Once this is done, and the necessary dataset and repositiory are placed in paths compatible with the following files one has to:
1. run the bash file _create_msas.sh_ to create all the necessary MSA for the different structures
2. run the python code _train_test_split.py_ to split the different MSAs into the train test split outlined
3. Run the python notebook _get_numerical_MSA.ipynb_ to get all the MSA's in numerical format, so that they are ready to use by the model and don't have to be converted at every update step of the training. 
4. Run the python notebook _get_encodings.ipynb_ to get the **esm1f** pretrained encodings to feed to one of our Potts decoders; 

## Training models

In the training folder one can find two files, to train respectively the standard pairwise potts model and the autoregressive potts model. If the steps detailled in the **Dataset creation** section have been performed properly, these code should run directly(provided one has all the necessary libraries like pytorch, pytorch-geometric etc).

We trained our model on a single **NVIDIA GeForce RTX 3090**, having $24$ Gbs of memory. Training time obviously depends on the choice of the GPU, but also where the data is stored during training. In our case, we could not store all the data on RAM, hence we loaded every batch during training from a local SSD disk' In the **util** folder there is also a dataloader **EncodedProteinDataset_old** which directly stores the data on RAM, which should boost performance. This should be done only if one has roughly $80Gbs$ of RAM available. As a default we set the hyperparameters of the model to those reported in the manuscript, and for $94.0$ epochs. The training of the standard potts model took roughly $10$ hours, while the training of the autoregressive potts model took about $24$ hours.

## Test

In the folder tests one can find the codes to replicate some of the tests available in the manuscript. To run these tests one will need to have downloaded

1. The [bmDCA](https://github.com/ranganathanlab/bmdca) library to generate efficiently MCMC samples from the Potts model.
2. The [deepStabP](https://github.com/CSBiology/deepStabP) repository for the melting temperature prediction experiment.
3. The [pyhmmer](https://pypi.org/project/pyhmmer/) library to re-align samples(for esm1f).
4. The [esm](https://github.com/facebookresearch/esm) repository.

**NB: The repository have to be placed into paths that match those of the testing files, or those paths into the testing files have to be changed to match the location of the repository on your machine.**

In the folder one can find
 - The file **get_correlations.py**: generates samples from the three different models(arDCA, Potts and esm) for a user specified test dataset and the computes the correlation of the covariance matrix of the generated samples with the true one coming from the MSA. For this experiment, we filter the structures to those of lenght smaller thant $512$ and that have at least $2000$ samples in the MSA trough the Dataloader **EncodedProteinDataset_aux**, so that the "true" covariance coming from the MSA suffieciently reliable. This code takes a long time, especially the Potts generation part which can take up to $3-4$ days on a multi-core machine. One can reduce the size of the testing dataset if needed. We suggest setting some of the [bmDCA](https://github.com/ranganathanlab/bmdca) sampling parameters inside the *bmdca.config* configuration file to low values, especially _resampling_max_; otherwise we could get stuck for a very long time trying to sample from a very badly conditioned Potts model. Find more details in the manuscript.
 - The notebook **generating_many_sequences.ipynb**: allows one to generate the sequences at different hamming distances from the truth for the different models. For this, for every selected test dataset, we select one candidate structure from every avaialable superfamily in the dataset. 
 - The notebook **testing_deepstab.ipynb**: predicts the melting temperature for a set of sequences alreadcy generated by the user, for example trough the previous code. If a GPU is available, this code should run in roughly $30$ mins. 
 - The notebook **plotting_correlations.ipynb**: allows one to replicate the plots of the manuscript, once the correlations have been generated with the code **get_correlations.py**.

 ## Util folder

 As the name suggests, in this folder we have defined all those function which allow for a modular code in all the other folders above reported. The functions inside this folder should be well explained/commented. 
