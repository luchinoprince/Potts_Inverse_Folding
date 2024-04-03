#!/bin/bash
# Looking at http://www.me.inf.kyushu-u.ac.jp/witmse2013/abst_files/watanabe2013.pdf, we can see that
# query full is the database of query sequences, uniref50 is the target database in which searching for homologs
# target full is the result database and tmp is a directory where to save auxiliary files necessary for some of the 
# operations, while the latter part are options for the search algorithm.



# -a: tells the algorithm to save the alignment information
# -s: adjust the sensitytivity of the search, 8 is VERY high
# -num-iterations: also increases sensitivity by going trough the dataset more then once when searching
# -e: is the criteria with with two sequences are "aligned" I guess. In the file it tells you the cutoff
    # "a maximum E-value threshold (option -e [0,\infty[) computed according to the
    # gap-corrected Karlin-Altschul statistics using the ALP library"
    # Looking at https://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/_E_value.html#:~:text=The%20default%20threshold%20for%20the,%3C%2010e%2D100%20Identical%20sequences. we get
    # 10e-50 < E-value < 10e-10 Closely related sequences, could be a domain match or similar.
    # 10e-10 < E-value < 1 Could be a true homologue but it is a gray area.

## E Value: The Expect value (E) is a parameter that describes the number of hits one can "expect" to see by chance when searching a database of a particular size. It decreases exponentially as the Score (S) of the match increases.
    ## Essentially, the E value describes the random background noise. For example, an E value of 1 assigned to a hit can be interpreted as meaning that in a database of the current size one might expect to see 1 match with a similar score simply by chance.

## Link to gap-corrected scores: http://etutorials.org/Misc/blast/Part+II+Theory/Chapter+4.+Sequence+Similarity/4.6+Karlin-Altschul+Statistics/



mmseqs search query_full uniref50 target_full tmp/ --num-iterations 3 -a -s 8 -e 0.001 --max-seqs 10000
mmseqs result2msa query_full uniref50 target_full msa_full --msa-format-mode 6
mmseqs unpackdb msa_full ./msas --unpack-name-mode 0 --unpack-suffix ".a3m"
for file in $(ls *a3m | grep -P '\d+.a3m'); do id=$(head -n 1 $file | awk -F'|' '{print $3}' | tr '/' '_'); of="$id".a3m; mv "$file" "$of"; echo $of; done
