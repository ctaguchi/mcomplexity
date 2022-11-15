# Measuring morphological complexity (reproduced)

This is a forked repository for reproducing the results of the original codes.
This repository also includes the results for some additional treebanks.
`correlation-signif.py`, `wals-values.csv`, `correlation.py`, `dimreduction.py`, `wals-regression.py`, `plot-measures.py` are
files from the original repository and they are not modified (and not necessary to reproduce the results themselves, either).
Files that have been added in this forked repository are:
- `mlc-morph_repr.py` for the main Python script to reproduce the morphological complexity;
- `measures_gsd.txt` for the raw results of measuring complexities;
- `calc_pearson.py` for calculating a Pearson correlation matrix among treebanks' complexities;
- `measures_gsd_pearson.txt` for the Pearson correlation matrix among treebanks' complexities;
- `conllu.py` for dealing with .conllu (UD) files.

Note that you will also need UDtrack dataset to run the experiments.

Original README.md by the original authors follows below.

----
# Measuring morphological complexity

The scripts in this repository is mainly provided for aiding
reproducibility of the following paper:

* Çağrı Çöltekin and Taraka Rama (accepted)
  "What do complexity measures measure?
  Correlating and validating corpus-based measures
  of morphological complexity", Linguistics Vanguard
  (special issue on linguistic complexity) [to be updated with full
  reference]

The scripts also may be useful for other researchers interested in
calculating the measures reported in this paper.

To run these scripts, you will need the
[data](http://www.christianbentz.de/MLC2019_data.html)
from the [MLC'2019 shared task on measuring linguistic complexity](http://www.christianbentz.de/MLC2019_index.html).
We use only the annotated [Universal Dependencies](https://universaldependencies.org/) (UD) data.

The scripts include some comments and command-line help, but
the code is provided without much cleaning or doucmentation.
We intend to add further explanations.
If you have difficulties using the code,
or reproducing the results, please contact the authors,
(send us an email,
or just create an [issue](https://github.com/coltekin/mcomplexity/issues)).
