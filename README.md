# Metagenomic Classification

It is a research internship project of Bayes Labs based on deep learning for metagenomic classification.
It's link of web app- https://metagenomics-classifier.herokuapp.com/


This project aims to build a inference engine which is a GeNet deep representation for Metagenomics Classification based on this research paper(https://arxiv.org/pdf/1901.11015.pdf) to replace the Kraken and Centrifuge DNA classification methods which
requires large database that makes them unaffordable untransferable and become challenging when the amount of noise in data increases. To counter these problems Deep learning systems is required that can learn from the noise distribution of the input reads.
Moreover, a classiﬁcation model learns a mapping from input read to class probabilities, and thus does not require a database at run-time. Deep learning systems provide representations of DNA sequences which can be leveraged for downstream tasks.
DNA sequence are also called read which is represented by (G,T,A,C) characters and varies form organism to organism.
Taxonomy is the classification of any organism by this order (Kingdom, Phylum, Class, Order, Family, Genus, Species).
We have to predict the taxonomy by passing read to seven models simultaneously and each models classifies a particular part of above taxa. Combined results of these models helps to classify the read.
There are six kingdoms in biological system -(Plants, Animals, Protists, Fungi, Archaebacteria, Eubacteria)
This project is vast and divided according to the kingdoms as a sub project and each sub project needs eight models taxa+organism name. This report is only on Eubacteria.

## Data Collection

This is the first steps of moving toward project. This project needs a data which has reads and it’s belongs taxonomy.

Data is collected from these resources for each kingdom-
1.NCBI (https://www.ncbi.nlm.nih.gov/) for all kingom.
2.DairyDB(https://github.com/marcomeola/DAIRYdb) for bacteria
3.PlantGDB(http://www.plantgdb.org/) for plants.
4.RVDB(https://rvdb.dbi.udel.edu/) for virus.
5.PlutoF(https://www.gbif.org/dataset/search) for fungi.
6.GreenGene(https://greengenes.secondgenome.com/) for archaea.

All these data are available in FASTA file format which need preprocessing to filter out the required data and stored in a csv file format. 

## Data Preprocessing

After filtering out the required data we have to prepare the balanced train, valid and test CSVs.

Each column of main csv file is considered as the particular model target labels and there are n labels in each columns .

I taked 35 rows of each labels and removed label rows which are less than 35 and truncate the rows of labels which are above 35. I created csv for each labels as row of 35 and put under the particular column folder.

Now I have to prepare train,valid and test csv files. From each label csv I took 20 rows column as train data, 10 rows as a valid data and 5 rows as test data of a particular column.

Above processing will create balanced dataset which helps the model to learn equally for each label. Imbalanced dataset decrease the accuracy.

Data files are in Data folder.

## GeNet: Deep Representations for Metagenomics

Pipelining process of the deep representation for metagenomic 
classification is divided into four parts. For better understanding here is the image below-

![Screenshot](screenshot.png)
