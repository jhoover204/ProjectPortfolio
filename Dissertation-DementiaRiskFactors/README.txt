# Dissertation_DementiaRisk
Dissertation 1: Identifying Risk Factors for Dementia (Cognitive Impairment) using a Bayesian Network Approach

Author: Jonathan Hoover

This folder contains three folders/files with:

A) The code used for 1) exploratory data analysis and data generation, and 2) a Bayesian Network Analysis.
B) Data and images used to generate my report
C) The pdf report of my dissertation

For A (Code_and_Data Folder):

My code is broken into two files. The first is an EDA and data generation file called Hoover_EDA_and_Data_Generation.RMD 
and the second is a Bayesian network analysis file called Hoover_BayesianRiskFactors_BNGraphResults.RMD

The EDA file reads in the easySHARE data from an easySHAREData folder included in the directory and needs nothing else to run. 
It will generate two images used in the report and a cross_sec1_data.csv of the data used for the Bayesian network analysis. 

The Bayesian Network analysis file can run using only the cross_sec1_data.csv file generated from the Data Generation file; 
however, it is very computationally intensive. To make life easier, I have saved the computationally intensive data
in the same folder as the code and commented out the code used to generate it. 
As long as no files are removed from the folder, it should be able to run as is. Without commenting these sections out, 
it may take 20 minutes or longer depending on how many cores you have (I use parallel computing). 
With those lines commented out, it should take a few minutes.

Note that the analysis is subject to minor stochastic variation since some of the functions do not allow a seed 
to be set (we set a seed for other functions). As such, if you run the analysis and compare to the final dissertation, 
some numerical values for validationresu lts will vary slightly. We checked the overall model selection, 
and it was consistent across multiple runs.

For B):

I've included a folder with the markdown file used to generate the report along with all images used for figures in the data.

For C):

This is just the final PDF version of my dissertation called Hoover_BayesianRiskFactors_Dissertation