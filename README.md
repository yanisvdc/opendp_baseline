# Baseline solution for the privacy part of the PETs for Public Health Challenge 2024

## Challenge Description:

data.org, in partnership with a global financial services institution, Harvard OpenDP, and University of Javariana, has launched a Privacy Enhancing Technologies (PETs) for Public Health Challenge. Up to five winners will be awarded $50,000 each.

This pioneering competition invites academic innovators (Masters, PhDs, Postdocs, faculty, etc.) in differential privacy,  epidemiology, data science, and machine learning, etc. to create privacy solutions that will help unlock sensitive data for public health advancements and drive social impact.  

You can also find more information about the challenge, timing, and funding awards, etc. by visiting Privacy-Enhancing Technologies (PETs) for Public Health Challenge - https://data.org/initiatives/pets-challenge/


## Baseline solution description:

Here is a non-exclusive list of the considerations of the baseline solution provided:

* Transform the dataset: add features corresponding to different location levels of granularity, truncate the dataset to select the desired time frame, preprocess the time column to match the desired time granularity
* Compute Tukey Fences's upper bounds based on DP-quantiles overall q0.25 and q0.75 to reduce the group sum query sensitivity
* Output the list of relevant grouping columns using the exponential mechanism with utility based on the proportion of bins with enough counts above a threshold related to the relative error tolerated
* Output the multi-index and the DP-counts of the stability based histogram (for either the Laplace or Gaussian mechanisms)
* Output the join between the group DP-sum and the stability based DP-counts histogram


## Thanks to all the contributors:

* From OpenDP: Yanis Vandecasteele, Michael Shoemate, Wanrong Zhang, Sharon Ayalde, Salil Vadhan
* From Oblivious: Jack Fitzsimons 
* From NIST: Gary Howarth
* From Knexus Research: Christine Task
