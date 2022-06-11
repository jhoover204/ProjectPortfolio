## Practical 5: Bayesian inference on Covid-19 incidence in England
## Author: Jonathan Hoover

## Description:
This project was an individual R coding project in my statistical programming course at the University of Edinburgh. The emphasis was proper coding, so analysis is minimal. The project prompt is given in the document "practical-5_description.pdf", but the goal is also defined below.

Note that the project was originally written as an R source file (just code), but I have also included an Rmarkdown file with a knitted pdf of the graphs/code for easier viewing. 

## Goal
The goal of this project is to model the the distribution of fatal Covid-19 infections in the UK
from March 11, 2020 to May 20, 2020 using daily Covid-19 death records from March 2, 2020
to June 9, 2020 (100 total days). The death counts on each day are referred to as yi where 
Y is a random variable distributed according to a Poisson with a mean mi, described below (Yi ~ Pois(mi))

We will take a Bayesian approach to this analysis, assigning prior distributions to some of the 
variables that influence the distribution of daily fatal covid cases. Using the priors and the likelihood,
we will generate a posterior distribution for infectious cases on a given day 
using Gibbs Sampling implemented via JAGS. The specified model is provided in a file called model.jags, 
but the parameters and priors are specified below. 

The definitions of priors comes from the project description (see "practical-5_description.pdf" for details) and the ISARIC study.

A few notes before specifying the parameters:
1) We are assuming that daily fatal infections can be modelled according to the death data from subsequent days
   because data on fatal disease duration (time from infection to death) has been shown to follow a log normal distribution.
   This is what allows us to predict fatal infections from death data (data is from the ISARIC study of 24000+ hospitalized patients).
2) We assume that differences in daily fatal infections from day to day are fairly smooth, which lets us model
   the daily fatal infection cases as a smooth distribution.

The variables we will use in this model are:
tau: Precision parameter that will partially define the distribution of log daily infections (xi; below)
     Tau is also a random variable, with a prior distribution tau ~ Gamma(4, 0.04). 
     An initial value for tau must be specified. We chose tau0 = 0.01.

xi:  Modelled log daily fatal infections (modelled as log since infection counts cannot be negative).
     Modelled as a second order random walk such that x[i+1] - 2[xi] + x[i-1] ~ N(0, tau), where tau is precision.
     Alternatively, xi can be modelled as xi ~ N(2x[i-1] - x[i-2], tau), which is the formulation we use in the model.
     Initial values for xi must be specifed. We chose x1 ~ N(0, tau0 = 0.01) and x2 ~ N(x1, tau)

ni:  Modelled daily fatal infections. Can be calculated as exp(xi).

B:   Square transition probability matrix defining the probability of dying (from infection) on day i
     given an infection on any previous day (lower probabilities for early infectious days).
     Here, Bij refers to probability of dying on day i given infection on day j.
     This probability, piD(i - j), is defined as a log normal distribution, log D ~ N(3.235, 0.4147),
     when i >= j (infected before or on day of death) and as 0 when i < j (Not possible to be infected after death).
     This is a transition probability matrix since dying is the equivalent of moving from state n to m (see next)

mi:  Modeled daily death counts. Calculated as the ith element of B %*% n, or B[1:i] %*% n[1:i]. 
     It is the sum of Prob(death | infection on day j)*(infections on day j) for all j until i.
     Also known as expected death count on day i

yi:  Observed deaths on day i. Y is a random variable such that Yi ~ Pois(mi). This only makes sense if
     The data starts with a reasonable number of zeros. Given no deaths occurred before March 2, it is fair
     to simply append 20 zeros to the start of y.