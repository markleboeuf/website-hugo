+++
date = "2017-01-12T19:41:01+05:30"
title = "Two Flavors of Parallel Simulation"
draft = false
image = "img/portfolio/two_flavors_of_parallelism_images/two_flavors.jpg"
showonlyimage = false
weight = 2
tags = ["PARALLEL", "R", "FOREACH", "MULTIDPLYR"]
+++

Tired of waiting around for your simulations to finish? Run them in parallel! This post covers two seperate ways to add parallelism to your R code. 
<!--more-->

<img src="../two_flavors_of_parallelism_images/two_flavors.jpg" class="img-responsive" style="display: block; margin: auto;" />

-   [Overview](#overview)
-   [Parallel Simulations with Foreach](#parallel-simulations-with-foreach)
-   [Parallel Simulations with Multidplyr](#parallel-simulations-with-multidplyr)

Overview
------------

In a prior [post](https://markleboeuf.github.io/portfolio/monte_carlo_mixed_effects) we discussed how to use monte carlo simulation for power analysis. We kept the total number of iterations relatively low to illustrate the process. However, the real value of simulation emerges when you run lots of simulations, because the more iterations you run the better idea you get about the thing you are trying to estimate (see [*Law of Large Numbers*](https://en.wikipedia.org/wiki/Law_of_large_numbers)). In the case of estimating the power of an experiment, the more simulated experiments we run the closer we'll get to the true probability of committing a [*Type II Error*](http://support.minitab.com/en-us/minitab/17/topic-library/basic-statistics-and-graphs/hypothesis-tests/basics/type-i-and-type-ii-error/). Simulating the experimental paradigm sequentially is fine but it takes a long time when you increase the number of simulations to, say, 10K or 100K. Any time you come across a task that involves repeated sampling from a distribution -- **think parallel**. The results of one simulation do not feed into or depend on the results of another. Thus we can run many simulated experiments at the same time. This is a common theme of any task that is parallelizable, which might be one of the most challenging words to say. In this post I'm going to discuss two seperate ways to implement a power analysis simulation in R. And although we'll focus only on paralellism in the context of experimental power, the workflow discussed here can be generalized to almost any task that involves repeated sampling. 

### Parallel Simulations with Foreach

Before starting let me provide a brief summary of the analytical dataset. Researchers conducted a study examining the impact of continued sleep deprivation (defined as receiving only 3 hours of sleep per night) on reaction time. The study was run for 9 days and the researchers found a significant effect for Number of Days. As you can imagine, participants were a lot slower to react on days 8 & 9 relative to days 0 & 1. We want to replicate this effect but don't have the time to wait 9 days for a result. Our question, then, is whether we could still detect an effect of sleep deprivation after only 3 days. The goal is to achieve at least 80% power, which means that if we replicated the experiment 10 times under the exact same conditions, we would find a significant effect (*p* < 0.05) in at least 8 experiments. 

We'll use the findings from the prior study over the first 3 days as our base data set. The process will be modeled with a mixed effects model with a random intercept for each participant. Our fixed effect -- the thing we are interested in -- is days of sleep deprivation. Let's load up our libraries and fit the initial model.

``` r
libs = c('foreach', 'doParallel', 'lme4', 'dplyr', 'broom', 'ggplot2', 'multidplyr', 'knitr')
lapply(libs, require, character.only = TRUE)
sleep_df = lme4::sleepstudy %>% 
           dplyr::filter(Days %in% c(0, 1, 2, 3))

fit = lmer(Reaction ~ Days + (1|Subject), data = sleep_df)
confidence_intervals = confint(fit)
```


```r
print(summary(fit))
print(confidence_intevals)
```

``` r
## Linear mixed model fit by REML ['lmerMod']
## Formula: Reaction ~ Days + (1 | Subject)
##    Data: sleep_df
## 
## REML criterion at convergence: 660.4
## 
## Scaled residuals: 
##      Min       1Q   Median       3Q      Max 
## -3.14771 -0.50969 -0.08642  0.48985  2.05082 
## 
## Random effects:
##  Groups   Name        Variance Std.Dev.
##  Subject  (Intercept) 755.7    27.49   
##  Residual             379.1    19.47   
## Number of obs: 72, groups:  Subject, 18
## 
## Fixed effects:
##             Estimate Std. Error t value
## (Intercept)  255.392      7.532   33.91
## Days           7.989      2.052    3.89
## 
## Correlation of Fixed Effects:
##      (Intr)
## Days -0.409

##                  2.5 %    97.5 %
## .sig01       18.702382  39.73719
## .sigma       16.152095  23.58800
## (Intercept) 240.429427 270.35528
## Days          3.931803  12.04555
```

Our model indicates that after controlling for baseline differences in participant reaction time (i.e., our random intercept), each additional day increases reaction time by about 8 seconds (7.989 to be exact). Our confidence interval for this coefficient indicates a significant effect, as the range does not contain zero. However, the range of our estimate is fairly wide. Let's determine how this uncertainty affects overall experimental power. We'll make predictions on our base dataset with the model defined above, and then add noise (defined by our residuals from our initial model fit) to simulate the sampling process. 

``` r
model_predictions = predict(fit, sleep_df)

standard_deviation = sd(fit@resp$y - fit@resp$mu)

n_simulations = 100
```

`For` loops in R are great for small operations but are the absolute worst for larger operations. Enter `foreach`. The syntax is a little bit different from your typical `for` loop. Let's first see how to implement our power simulation **sequentially** using `foreach`. Note that this approach is identical to using a regular `for` loop. 


``` r
sequential_start_time <- Sys.time()

sequential_results = foreach(
                        i = 1:n_simulations,
                        .combine = "rbind",
                        .packages = c("lme4", "broom", "dplyr")) %do% {
                        # generate residuals
                        temporary_residuals = rnorm(nrow(sleep_df), mean = 0, sd = standard_deviation)

                        #create simulated reaction time
                        sleep_df$Simulated_Reaction <- model_predictions + temporary_residuals

                        #refit our model on the simulated data
                        temp_fit = lmer(Simulated_Reaction ~ Days + (1|Subject), data = sleep_df)

                        # return confidence interval for the Days coefficient
                        tidy(confint(temp_fit)) %>%
                        dplyr::rename(coefficients = .rownames,
                        lower_bound = X2.5..,
                        upper_bound = X97.5..) %>%
                        dplyr::filter(coefficients == 'Days') %>%
                        dplyr::select(lower_bound, upper_bound)
}

sequential_end_time <- Sys.time()
sequential_run_time <- sequential_end_time - sequential_start_time
print(paste0("TOTAL RUN TIME: ", sequential_run_time))
```
```
## [1] "TOTAL RUN TIME: 1.25311950047811"
```

Implementing the power simulation sequentially took 1.25 minutes on my computer (~75 seconds). Let's compare that to a parallel implementation. All we have to do is change the `%do%` to  `%dopar%` to shift the execution from sequential to parallel. But first we'll have to set up a computing cluster.


``` r 
# register our cluster
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)
```
My computing cluster will have 3 cores. I have a total of 4 cores on my machine, but I want to save one for browsing the internet, cat gifs, etc. Now that we've registered our cluster, let's re-run the above code block but replace `%do%` with `%dopar%` and compare the run time. 

``` r 
parallel_start_time <- Sys.time()
parallel_results = foreach(
                           i = 1:n_simulations,
                           .combine = "rbind",
                           .packages = c("lme4", "broom", "dplyr")) %dopar% {
                           # generate residuals
                           temporary_residuals = rnorm(nrow(sleep_df), mean = 0, sd = standard_deviation)

                           #create simulated reaction time
                           sleep_df$Simulated_Reaction = model_predictions + temporary_residuals

                           #refit our model on the simulated data
                           temp_fit = lmer(Simulated_Reaction ~ Days + (1|Subject), data = sleep_df)

                           #return confidence interval for the Days coefficient
                           tidy(confint(temp_fit)) %>%
                           dplyr::rename(coefficients = .rownames,
                           lower_bound = X2.5..,
                           upper_bound = X97.5..) %>%
                           dplyr::filter(coefficients == 'Days') %>%
                           dplyr::select(lower_bound, upper_bound)
}

parallel_end_time <- Sys.time()
parallel_run_time <- parallel_end_time - parallel_start_time

print(paste0("TOTAL RUN TIME: ", parallel_run_time))
```
``` r
## [1] "TOTAL RUN TIME: 34.4293410778046"
```

So that only took 34.4 seconds, which over a 50% reduction in runtime! That means 50% more time to to view cat gifs or do other, productive activities. Let's check and see how our power calculations panned out. Every instance in which we find a zero in our confidence interval for the Days estimate is a type II error.  


``` r
power_results = parallel_results %>% 
                dplyr::mutate(row_index = 1:nrow(parallel_results)) %>% 
                dplyr::group_by(row_index) %>% 
                dplyr::do(result = dplyr::between(0, .$lower_bound, .$upper_bound) %>% 
                dplyr::mutate(result = as.integer(unlist(result))) %>% 
                data.frame()

print(paste0("TOTAL POWER: ", (n_simulations - sum(power_results$result)), "%"))

```
```r
## [1] "TOTAL POWER: 99%"
```

If we ran our experiment under these conditions, we'd detect an effect that we know exists in about 99 of every 100 experiments. So it turns out we can reliably detect an effect with only 3 days instead of running it for all 9, saving us time and money. Let's move on to the second approach to parallelism that keeps all of our operations in the `tidyverse`, which is total R programming zen. 

### Parallel Simulations with multidplyr

If you haven't used `dplyr` before, I would strongly suggest learning it. It is a huge boon for productivity, as you can express all of your data munging operations in clean, easy to read syntax. `Multidplyr` builds on `dplyr` by allowing operations to performed in parallel. It is a natural fit when you have grouped data and want to apply the same function to each group. The groups in our data will be each sampling iteration. I'll go through line by line and explain what's happening. 

Here we are going to make 100 copies of our dataset and bind them together. We'll also generate all the errors for each of the iterations and bind that to our 100 copies of the dataset. 

``` r
sleep_df_copy = data.frame(sapply(sleep_df, rep.int, times = n_simulations))

temporary_residuals = c()
for(i in 1:n_simulations){
    temporary_residuals = c(temporary_residuals, rnorm(nrow(sleep_df), mean = 0, sd = standard_deviation))
}
sleep_df_copy$iteration = rep(1:n_simulations, each = nrow(sleep_df))

sleep_df_copy$Simulated_Reaction = temporary_residuals + rep(model_predictions, n_simulations)
```

At this point each study has 72 observations (18 participants with 4 data points each). We created 100 replications of the study, so our total dataset size is now 7200 rows. Each 72 observation "group" is identified by the `iteration` field. This means that each core should receive approximately 33 iterations with 72 observations per iteration, for a total of around 2400 observations. The data structure that holds the data partitions is called a `party_df` for `partitioned data frame`..or maybe because all of the cores can join the party..I'm not sure. Let's create one and examine the distribution of our observations. 

```{r}
partitioned_experiment <- multidplyr::partition(sleep_df_copy, iteration)

# Source: party_df [7,200 x 5]
# Groups: iteration
# Shards: 3 [2,376--2,448 rows]
# 
# # S3: party_df
#    Reaction  Days Subject Simulated_Reaction iteration
#       <dbl> <dbl>   <dbl>              <dbl>     <int>
# 1  249.5600     0       1           261.4119         2
# 2  258.7047     1       1           269.7817         2
# 3  250.8006     2       1           278.6748         2
# 4  321.4398     3       1           290.9825         2
# 5  222.7339     0       2           192.3452         2
# 6  205.2658     1       2           169.2214         2
# 7  202.9778     2       2           196.4420         2
# 8  204.7070     3       2           228.1363         2
# 9  199.0539     0       3           205.1141         2
# 10 194.3322     1       3           247.2929         2
# ... with 7,190 more rows
```

Sweet! `Multidplyr` has producted a relatively even split of our dataset, which is exactly what we want. Next we'll load up the libraries and pass the function we want to apply to each of our partitions. In this case we've defined it as `calculateCI`. 

```r
# Create function to calculate confidence intervals
calculateCI = function(sleep_data){
    # fit the model
    temp_fit <- lmer(Simulated_Reaction ~ Days + (1|Subject), data = sleep_data)

    # return the coefficients
    tidy(confint(temp_fit)) %>% 
    dplyr::rename(coefficients = .rownames,
    lower_bound = X2.5..,
    upper_bound = X97.5..) %>% 
    dplyr::filter(coefficients == 'Days') %>% 
    dplyr::select(lower_bound, upper_bound)

}

# load our libraries and calculateCI function on each of the cores
partitioned_experiment %>% 
    cluster_library("lme4") %>% 
    cluster_library("dplyr") %>% 
    cluster_library("broom") %>% 
    cluster_assign_value("calculateCI", calculateCI)

```

That's it. We are now ready to run our simulated power experiment. We'll use the `collect` verb to bring the partitioned results back. We'll also do a little list processing, as the results are returned in a list format. 

``` r
multidplyr_start_time <- Sys.time()

multidplyr_results <- partitioned_experiment %>% 
dplyr::do(results = calculateCI(.)) %>% 
          collect() %>% 
          dplyr::mutate(lower_bound = unlist(lapply(results, function(x) x[[1]][1])),
          upper_bound = unlist(lapply(results, function(x) x[[2]][1]))) %>%
          dplyr::select(-results) %>% 
          data.frame()

multidplyr_end_time <- Sys.time()
multidplyr_run_time <- multidplyr_end_time - multidplyr_start_time

print(paste0("TOTAL RUN TIME: ", round(multidplyr_run_time)))
```

``` r
## "TOTAL RUN TIME: 27.8141870498657"
```

Run time is somewhat faster than `foreach` but the two are pretty close. Let's check out the power estimate. 


```r
power_results = multidplyr_results %>% 
                dplyr::mutate(row_index = 1:nrow(parallel_results)) %>% 
                dplyr::group_by(row_index) %>% 
                dplyr::do(result = dplyr::between(0, .$lower_bound, .$upper_bound)) %>% 
                dplyr::mutate(result = as.integer(unlist(result))) %>% 
                data.frame()

print(paste0("TOTAL POWER: ", (n_simulations - sum(power_results$result)), "%"))
```

```
## [1] "TOTAL POWER: 100%"
```

This simulation gave us a nearly identical estimate relative to the `foreach`.Taken together, the results from both of these simulations indicate that we wouldn't need the full 9 days to show an effect. When you are finished, remember to shut your cluster down with the following command. 

``` r
stopCluster(cl)
```

I hope this post provides you with better idea of how easy it is to add some parallelism to your R code. Time always seems to be in short supply when you are developing, and waiting around for an answer is a total momentum killer. Taking a bit more time up front to understand whether you can run your code in parallel -- and avoiding sequential for loops -- will save you a ton of time down the line.
