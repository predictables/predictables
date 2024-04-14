# synthetic data specs

## overview
your task is to use test driven development to create a synthetic data set to validate a hierarchical naics code embedding model produces embedding vectors that accurately represent the generating distribution

## data set
- the synthetic data set contains the following columns:
    1. naics_2_cd
    2. naics_3_cd
    3. naics_4_cd
    4. naics_5_cd
    5. naics_6_cd
    6. target
- each row in the data represents an observation and should be considered roughly independent

## hierarchical structure of naics codes
the naics code columns contain synthetic naics codes according to the following rules:
1. there are 10 unique naics 2 codes:
    - 10
    - 11
    - 12
    - etc, through 19
2. there are 5 naics 3 codes under each naics 2 code
    - for example, under the naics 2 code 10, there are the naics 3 codes
        1. 101
        2. 102
        3. 103
        4. 104
        5. 105
    - note the hierarchy: a naics 3 code is its parent naics 2 code with an additional digit on the end
    - the same pattern is followed for all remaining naics 2 codes
3. following the same general hierarchy:
    - there are 4 naics 4 codes under each naics 3 code
    - there are 3 naics 5 codes under each naics 4 code
    - there are also 3 naics 6 codes under each naics 5 code

## target variable
- the target variable is a binary 0/1 coded variable 
- the probability of a 1 depends on the naics 2 code:
    1. naics 2 code 10 has a 05% chance of a 1
    2. naics 2 code 11 has an average 15% chance of a 1
    3. naics 2 code 12 has an average 25% chance of a 1
    ...
    10. naics 2 code 19 has a 95% chance of a 1
- the finer grain naics codes (3-6) maintain the average probability of a 1 from their parent code, but apply an additive noise term to that parents mean

## random noise term 
- mean = mean of parents probability
    - for stability, converted to logits before deviating
- standard deviation is the square root of the absolute value of the mean logit

## test suite 
- this data set will be generated with test driven development
    - define parametrized pytest tests to check each of the above requirements are met
        - allow numeric requirements to be within 1% of their expected value
        - test the hierarchy in the naics codes as described above specifically for every row in the dataset

## desired output
ultimately two scripts should be created:
1. test suite (first)
2. dataset generation script (second, following TDD)
