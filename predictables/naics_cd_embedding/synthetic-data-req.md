# synthetic data specs

## overview
the synthetic data set contains the following columns:
1. naics_2_cd
2. naics_3_cd
3. naics_4_cd
4. naics_5_cd
5. naics_6_cd
6. target

the naics code columns contain synthetic naics codes according to the following:
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
    - the same pattern is followed for all remaining naics 2 codes
3. following the same general hierarchy:
    - there are 4 naics 4 codes under each naics 3 code
    - there are 3 naics 5 codes under each naics 4 code
    - there are also 3 naics 6 codes under each naics 5 code
4. the target variable is a binary 0/1 coded variable 
