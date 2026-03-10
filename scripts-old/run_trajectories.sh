#!/bin/bash -l

sample_ids=(
415
719
1666
6982
7110
7329
7396
8607
8742
9900
10316
10547
10737
11404
13159
13339
13441
14049
14176
14540
15447
19606
20052
22184
22610
24071
29895
31653
32734
37374
)

for sample_idx in "${sample_ids[@]}"; do
  echo "Running inference for sample_idx=${sample_idx}"
  bash scripts/run_inference.sh mimic_old_ed hospital_mortality_single rep_size=100 res_name_prefix="${sample_idx}" +dataset_kwargs.sample_idx="${sample_idx}"
  bash scripts/run_inference.sh mimic_old_ed icu_admission_single rep_size=100 res_name_prefix="${sample_idx}" +dataset_kwargs.sample_idx="${sample_idx}"
done
