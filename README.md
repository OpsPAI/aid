# AID

This repository contains the code for ASE'21 paper *AID: Efficient Prediction of Aggregated Intensity of Dependency in Large-scale Cloud Systems*.

## Data

Please download the data from https://doi.org/10.5281/zenodo.5638238.

The file contains a small portion of the preprocessed traces used in our paper. The traces are sanitized subsets of the first-party microservice invocations in one of Huawei Cloud's geographical regions on April 11, 2021. The service names are desensitized.

## Usage

1. `pip install -r requirements.txt`
2. `python intensity.py`

## Reference

If you use our data or code, please kindly cite our paper.
