#!/usr/bin/env bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
7z x HIGGS.csv.gz

wget azuremlsampleexperiments.blob.core.windows.net/criteo/day_0.gz
wget azuremlsampleexperiments.blob.core.windows.net/criteo/day_1.gz
