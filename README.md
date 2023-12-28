KNN GPU Algorithms

This repository contains 4 KNN GPU Algorithms. We also provide some datafiles, which were created using the Spider Spatial Generator (https://spider.cs.ucr.edu/).

Algorithms with pinned memory, are using the concurrent kernel execution technique.

After compiling each kernel.cu file, you need to provide the following parameters:

1) training dataset
2) query dataset
3) KNN (number of K > 0)
4) In memory process records
5) Number of experiment iterations (1=only one run)
6) o=brief output, p=detailed output
7) Number of streams

eg: kernel.out 1K-uniform.dat 10K-uniform.dat 10 1000 1 o 10


Polychronis Velentzas, University of Thessaly
