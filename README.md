# CUDAatScaleForTheEnterpriseCourseProjectTemplate
This the final course project for the "CUDA at Scale for the Enterprise"
coursera course by Guy Baruch.

## Project Description

The purpose of this application is to present an input image at consecutive subresolutions on a single image.
It inputs a number k=1,..,7 and a greyscale N by M image and outputs an (3/2)N by M image (in PNG format), with copies of the original image downscaled by factors of X2, X4, X8 .. X2^k within the same image.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```lib/```
Any libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.
