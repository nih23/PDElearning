#!/bin/bash

mpirun -np 4 -bind-to none -map-by slot -mca pml ob1 -mca btl ^openib python3 src/Schrodinger/Schrodinger2D_DeepHPM.py
