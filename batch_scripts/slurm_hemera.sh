#!/bin/bash

echo "Running exp" $1

mpirun -np 4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3.6 Schrodinger2D_nh_hvd.py --identifier $1 \
                                --batchsize 25000 \
                                --numbatches 240 \
                                --initsize 15000 \
                                --epochssolution 1000 \
                                --epochsPDE 7000 \
                                --energyloss 1 \
                                --pretraining 1 \
                                --noFeatures 700 \
                                --noLayers 8
