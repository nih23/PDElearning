#!/bin/bash

mpirun -np 6 \
       -bind-to none -map-by slot \
          -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python Schrodinger2D_nh_hvd_v2.py --identifier e1_binet_dbg \
                                --batchsize 10000 \
                                --numbatches 240 \
                                --initsize 1000 \
                                --epochssolution 1000 \
                                --epochsPDE 7000 \
                                --energyloss 0 \
                                --pretraining 1 \
                                --noFeatures 400 \
                                --noLayers 8 \
                                --binet
