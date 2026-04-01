#!/bin/bash

# version that generates low training data results -- based on our convo tuesday.
# call with dynamical system to train and generate plot that demonstrates predicted motion of object from network outputs

if [ "$1" = "lorenz" ]; then
    python scripts/generation/mambaLorenzAttractor.py
    if [ "$2" = "save" ]; then
        python scripts/visualizer_off.py --system lorenz --save
    else
        python scripts/visualizer_off.py --system lorenz
    fi
fi

if [ "$1" = "2bp" ]; then
    python scripts/generation/mamba2BP_low.py

    if [ "$2" = "save" ]; then
        python scripts/visualizer_off.py --system 2bp_low --save
    else
        python scripts/visualizer_off.py --system 2bp_low
    fi
fi

if [ "$1" = "3bp" ]; then
    python scripts/generation/mambaCR3BP6d_low.py

    if [ "$2" = "save" ]; then
        python scripts/visualizer_off.py --system CR3BP_Halo --save
    else
        python scripts/visualizer_off.py --system CR3BP_Halo
    fi
fi