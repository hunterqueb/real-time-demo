#!/bin/bash

# call with dynamical system to train and generate plot that demonstrates predicted motion of object from network outputs

# remove lorenz?
if [ "$1" = "lorenz" ]; then
    # python scripts/generation/mambaLorenzAttractor.py -- change

    if [ "$2" = "save" ]; then
        python scripts/visualizer_off.py --system lorenz --save
    else
        python scripts/visualizer_off.py --system lorenz
    fi
fi

if [ "$1" = "2bp" ]; then
    python scripts/generation/mamba2BP.py
    if [ "$2" = "save" ]; then
        python scripts/visualizer_off.py --system 2bp --save
    else
        python scripts/visualizer_off.py --system 2bp
    fi
fi

if [ "$1" = "3bp" ]; then
    python scripts/generation/mamba2.1retrograde.py
    if [ "$2" = "save" ]; then
        python scripts/visualizer_off.py --system CR3BP_Retrograde --save
    else
        python scripts/visualizer_off.py --system CR3BP_Retrograde
    fi
fi