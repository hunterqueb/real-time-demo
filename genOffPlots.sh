# !/bin/bash

# generates plots and saves them in directory for offline viewing

# ./visualize.sh lorenz save
./visualize.sh 2bp save
./visualize.sh 3bp save

./visualize_low.sh lorenz save
./visualize_low.sh 2bp save
./visualize_low.sh 3bp save