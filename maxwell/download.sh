#! /bin/bash
echo "Downloading data from Maxwell server..."
# user '-r' to download directories recursively
scp -r vozdavoj@max-exfl-display.desy.de:/home/vozdavoj/p2838 ./