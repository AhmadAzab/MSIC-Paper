#!/bin/bash

sox -t raw -r 44100 -b 16 -e signed-integer -L -c 1  "$1" -n spectrogram -r -l -o "$2".png


