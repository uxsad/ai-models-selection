#!/bin/sh

INFILE=${1:-/dev/stdin}

cut -d',' -f 29-38 $INFILE
