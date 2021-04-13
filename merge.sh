#!/bin/sh

[ -z "$1" ] && echo "$0: missing datasets base path
Use '$0 -h' to get more information" >&2 && exit 1

[ ! -d "$1" ] && echo "$0: dataset base path does not exists or not readable
Use '$0 -h' to get more information" >&2 && exit 1

[ "$1" = "-h" ] && echo "Usage: $0 [-h] PATH
Merges all the aggregate datasets in PATH and print the output to stdout." && exit

needs_head=''
for f in "${1%%/}/"*/aggregate.csv; do
    [ -z "$needs_head" ] && head -n1 "$f"
    needs_head='no'
    tail -n+2 "$f" | awk 'BEGIN { FS=OFS="," } { for(i=30; i<=38; ++i) { if($i == "False") { $i = "" } } } 1'
done
