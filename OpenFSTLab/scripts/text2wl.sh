#!/bin/sh

if [[ ! $1 ]]
then
    echo "usage: $0 text_file"
    exit
fi

if [ -e 'special.txt' ]
then
    add=$(wc -l special.txt)
    cat special.txt
else
    add=0
fi

export add

cat $1 \
    | perl -pe 'tr/a-z/A-Z/; s/ /\n/g' \
    | sort \
    | uniq \
    | egrep -v '[0-9]' \
    | perl -ne 'chomp; printf("$_\t%d\n", $.+$ENV{add})'
