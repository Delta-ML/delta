#!/bin/bash

# Creates a lexicon in which each word is represented by the sequence of its characters (spelling). 

srcdict=data/local/dict/lexicon.txt
dir=data/local/dict_char
mkdir -p $dir

[ -f path.sh ] && . ./path.sh

cat $srcdict | grep -v "!SIL" | grep -v "\[VOCALIZED-NOISE\]" | grep -v "\[NOISE\]" | \
  grep -v "\[LAUGHTER\]" | grep -v "\<UNK\>" | \
  perl -e 'while(<>){@A = split; if(! $seen{$A[0]}) {$seen{$A[0]} = 1; print $_;}}' \
  > $dir/lexicon1.txt || exit 1;

#cat $phndir/lexicon.txt | grep -v "\[VOCALIZED-NOISE\]" | grep -v "\[NOISE\]" | \
#  grep -v "\[LAUGHTER\]" | grep -v "\<UNK\>" \
#  > $dir/lexicon1.txt

unset LC_ALL
cat $dir/lexicon1.txt | awk '{print $1}' | \
  perl -e 'use encoding utf8; while(<>){ chop; $str="$_"; foreach $p (split("", $_)) {$str="$str $p"}; print "$str\n";}' \
  > $dir/lexicon2.txt

#  Get the set of lexicon units without noises
cut -d' ' -f2- $dir/lexicon2.txt | tr ' ' '\n' | sort -u > $dir/units_nosil.txt

# Add special noises words & characters into the lexicon.
(echo '[VOCALIZED-NOISE] [VOCALIZED-NOISE]'; echo '[NOISE] [NOISE]'; echo '[LAUGHTER] [LAUGHTER]'; echo '<UNK> <UNK>'; echo '<space> <space>';) | \
  cat - $dir/lexicon2.txt | sort | uniq > $dir/lexicon3.txt || exit 1;

cat $dir/lexicon3.txt | sort -u > $dir/lexicon.txt || exit 1;

# The complete set of lexicon units, indexed by numbers starting from 1
(echo '[VOCALIZED-NOISE]'; echo '[NOISE]'; echo '[LAUGHTER]'; echo '<UNK>'; echo '<space>';) | cat - $dir/units_nosil.txt | awk '{print $1 " " NR}' > $dir/units.txt

# Convert phoneme sequences into the corresponding sequences of units indices, encoded by units.txt
utils/sym2int.pl -f 2- $dir/units.txt < $dir/lexicon.txt > $dir/lexicon_numbers.txt

echo "Character-based dictionary preparation succeeded"
