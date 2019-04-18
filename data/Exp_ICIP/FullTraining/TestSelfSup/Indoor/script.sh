#!/bin/bash

echo Computing pose error with nn indexing

ext=jpg

cpt=0
for f in chess fire heads office pumpkin redkitchen stairs
       	
do
  cd $f && echo ef | python main.py && cd .. &  
  ((cpt+=1))
  echo $cpt
  if (($cpt == 4))
  then
    wait
    cpt=0
  fi
done
wait # wait for parallel process to finish

