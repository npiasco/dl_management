#!/bin/bash

echo Computing pose error with nn indexing

ext=jpg

cpt=0
for f in apt1-living apt1-kitchen apt2-living apt2-kitchen apt2-bed apt2-luke
       	
do
  cd $f && echo ef | python main.py && cd .. &  
  ((cpt+=1))
  echo $cpt
  if (($cpt == 3))
  then
    wait
    cpt=0
  fi
done
wait # wait for parallel process to finish

