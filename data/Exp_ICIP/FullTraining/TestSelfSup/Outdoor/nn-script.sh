#!/bin/bash

echo Computing pose error nn indexing

ext=jpg

cpt=0
for f in GreatCourt KingsCollege OldHospital ShopFacade StMarysChurch Street
       	
do
  cd $f && echo ef | python main-nn.py && cd .. &  
  ((cpt+=1))
  echo $cpt
  if (($cpt == 3))
  then
    wait
    cpt=0
  fi
done
wait # wait for parallel process to finish

