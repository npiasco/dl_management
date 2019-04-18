#!/bin/bash

echo Computing pose error P3P correction

ext=jpg

cpt=0
#for f in GreatCourt KingsCollege OldHospital ShopFacade StMarysChurch Street
for f in OldHospital ShopFacade StMarysChurch Street
       	
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

