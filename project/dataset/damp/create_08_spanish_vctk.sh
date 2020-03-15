SCRIPT=../../andrey/advanced-ml/project/src/preprocess_sing_300_30_2.py
python $SCRIPT sing_300x30x2/sing_300x30x2 08_spanish_vctk --country=AR,CL,ES,MX --gender=A --followers_min=256 --segment_length=10 --out_format=vctk
