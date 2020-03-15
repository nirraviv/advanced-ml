SCRIPT=../../andrey/advanced-ml/project/src/preprocess_sing_300_30_2.py
python $SCRIPT sing_300x30x2/sing_300x30x2 01_us_f_vctk --country=US --gender=F --followers_min=10 --segment_length=5 --out_format=vctk
