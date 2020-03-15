SCRIPT=../../andrey/advanced-ml/project/src/preprocess_sing_300_30_2.py
python $SCRIPT sing_300x30x2/sing_300x30x2 03_us_a_vctk --country=US --gender=A --followers_min=10 --segment_length=5 --out_format=vctk
