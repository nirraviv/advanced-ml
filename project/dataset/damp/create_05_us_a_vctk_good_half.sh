SCRIPT=../../andrey/advanced-ml/project/src/preprocess_sing_300_30_2.py
python $SCRIPT sing_300x30x2/sing_300x30x2 05_us_a_vctk_good_half --country=US --gender=A --followers_min=256 --segment_length=5 --out_format=vctk
