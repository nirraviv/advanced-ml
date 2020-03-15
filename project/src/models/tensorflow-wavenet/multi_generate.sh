_term() { 
  echo "Caught SIGTERM signal!" 
	kill -TERM "$child" 2>/dev/null
}

trap _term SIGTERM

SAMPLES=160000
MODEL=logdir/train/2018-08-19T22-59-03_01_us_f_silence_02/model.ckpt-99999
OUTPUT=results_multi/result_damp_05_g

source env_wn/bin/activate

(for gpu in $(seq 1 7); do
	export CUDA_VISIBLE_DEVICES=$gpu;
	(for i in $(seq 1 40); do
		H=$(date "+%k");
		if (( 9 <= H && H <= 12 )); then
			break;
		fi
		let person="$gpu * 40 - 40 + $i";
		echo python generate.py --samples $SAMPLES $MODEL --wav_out_path ${OUTPUT}${person}'.wav' --gc_cardinality=275 --gc_channels=32 --gc_id=${person}
		python generate.py --samples $SAMPLES $MODEL --wav_out_path ${OUTPUT}${person}'.wav' --gc_cardinality=275 --gc_channels=32 --gc_id=${person}
	done) &
done)

wait

