# RAP defense
python3 rap_defense.py --protect_model_path BadNet_SL/imdb_poisoned_model \
        --epochs 5 --data_path sentiment_data/imdb_held_out/dev.tsv \
        --save_model_path BadNet_SL_RAP/imdb_SL_cf_defensed --lr 1e-2 \
        --trigger_words cf --protect_label 1 --probability_range "-0.1 -0.3" \
        --scale_factor 1 --batch_size 32


# test defending performance (FRRs and FARs)
python3 evaluate_rap_performance.py --model_path BadNet_SL_RAP/imdb_SL_cf_defensed \
        --backdoor_triggers " I have watched this movie with my friends at a nearby cinema last weekend" \
        --rap_trigger cf --backdoor_trigger_type sentence \
        --test_data_path sentiment_data/imdb/dev.tsv --constructing_data_path sentiment_data/imdb_held_out/dev.tsv \
        --batch_size 1000 --protect_label 1