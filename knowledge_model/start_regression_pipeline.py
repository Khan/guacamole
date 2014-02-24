#!/usr/bin/env python
import ast
import optparse
import sys

from accuracy_model import accuracy_model_featureset
from accuracy_model import accuracy_model_train


def get_cmd_line_options():
    parser = optparse.OptionParser()

    # TODO(jace): convert to argparse. Until then, formatting will be screwed.
    parser.add_option("-f", "--input_file",
        default="sample_data/accuracy_model.responses",
        help="Path to the file with user responses we'll learn from")

    parser.add_option("-d", "--output_file",
        default="sample_data/accuracy_model.datapoints",
        help="Path to the file we'll write datapoints out to.")

    parser.add_option("-s", "--sampling_mode", default="prob_num",
        help="Determines which problem attempts get included in "
             "the data sample.  Three modes are possible:"
             "  randomized - use only the random assessment cards. NOTE: This "
             "      mode is currently broken, since problem log input is "
             "      assumed, and we need topic_mode input to know which cards "
             "      were random. "
             "  nth - use only 1 in every N cards as a sample "
             "  prob_num - output only if problem_number is within the range "
             "      specified by the sampling_param option. sampling_param is "
             "      a string, but should evaluate to a tuple of length 2 "
             "      through ast.literal_eval(). The 2 values are the start "
             "      and end of a range. "
             "      Ex:  '--sampling_mode=prob_num --sampling_param=(1,6)' "
             "      sample problem numbers 1 through 5. ")

    parser.add_option("-p", "--sampling_param", type=str, default=None,
        help="This parameter is used in conjuction with some samlpling "
             "modes. See documentation sampling_mode for each mode.")

    parser.add_option("-r", "--rand_comp_output_file", default=None,
        help="If provided, a file containing the random components will be "
             "output. If a model gets productionized, you need to have this "
             "record of the random component vectors.")

    parser.add_option("-c", "--classifier", default='logistic_log',
            help="Type of classifier.  'logistic_log' or 'random_forest'.")
    parser.add_option("-f", "--feature_list", default='baseline',
            help="Comma seprated list to feature types to use. Choose from "
                 "baseline, custom, random, and mirt. If you want to run "
                 "with nothing but a bias feature, use 'none'.")
    parser.add_option("-n", "--no_bias", dest="no_bias", action="store_true",
            help="Whether to omit using a bias in the feature set.  By "
                 "default a bias unit is included.")
    parser.add_option("-r", "--rand_comp_file",
            help="Name of a file to optionally write the random components "
                 "to.")
    parser.add_option("-o", "--output_model_file",
            help="Name of a file you optionally want to write a pickeled "
                 "file of all the models to.")
    parser.add_option("-t", "--topic",
            help="If given, then models are not fit per exercise; rather "
                 "all exercises are treated as part of one 'topic' and one "
                 "model is fit for all of the data.")

    options, _ = parser.parse_args()

    if options.sampling_mode == "prob_num":
        options.sampling_param = ast.literal_eval(options.sampling_param)
        if not isinstance(options.sampling_param, tuple):
            print >>sys.stderr, (
                    "ERROR: sampling_param should evaluate to a tuple.")
            parser.print_help()
            exit(-1)
    return options


def main():
    options = get_cmd_line_options()

    dataset = accuracy_model_featureset.main(options)

    accuracy_model_train.main(options, dataset)
    # Step 4) Train a few models using accuracy_model_train.py.
    # DATAFILE=/ebs/kadata/accmodel/plog/feat100.1-2.sorted.csv
    # CODEDIR=/ebs/kadata/accmodel/code
    # OUTDIR=/home/analytics/tmp/jace/roc
    # cd $CODEDIR
    # time cat $DATAFILE | python accuracy_model_train.py \
    #     --feature_list=baseline --no_bias \
    #     | grep "rocline" > $OUTDIR/baseline.csv
    # time cat $DATAFILE | python accuracy_model_train.py \
    #     --feature_list=none | grep "rocline" > $OUTDIR/bias.csv
    # time cat $DATAFILE | python accuracy_model_train.py \
    #     --feature_list=custom -r comps.pickle -o models_custom_only.pickle \
    #     | grep "rocline" > $OUTDIR/bias+custom.csv
    # time cat $DATAFILE | python accuracy_model_train.py \
    #     --feature_list=random -r comps.pickle -o models_random_only.pickle \
    #     | grep "rocline" > $OUTDIR/bias+random.csv
    # time cat $DATAFILE | python accuracy_model_train.py \
    #     --feature_list=custom,random -r comps.pickle -o models.pickle \
    #     | grep "rocline" > $OUTDIR/bias+custom+random.csv

if __name__ == '__main__':
    main()
