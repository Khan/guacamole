#!/usr/bin/env python
"""This file will take you all the way from a CSV of student performance on
test items to trained parameters describing the difficulties of the assessment
items.
The parameters can be used to identify the different concepts in your
assessment items, and to drive your own adaptive test. The mirt_engine python
file included here can be used to run an adaptive pretest that will provide an
adaptive set of assessment items if you provide information about whether the
questions are being answered correctly or incorrectly.

Example Use:
    with a file called my_data.csv call
    ./start_mirt_pipeline -i path/to/my_data.csv
    let a1_time.json be the name of the output json file
        (Congrats! Examine that for information about item difficulty!)

    To run an adaptive test with your test items:
    ./run_adaptive_test.py -i a1_time.json
    This will open an interactive session where the test will ask you questions
    according to whatever will cause the model to gain the most information to
    predict your abilities.
"""
import argparse
import datetime
import multiprocessing
import os

from mirt import mirt_train_EM, mirt_npz_to_json, generate_predictions
from train_util import model_training_util

# Necessary on some systems to make sure all cores are used. If not all
# cores are being used and you'd like a speedup, pip install affinity
try:
    import affinity
    affinity.set_process_affinity_mask(0, 2 ** multiprocessing.cpu_count() - 1)
except NotImplementedError:
    pass


def get_command_line_arguments(arguments=None):
    """Gets command line arguments passed in when called, or
    can be called from within a program.

    Parses input from the command line into options for running
    the MIRT model. For more fine-grained options, look at
    mirt_train_EM.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        default=os.path.dirname(
            os.path.abspath(__file__)) + '/sample_data/all.responses',
        help=("Name of file where data of interest is located."))
    parser.add_argument(
        '-a', '--abilities', default=[1], type=int,
        nargs='+', help='The dimensionality/number of abilities.'
        'this can be a series of values for multiple models, ie. -a 1 2 3')
    parser.add_argument(
        '-t', '--time', default='with_time',
        help=("Whether to train with time (default=False).\n"
              "Valid inputs:\n"
              "\twith_time : default. trains with time\n"
              "\twithout_time : Trains a mirt model with no time\n"
              "\twith_and_without_time : trains both types of models!\n"
              "\t\tDouble the time, double the fun!"))
    parser.add_argument(
        '-w', '--workers', type=int, default=1,
        help=("The number of processes to use to parallelize mirt training"))
    parser.add_argument(
        "-n", "--num_epochs", type=int, default=100,
        help=("The number of EM iterations to do during learning"))
    parser.add_argument(
        "-o", "--output",
        default=os.path.dirname(
            os.path.abspath(__file__)) + '/sample_data/output/',
        help=("The directory to write output"))

    if arguments:
        arguments = parser.parse_args(arguments)
    else:
        arguments = parser.parse_args()

    return arguments


def get_time_arguments(arguments):
    """We take arguments about training models incorporating
    response time.
    """
    if arguments.time == 'with_and_without_time':
        time_arguments = ['', '-z']
    elif arguments.time == 'without_time':
        time_arguments = ['-z']
    elif arguments.time == 'with_time':
        time_arguments = ['']
    else:
        print 'Invalid argument selected for --time'
        print 'Choosing default behavior - with time'
        time_arguments = ['']
    return time_arguments


def get_latest_npz_file_name(path):
    """Gets the most recent of many parameter files in a directory.

    There will be many .npz files written; we take the last one.
    """
    npz_files = os.listdir(path)
    npz_files.sort(key=lambda fname: fname.split('_')[-1])
    return path + npz_files[-1]


def get_json_path(arguments):
    """Output the location of the place jsons will be stored."""
    mirt_dir = arguments.output + 'mirt/'
    return mirt_dir + 'jsons/'


def main():
    """Gets arguments from the command line and runs with those arguments."""
    arguments = get_command_line_arguments()
    run_with_arguments(arguments)


def make_necessary_directiories(arguments):
    """Ensure that output directories for the data we'll be writing exist.
    """
    json_dir = get_json_path(arguments)
    mirt_dir = arguments.output + 'mirt/'
    roc_dir = mirt_dir + 'rocs/'
    model_training_util.mkdir_p([json_dir, mirt_dir, roc_dir])


def gen_param_str(abilities, datetime_str, time):
    """Transform data about current run into a param string for file names.
    """
    time_str = 'time' if time else 'no_time'
    return "%s_%s_%s" % (abilities, time_str, datetime_str)


def generate_model_with_parameters(
        arguments, abilities, time, datetime_str):
    """Trains a model with the given parameters, saving results
    """
    param_str = gen_param_str(abilities, datetime_str, time)
    out_dir_name = arguments.output + 'mirt/' + param_str + '/'
    model_training_util.mkdir_p(out_dir_name)
    # to set more fine-grained parameters about MIRT training, look at
    # the arguments at mirt/mirt_train_EM.py
    mirt_train_params = [
        '-a', str(abilities),
        '-w', str(arguments.workers),
        '-n', str(arguments.num_epochs),
        '-f', arguments.output + 'train.responses',
        '-o', out_dir_name]
    if time:
        mirt_train_params.append(time)

    mirt_train_EM.run_programmatically(mirt_train_params)


def generate_roc_curve_from_model(
        arguments, abilities, time, datetime_str):
    """Read results from each model trained and generate roc curves.
    """
    # There will be many .npz files written; we take the last one.
    mirt_dir = arguments.output + 'mirt/'
    roc_dir = mirt_dir + 'rocs/'
    test_file = arguments.output + 'test.responses'
    param_str = gen_param_str(abilities, datetime_str, time)
    out_dir_name = arguments.output + 'mirt/' + param_str + '/'
    last_npz = get_latest_npz_file_name(out_dir_name)
    print 'getting last npz'
    print last_npz
    json_outfile = get_json_path(arguments) + param_str + '.json'
    print 'writing json'
    print json_outfile
    mirt_npz_to_json.mirt_npz_to_json(
        last_npz, outfilename=json_outfile, slug=param_str,
        title='math', description='math')
    roc_file = roc_dir + param_str + '.roc'
    print 'writing to roc file'
    print roc_file
    generate_predictions.load_and_simulate_assessment(
        json_outfile, roc_file, test_file, user_index=0,
        exercise_index=2, time_index=3, correct_index=4)


def run_with_arguments(arguments):
    """Takes you through every step from having a model, training it,
    testing it, and potentially uploading it to a testing engine.
    """
    # Set up directories
    make_necessary_directiories(arguments)

    # Generate data, either by downloading from AWS or by providing your own
    # data from some other source
    model_training_util.sep_into_train_and_test(arguments)

    print 'Training MIRT models'
    datetime_str = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    # For each combination of the setting for "abilities" and
    # "response_time_mode", we want to fit a model.
    # Loop through the combinations and fit a model for each.
    for abilities in arguments.abilities:
        for time in get_time_arguments(arguments):
            generate_model_with_parameters(
                arguments, abilities, time, datetime_str)
            generate_roc_curve_from_model(
                arguments, abilities, time, datetime_str)

    print
    print "If you're running this script somewhere you can't see matplotlib, "
    print "copy %s* to somewhere you can see it, then run " % arguments.output
    print "mirt/rocs/train_util/plot_roc_curves.py rocs/* on that machine."
    print
    print "Generated JSON MIRT models are in %s " % get_json_path(arguments)
    # TODO(eliana): Modify to automatically save pdfs of the roc curves.

if __name__ == '__main__':
    main()
