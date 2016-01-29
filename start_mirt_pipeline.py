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

Authors: Eliana Feasley, Jace Kohlmeier, Matt Faus, Jascha Sohl-Dickstein
(2014)
"""
import argparse
import datetime
import multiprocessing
import os
import shutil
import sys

from mirt import mirt_train_EM, generate_predictions, score
from mirt import visualize, adaptive_pretest, generate_responses
from train_util import model_training_util

# Necessary on some systems to make sure all cores are used. If not all
# cores are being used and you'd like a speedup, pip install affinity
try:
    import affinity
    affinity.set_process_affinity_mask(0, 2 ** multiprocessing.cpu_count() - 1)
except NotImplementedError:
    pass
except ImportError:
    sys.stderr.write('If you find that not all cores are being '
                     'used, try installing affinity.\n')


def get_command_line_arguments(arguments=None):
    """Gets command line arguments passed in when called, or
    can be called from within a program.

    Parses input from the command line into options for running
    the MIRT model. For more fine-grained options, look at
    mirt_train_EM.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help=("Generate fake training data."))
    parser.add_argument("--train", action="store_true",
                        help=("Train a model from training data."))
    parser.add_argument("--visualize", action="store_true",
                        help=("Visualize a trained model."))
    parser.add_argument("--test", action="store_true",
                        help=("Take an adaptive test from a trained model."))
    parser.add_argument("--score", action="store_true",
                        help=("Score the responses of each student."))
    parser.add_argument("--report", action="store_true",
                        help=("Report on the parameters of each exercise."))
    parser.add_argument("--roc_viz", action="store_true",
                        help=("Examine the roc curve for the current model"
                              " on the data in the data file."))
    parser.add_argument("--sigmoid_viz", action="store_true",
                        help=("Examine the sigmoids generated for the model in"
                              " the model file."))
    parser.add_argument(
        "-d", "--data_file",
        default=os.path.dirname(
            os.path.abspath(__file__)) + '/sample_data/all.responses',
        help=("Name of file where data of interest is located."))
    parser.add_argument(
        '-a', '--abilities', default=1, type=int,
        help='The dimensionality/number of abilities.')
    parser.add_argument(
        '-s', '--num_students', default=500, type=int,
        help="Number of students to generate data for. Only meaningful when "
        "generating fake data - otherwise it's read from the data file.")
    parser.add_argument(
        '-p', '--num_problems', default=10, type=int,
        help="Number of problems to generate data for. Only meaningful when "
        "generating fake data - otherwise it's read from the data file.")
    parser.add_argument("-t", "--time", action="store_true",
                        help=("Whether to include time as a parameter."
                              "If you do not select time, the 'time' field"
                              "in your data is ignored."))
    parser.add_argument(
        '-w', '--workers', type=int, default=1,
        help=("The number of processes to use to parallelize mirt training"))
    parser.add_argument(
        "-n", "--num_epochs", type=int, default=20,
        help=("The number of EM iterations to do during learning"))
    parser.add_argument(
        "-o", "--model_directory",
        default=os.path.dirname(
            os.path.abspath(__file__)) + '/sample_data/models/',
        help=("The directory to write models and other output"))
    parser.add_argument(
        "-m", "--model",
        default=os.path.dirname(
            os.path.abspath(__file__)) + '/sample_data/models/model.json',
        help=("The location of the model (to write if training, and to read if"
              " visualizing or testing."))
    parser.add_argument(
        "-q", "--num_replicas", type=int, default=1, help=(
            "The number of copies of the data to train on.  If there is too "
            "little training data, increase this number in order to maintain "
            "multiple samples from the abilities vector for each student.  A "
            "sign that there is too little training data is if the update step"
            " length ||dcouplings|| remains large."))
    parser.add_argument(
        "-i", "--items", type=int, default=5, help=(
            "Number of items to use in adaptive test."))

    if arguments:
        arguments = parser.parse_args(arguments)
    else:
        arguments = parser.parse_args()

    # Support file paths in the form of "~/blah", which python
    # doesn't normally recognise
    if arguments.data_file:
        arguments.data_file = os.path.expanduser(arguments.data_file)
    if arguments.model_directory:
        arguments.model_directory = os.path.expanduser(
                arguments.model_directory)
    if arguments.model:
        arguments.model = os.path.expanduser(arguments.model)

    # When visualize is true, we do all visualizations
    if arguments.visualize:
        arguments.roc_viz = True
        arguments.sigmoid_viz = True
        arguments.report = True
    # if we haven't been instructed to do anything, then show the help text
    if not (arguments.generate or arguments.train
            or arguments.visualize or arguments.test
            or arguments.roc_viz or arguments.sigmoid_viz
            or arguments.report or arguments.score):
        print ("\nMust specify at least one task (--generate, --train,"
               " --visualize, --test, --report, --roc_viz, --sigmoid_viz, "
               "--score).\n")
        parser.print_help()

    # Save the current time for reference when looking at generated models.
    DATE_FORMAT = '%Y-%m-%d-%H-%M-%S'
    arguments.datetime = str(datetime.datetime.now().strftime(DATE_FORMAT))

    return arguments


def save_model(arguments):
    """Look at all generated models, and save the most recent to the correct
    location"""
    latest_model = get_latest_parameter_file_name(arguments)
    print "Saving model to %s" % arguments.model
    shutil.copyfile(latest_model, arguments.model)


def get_latest_parameter_file_name(arguments):
    """Get the most recent of many parameter files in a directory.

    There will be many .npz files written; we take the last one.
    """
    params = gen_param_str(arguments)
    path = arguments.model_directory + params + '/'
    npz_files = os.listdir(path)
    npz_files.sort(key=lambda fname: fname.split('_')[-1])
    return path + npz_files[-1]


def main():
    """Get arguments from the command line and runs with those arguments."""
    arguments = get_command_line_arguments()
    run_with_arguments(arguments)


def make_necessary_directories(arguments):
    """Ensure that output directories for the data we'll be writing exist."""
    roc_dir = arguments.model_directory + 'rocs/'
    model_training_util.mkdir_p([roc_dir])


def gen_param_str(arguments):
    """Transform data about current run into a param string for file names."""
    time_str = 'time' if arguments.time else 'no_time'
    return "%s_%s_%s" % (arguments.abilities, time_str, arguments.datetime)


def generate_model_with_parameters(arguments):
    """Trains a model with the given parameters, saving results."""
    param_str = gen_param_str(arguments)
    out_dir_name = arguments.model_directory + param_str + '/'
    model_training_util.mkdir_p(out_dir_name)
    # to set more fine-grained parameters about MIRT training, look at
    # the arguments at mirt/mirt_train_EM.py
    mirt_train_params = [
        '-a', str(arguments.abilities),
        '-w', str(arguments.workers),
        '-n', str(arguments.num_epochs),
        '-f', arguments.model_directory + 'train.responses',
        '-o', out_dir_name,
        ]
    if arguments.time:
        mirt_train_params.append('--time')

    mirt_train_EM.run_programmatically(mirt_train_params)


def generate_roc_curve_from_model(arguments):
    """Read results from each model trained and generate roc curves."""
    roc_dir = arguments.model_directory + 'rocs/'
    roc_file = roc_dir + arguments.datetime
    test_file = arguments.model_directory + 'test.responses'
    return generate_predictions.load_and_simulate_assessment(
        arguments.model, roc_file, test_file)


def run_with_arguments(arguments):
    """Generate data, train a model, visualize your trained data, and score
    students based on a trained model.
    """
    params = gen_param_str(arguments)
    # Set up directories
    make_necessary_directories(arguments)

    if arguments.generate:
        print 'Generating Responses'
        generate_responses.run(arguments)
        print 'Generated responses for %d students and %d problems' % (
            arguments.num_students, arguments.num_problems)
    if arguments.train:
        # Only re-separate into test and train when resume_from_file
        # is False.
        # Separate provided data file into a train and test set.
        model_training_util.sep_into_train_and_test(arguments)

        print 'Training MIRT models'
        generate_model_with_parameters(arguments)
        save_model(arguments)

    if arguments.roc_viz:
        print 'Generating ROC for %s' % arguments.model
        roc_curve = generate_roc_curve_from_model(arguments)
        print 'Visualizing roc for %s' % arguments.model
        visualize.show_roc({params: [r for r in roc_curve]})

    if arguments.sigmoid_viz:
        print 'Visualizing sigmoids for %s' % arguments.model
        visualize.show_exercises(arguments.model)

    if arguments.test:
        print 'Starting adaptive pretest'
        adaptive_pretest.main(arguments.model, arguments.items)

    if arguments.report:
        print "Generating problems report based on params file."
        visualize.print_report(arguments.model)

    if arguments.score:
        print "Scoring all students based on trained test file"
        score.score_students(arguments.model, arguments.data_file)

if __name__ == '__main__':
    main()
