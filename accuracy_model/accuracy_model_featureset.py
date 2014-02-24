#!/usr/bin/env python

import json
import math
import numpy as np
import random
import re
import sys

from accuracy_model import model

from train_util import model_training_util as acc_util
from accuracy_model import random_features

NUM_RANDOM_FEATURES = 100

error_invalid_history = 0

idx = acc_util.FieldIndexer(acc_util.FieldIndexer.plog_fields)

rand_features = random_features.RandomFeatures(NUM_RANDOM_FEATURES)


def get_baseline_features(ex_state):
    """Return a list of feature values from the baseline AccuracyModel."""
    if ex_state.total_done:
        log_num_done = math.log(ex_state.total_done)
        pct_correct = float(ex_state.total_correct()) / ex_state.total_done
    else:
        log_num_done = 0.0  # avoid log(0.)
        pct_correct = model.PROBABILITY_FIRST_PROBLEM_CORRECT

    return [ex_state.exp_moving_avg(0.333),
            ex_state.exp_moving_avg(0.1),
            ex_state.streak(),
            log_num_done,
            math.log(ex_state.total_done - ex_state.total_correct() + 1),
            pct_correct]


def generate_sample(attempt, attempt_number, ex_states, output_file):
    """Emit a single sample vector based on state prior to this attempt."""
    ex = attempt[idx.exercise]
    outlist = []
    outlist += [attempt[idx.exercise]]
    outlist += ["%d" % attempt[idx.correct]]
    outlist += ["%.4f" % ex_states[ex].predict()]
    outlist += ["%d" % len(ex_states)]
    outlist += ["%d" % attempt[idx.problem_number]]
    outlist += ["%d" % attempt_number]

    # print all the feature values for the existing accuracy model
    for feature in get_baseline_features(ex_states[ex]):
        outlist += ["%.6f" % feature]

    # print random features
    outlist += ["%.6f" % f for f in rand_features.get_features()]
    output_file.write("\t".join(outlist) + "\n")
    return outlist


def generate_samples(attempts, options):
    """TODO(jace)"""

    # make absolutely sure that the attempts are ordered by time_done
    attempts.sort(key=lambda x: x[idx.time_done])

    # If we know we don't have full history for this user, skip her.
    # TODO(jace): restore this check?
    #if acc_util.incomplete_history(attempts, idx):
        #return

    if not acc_util.sequential_problem_numbers(attempts, idx):
        global error_invalid_history
        error_invalid_history += len(attempts)
        return

    # We've passed data validation. Go ahead and process this user.

    ex_states = {}
    rand_features.reset_features()
    with open(options.output_file, 'w') as output_file:
        # Loop through each attempt, already in proper time order.
        for i, attempt in enumerate(attempts):

            ex = attempt[idx.exercise]
            ex_state = ex_states.setdefault(ex, model.AccuracyModel())

            problem_number = int(attempt[idx.problem_number])
            # *Before* we update state, see if we want to sample
            if options.sampling_mode == 'nth':
                freq = int(options.sampling_param)
                if random.randint(1, freq) == freq:
                    samples = generate_sample(
                        attempt, i, ex_states, output_file)
            elif options.sampling_mode == 'prob_num':
                if problem_number >= options.sampling_param[0] and (
                        problem_number < options.sampling_param[1]):
                    samples = generate_sample(
                        attempt, i, ex_states, output_file)
            elif options.sampling_mode == 'randomized':
                purpose = attempt[idx.scheduler_info].get('purpose', None)
                if purpose == 'randomized':
                    samples = generate_sample(
                        attempt, i, ex_states, output_file)

            # Now that we've written out the sample, update features.
            # First, the baseline features.
            ex_state.update(attempt[idx.correct])

            # Next, the random features.  IMPORTANT: For right now, we only
            # update the random features with the *first* attempt on each
            # exercise. This was done in the hopes that the feature
            # distributions would remain as stable as possible in the context
            # of rolling out the "early proficiency" experiment that was
            # expected to modify the typical number of problems done on
            # exercises.

            if problem_number == 1:
                component = (
                    ex, attempt[idx.problem_type], attempt[idx.correct])
                rand_features.increment_component(component)
            return samples


def main(options):
    linesplit = re.compile('[\t,\x01]')
    # Seed the random number generator so experiments are repeatable
    random.seed(909090)
    np.random.seed(909090)

    prev_user = None
    attempts = []
    samples = []
    with open(options.input_file, 'r') as training_set:
        for line in training_set:
            row = linesplit.split(line.strip())

            user = row[idx.user]
            if user != prev_user:
                # We're getting a new user, so perform the reduce operation
                # on all the attempts from the previous user
                samples.append(generate_samples(attempts, options))
                attempts = []

            row[idx.correct] = row[idx.correct] == 'true'
            row[idx.problem_number] = int(row[idx.problem_number])
            row[idx.time_done] = float(row[idx.time_done])
            if options.sampling_mode == 'random':
                row[idx.scheduler_info] = json.loads(
                        row[idx.scheduler_info])

            attempts.append(row)

            prev_user = user
        samples.append(generate_samples(attempts, options))

    sorted_samples = sorted(samples)

    if options.rand_comp_output_file:
        rand_features.write_components(options.rand_comp_output_file)

    print >>sys.stderr, "%d history errors." % error_invalid_history

    return sorted_samples

if __name__ == '__main__':
    main()
