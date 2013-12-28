#!/usr/bin/env python
"""
This script generates a data set of features which can be passed to a
classifier training program to build an accuracy model. The following steps
document example usage of this script in conjunction with
accuracy_model_train.py.

TODO(jace): Make the following into a script of it's own.
---

Step 1)  Execute the following query on Hive

ADD FILE s3://ka-mapreduce/code/py/stacklog_cards_mapper.py;

set hivevar:start_dt=2013-01-01;
set hivevar:end_dt=2013-05-08;

INSERT OVERWRITE DIRECTORY 's3://ka-mapreduce/temp/jace/accmodel'
SELECT
  problemtable.*,
  stacktable.topic,
  get_json_object(stacktable.scheduler_info, '$.purpose')='randomized'
        AS is_random_card
FROM (
  SELECT
    get_json_object(problemlog.json, '$.user') AS user,
      cast(get_json_object(problemlog.json, '$.time_done') as double)
        AS time_done,
    'problemlog',
      get_json_object(problemlog.json, '$.exercise') AS exercise,
      get_json_object(problemlog.json, '$.problem_type') AS problem_type,
      get_json_object(problemlog.json, '$.seed') AS seed,
      cast(get_json_object(problemlog.json, '$.time_taken') as int)
        AS time_taken,
      cast(get_json_object(problemlog.json, '$.problem_number') as int)
        AS problem_number,
      get_json_object(problemlog.json, '$.correct') = "true"
        AS correct,
      get_json_object(problemlog.json, '$.count_attempts')
        AS number_attempts,
      get_json_object(problemlog.json, '$.count_hints')
        AS number_hints,
      (get_json_object(problemlog.json, '$.count_hints') = 0 AND
         (   get_json_object(problemlog.json, '$.count_attempts') > 1
          OR get_json_object(problemlog.json, '$.correct') = "true" ))
        AS eventually_correct,
      get_json_object(problemlog.json, '$.topic_mode') AS topic_mode,
      get_json_object(problemlog.json, '$.key') AS key,
      dt AS dt
  FROM problemlog
  WHERE
    dt >= '${start_dt}' AND dt < '${end_dt}'
  ) problemtable
LEFT OUTER JOIN (
  FROM stacklog
    SELECT TRANSFORM(user, json, dt)
    USING 'stacklog_cards_mapper.py'
    AS key, user, topic, scheduler_info, user_segment
    WHERE stacklog.dt >= '${start_dt}' AND stacklog.dt < '${end_dt}'
  ) stacktable
ON (problemtable.key = stacktable.key);


Step 2) Prepare the data to be piped to this script
cd /ebs/kadata/accmodel/plog
s3cmd get --recursive s3://ka-mapreduce/temp/jace/accmodel
time cat -v accmodel/000* | sed "s/\^A/,/g" | sed "s/\\\N/NULL/g" \
  | sort -s -t, -k1,1 --temporary-directory /ebs/kadata/accmodel/plog/tmp/ \
    --output=accmodel.sorted


Step 3) Pipe the data to this script. (possibly once for each filtering mode)
cd /ebs/kadata/accmodel/code
time cat ../plog/accmodel.sorted \
    | python accuracy_model_featureset.py -s prob_num -p 1,6 -r comps.pickle \
        2>../plog/feat.err \
    | perl -ne 's/\t/,/g; print $_;' \
    > ../plog/feat100.csv
# and sort it
sort -s -t, -k1,1 --temporary-directory /ebs/kadata/accmodel/plog/tmp/ \
    --output=../plog/feat100.sorted.csv ../plog/feat100.csv


Step 4) Train a few models using accuracy_model_train.py.
DATAFILE=/ebs/kadata/accmodel/plog/feat100.1-2.sorted.csv
CODEDIR=/ebs/kadata/accmodel/code
OUTDIR=/home/analytics/tmp/jace/roc
cd $CODEDIR
time cat $DATAFILE | python accuracy_model_train.py \
    --feature_list=baseline --no_bias \
    | grep "rocline" > $OUTDIR/baseline.csv
time cat $DATAFILE | python accuracy_model_train.py \
    --feature_list=none | grep "rocline" > $OUTDIR/bias.csv
time cat $DATAFILE | python accuracy_model_train.py \
    --feature_list=custom -r comps.pickle -o models_custom_only.pickle \
    | grep "rocline" > $OUTDIR/bias+custom.csv
time cat $DATAFILE | python accuracy_model_train.py \
    --feature_list=random -r comps.pickle -o models_random_only.pickle \
    | grep "rocline" > $OUTDIR/bias+random.csv
time cat $DATAFILE | python accuracy_model_train.py \
    --feature_list=custom,random -r comps.pickle -o models.pickle \
    | grep "rocline" > $OUTDIR/bias+custom+random.csv

"""

import ast
import json
import math
import numpy as np
import optparse
import random
import sys

from accuracy_model import model

import accuracy_model_util as acc_util
import random_features

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


def emit_sample(attempt, attempt_number, ex_states):
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

    sys.stdout.write("\t".join(outlist) + "\n")


def emit_samples(attempts, options):
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

    # Loop through each attempt, already in proper time order.
    for i, attempt in enumerate(attempts):

        ex = attempt[idx.exercise]
        ex_state = ex_states.setdefault(ex, model.AccuracyModel())

        problem_number = int(attempt[idx.problem_number])

        # *Before* we update state, see if we want to sample
        if options.sampling_mode == 'nth':
            freq = int(options.sampling_param)
            if random.randint(1, freq) == freq:
                emit_sample(attempt, i, ex_states)
        elif options.sampling_mode == 'prob_num':
            if problem_number >= options.sampling_param[0] and (
                    problem_number < options.sampling_param[1]):
                emit_sample(attempt, i, ex_states)
        elif options.sampling_mode == 'randomized':
            purpose = attempt[idx.scheduler_info].get('purpose', None)
            if purpose == 'randomized':
                emit_sample(attempt, i, ex_states)

        # Now that we've written out the sample, update features.
        # First, the baseline features.
        ex_state.update(attempt[idx.correct])

        # Next, the random features.  IMPORTANT: For right now, we only update
        # the random features with the *first* attempt on each exercise. This
        # was done in the hopes that the feature distributions would remain
        # as stable as possible in the context of rolling out the
        # "early proficiency" experiment that was expected to modify the
        # typical number of problems done on exercises.
        if problem_number == 1:
            component = (ex, attempt[idx.problem_type], attempt[idx.correct])
            rand_features.increment_component(component)


def get_cmd_line_options():
    parser = optparse.OptionParser()

    # TODO(jace): convert to argparse. Until then, formatting will be screwed.
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

    # Seed the random number generator so experiments are repeatable
    random.seed(909090)
    np.random.seed(909090)

    options = get_cmd_line_options()

    prev_user = None
    attempts = []

    for line in sys.stdin:
        row = acc_util.linesplit.split(line.strip())

        user = row[idx.user]
        if user != prev_user:
            # We're getting a new user, so perform the reduce operation
            # on all the attempts from the previous user
            emit_samples(attempts, options)
            attempts = []

        row[idx.correct] = row[idx.correct] == 'true'
        row[idx.problem_number] = int(row[idx.problem_number])
        row[idx.time_done] = float(row[idx.time_done])
        if options.sampling_mode == 'random':
            row[idx.scheduler_info] = json.loads(
                    row[idx.scheduler_info])

        attempts.append(row)

        prev_user = user

    emit_samples(attempts, options)

    if options.rand_comp_output_file:
        rand_features.write_components(options.rand_comp_output_file)

    print >>sys.stderr, "%d history errors." % error_invalid_history

if __name__ == '__main__':
    main()
