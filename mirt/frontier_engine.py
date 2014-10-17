"""WIP, Use not recommended:
This engine looks for the fronteir
"""

from collections import defaultdict
import numpy as np
import random

from mirt import engine
from mirt import mirt_engine

import json


class ExerciseGraph(object):

    def __init__(self, topic):
        self.children_lookup = defaultdict(list)
        self.parents_lookup = defaultdict(list)
        self.topic = topic
        # Add topic relationships
        (self.topic_to_exercise_mapping,
            prereq_table,
            postreq_table,
            self.all_exercise_names) = get_cached_mission_data(topic)
        for topic in self.topic_to_exercise_mapping.values():
            for i in range(len(topic)):
                if i > 0:
                    self.children_lookup[topic[i]].append(topic[i - 1])
                if i < len(topic) - 1:
                    self.parents_lookup[topic[i]].append(topic[i + 1])
        for exercise in prereq_table:
            children = self.get_all_relatives(exercise, 'child')
            parents = self.get_all_relatives(exercise, 'parent')

            for prereq in prereq_table[exercise]:
                if prereq not in (parents | children):
                    self.children_lookup[exercise].append(prereq)
            for postreq in postreq_table.get(exercise, []):
                if postreq not in (parents | children):
                    self.parents_lookup[exercise].append(prereq)
        self.reset_history()

    def reset_history(self):
        self.exercise_states = {}
        for exercise in self.all_exercise_names.keys():
            self.exercise_states[exercise] = []

    def update_with_history(self, history):
        self.reset_history()
        for event in history:
            exercise_slug = event['exercise']
            correct = event['correct']
            if correct:
                self.set_exercise_and_children_known(exercise_slug)
            else:
                self.set_exercise_and_parents_unknown(exercise_slug)

    def set_exercise_and_children_known(self, exercise_slug):
        for exercise in self.get_all_relatives(exercise_slug,
                                               relationship='child'):
            self.exercise_states[exercise].append(1)

    def set_exercise_and_parents_unknown(self, exercise_slug):
        for exercise in self.get_all_relatives(exercise_slug, 'parent'):
            self.exercise_states[exercise].append(0)

    def get_all_relatives(self, exercise_slug, relationship):
        if relationship == 'child':
            lookup_table = self.children_lookup
        if relationship == 'parent':
            lookup_table = self.parents_lookup
        children = set([exercise_slug])
        unexplored = set(lookup_table[exercise_slug])
        while unexplored:
            exploring = unexplored.pop()
            children.add(exploring)
            for child in lookup_table[exploring]:
                if child not in children and child not in unexplored:
                    unexplored.add(child)
        return children & set(self.all_exercise_names.keys())

    def get_unknown_relatives(self, exercise_slug, relationship):
        relatives = self.get_all_relatives(exercise_slug, relationship)
        return [r for r in relatives if len(self.exercise_states[r]) == 0]

    def unknown_exercises(self):
        for e in self.all_exercise_names.keys():
            if not self.exercise_states.get(e, []):
                yield e

    def get_expected_new_nodes(self, exercise, probability_correct):
        if len(self.exercise_states[exercise]) > 0:
            return 0
        children = self.get_unknown_relatives(exercise, 'child')

        parents = self.get_unknown_relatives(exercise, 'parent')
        return (probability_correct * len(children) +
                (1 - probability_correct) * len(parents))

    def progress(self, exercises):
        return (len([s for s in self.exercise_states.values() if s]) /
                float(len(exercises)))

    def get_predicted_accuracies(self, mirt_predictions):
        predictions = {}
        avg_mirt = (sum(mirt_predictions.values()) /
                    len(mirt_predictions))
        for exercise in self.all_exercise_names:
            states = self.exercise_states.get(exercise, [])
            avg_evidence = (sum(states) + avg_mirt) / (len(states) + 1)
                # (sum(states) +
                #     mirt_predictions.get(exercise, avg_mirt_prediction)) /
                # (len(states) + 1))
            predictions[exercise] = avg_evidence
        return predictions

    def get_structure_and_states(self, mirt_predictions):
        structure = {}
        structure['name'] = self.topic.title
        structure['children'] = []
        predicted_accuracies = self.get_predicted_accuracies(mirt_predictions)
        for topic in load_children_for(self.topic):
            data = {}
            data['name'] = topic['title']
            data['children'] = []
            for exercise in self.topic_to_exercise_mapping[topic['title']]:
                data['children'].append({
                    'name': self.all_exercise_names[exercise],
                    'prediction': predicted_accuracies[exercise],
                    'size': 1
                })
            if data['children']:
                data['prediction'] = (
                    sum(c['prediction'] for c in data['children']) /
                    len(data['children']))
            else:
                data['prediction'] = .5
            structure['children'].append(data)
        structure['prediction'] = (
            sum(c['prediction'] for c in structure['children']) /
            len(structure['children']))
        return structure


class FrontierEngine(mirt_engine.MIRTEngine):

    def __init__(self, model_data, topic_slug='math'):
        self.topic = topic_slug
        super(FrontierEngine, self).__init__(model_data)
        self.graph = ExerciseGraph(self.topic)

    def export_current_predictions_as_json(self, history):
        self.graph.update_with_history(history)
        predicted_accuracies = dict(
            (ex, self.estimated_exercise_accuracy(history, ex, False))
            for ex in self.graph.all_exercise_names if
            self.estimated_exercise_accuracy(history, ex, False))
        structure = self.graph.get_structure_and_states(predicted_accuracies)
        return structure

    def next_suggested_item(self, history):
        """Return an ItemSuggestion for this Engine's preferred next item."""
        metadata = {}
        # we want to be sure we are only choosing from exercises the user has
        # not seen too recently or too often.

        self.graph.update_with_history(history)
        eligible_exs = self.graph.unknown_exercises()
        if not eligible_exs:
            eligible_exs = self.graph.all_exercise_names
        if not history:
            # first question chosen randomly to avoid overuse of a single ex
            metadata["random"] = True

            # Find a random exercise that we haven't just seen
            ex = random.choice(list(eligible_exs))
        else:
            metadata["random"] = False
            # update ability estimates only once -- outside the loop
            self._update_abilities(history)

            max_info = float("-inf")
            max_info_ex = None
            for ex in eligible_exs:
                fi = self.node_information(history, ex)
                if fi > max_info:
                    max_info = fi
                    max_info_ex = ex
            ex = max_info_ex

        metadata["estimated_accuracy"] = self.estimated_exercise_accuracy(
            history, ex, False)
        self.export_current_predictions_as_json(history)
        return engine.ItemSuggestion(ex, metadata=metadata)

    def score(self, history):
        """Returns a float that is the overall score on this assessment.
        Caller beware: may not be useful of valid is the assessment if the
        assessment has not been fully completed.  Check if is_complete().
        """
        # use lots of steps when estimating score to make
        # the score seeem close to deterministic
        self._update_abilities(history, num_steps=1000)

        predicted_accuracies = np.asarray([
            self.estimated_exercise_accuracy(history, ex, False)
            for ex in self.exercises()], dtype=float)

        return np.mean(predicted_accuracies)

    def readable_score(self, history):
        score = self.score(history)
        return str(int(score * 100.0))

    # def progress(self, history):
    #     self.graph.update_with_history(history)
    #     return self.graph.progress(self.topic.get_exercises(True))

    def node_information(self, history, exercise_name):
        """Compute Fisher information for exercise at current ability."""
        p = self.estimated_exercise_accuracy(history, exercise_name, False)
        if exercise_name in self.exercise_ind_dict:
            a = self.theta.W_correct[self.exercise_ind_dict[exercise_name],
                                     :-1]

            # TODO(jascha) double check this formula for the multidimensional
            # case
            fisher_info = np.sum(a ** 2) * p * (1. - p)

            return fisher_info
        if p is None:
            p = self.score(history)
        return self.graph.get_expected_new_nodes(exercise_name, p)

    def estimated_exercise_accuracy(
            self, history, exercise_name,
            update_abilities=True, ignore_analytics=False):
        """Returns the expected probability of getting a future question
        correct on the specified exercise.
        """
        if update_abilities:
            self._update_abilities(history, ignore_analytics=ignore_analytics)
        # try:
        #     exercise_ind = mirt_util.get_exercise_ind(
        #         exercise_name, self.exercise_ind_dict)
        # except KeyError:
        #     # If we don't have this exercise, predict the mean predicted
        #     # accuracy over all exercises we do have.
        #     return self.score(history)
        # return mirt_util.conditional_probability_correct(
        #     self.abilities, self.theta, exercise_ind)[0]
        return self


def get_cached_mission_data(slug):
    """Cache exercise orderings to use in getting related tasks
    """
    data = json.load(open('mirt/topic_data.json', 'r'))
    return data[slug]


def load_children_for(slug):
    """Get the children of each topic
    """
    data = json.load(open('mirt/topic_children.json', 'r'))
    return data[slug]
