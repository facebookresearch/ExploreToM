# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
import copy
import dill
import heapq
import random
import argparse
import collections
import itertools
import numpy as np
import pandas as pd
from utils import ModelCallHandler, ModelAnswerEvaluator, _dump_dict_list_to_jsonl
from belief_tracker import FullBeliefTracker, QuestionGenerator


def _setup_config_for_action_sampling(interaction_types_included):
    differentiating_actions = []
    if interaction_types_included in ["tomi", "tomi+asymmetric"]:
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "move_object_container",
        ]
        differentiating_actions = ["move_object_container"]
    elif interaction_types_included in [
        "tomi+object-state",
        "tomi+object-state+asymmetric",
    ]:
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "move_object_container",
            "update_object_state",
        ]
        differentiating_actions = ["update_object_state", "move_object_container"]
    elif interaction_types_included in [
        "tomi-object-state",
        "tomi-object-state+asymmetric",
    ]:
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "update_object_state",
        ]
        differentiating_actions = ["update_object_state"]
    elif interaction_types_included in [
        "tomi+room-changes",
        "tomi+room-changes+asymmetric",
    ]:
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "move_object_container",
            "move_object_room",
        ]
        differentiating_actions = ["move_object_room"]
    elif interaction_types_included in ["fantom-public", "fantom-public+asymmetric"]:
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "publicly_tell_about_abstract_topic",
        ]
        differentiating_actions = ["publicly_tell_about_abstract_topic"]
    elif interaction_types_included in ["fantom-private", "fantom-private+asymmetric"]:
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "privately_tell_about_abstract_topic",
        ]
        differentiating_actions = ["privately_tell_about_abstract_topic"]
    elif interaction_types_included in ["fantom", "fantom+asymmetric"]:
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "privately_tell_about_abstract_topic",
            "publicly_tell_about_abstract_topic",
        ]
        differentiating_actions = [
            "publicly_tell_about_abstract_topic",
            "privately_tell_about_abstract_topic",
        ]

    elif interaction_types_included in [
        "tomi+info-exchange",
        "tomi+info-exchange+asymmetric",
    ]:
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "move_object_container",
            "privately_tell_about_object_container_location",
            "publicly_tell_about_object_container_location",
        ]
        differentiating_actions = [
            "publicly_tell_about_object_container_location",
            "privately_tell_about_object_container_location",
        ]
    elif interaction_types_included in [
        "allbutfantom",
        "allbutfantom+asymmetric",
        "tomi+room-changes+info-exchange",
        "tomi+room-changes+info-exchange+asymmetric",
    ]:
        assert num_rooms > 1
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "move_object_container",
            "privately_tell_about_object_container_location",
            "publicly_tell_about_object_container_location",
            "move_object_room",
            "privately_tell_about_object_room_location",
            "publicly_tell_about_object_room_location",
        ]
        differentiating_actions = [
            "publicly_tell_about_object_container_location",
            "privately_tell_about_object_container_location",
            "privately_tell_about_object_room_location",
            "publicly_tell_about_object_room_location",
        ]
    elif interaction_types_included in [
        "all",
        "all+asymmetric",
        "tomi+fantom+room-changes+info-exchange",
        "tomi+fantom+room-changes+info-exchange+asymmetric",
    ]:
        assert num_rooms > 1
        all_allowed_actions = [
            "enter_room",
            "leave_room",
            "move_object_container",
            "privately_tell_about_object_container_location",
            "publicly_tell_about_object_container_location",
            "move_object_room",
            "privately_tell_about_object_room_location",
            "publicly_tell_about_object_room_location",
            "privately_tell_about_abstract_topic",
            "publicly_tell_about_abstract_topic",
        ]
        differentiating_actions = [
            "publicly_tell_about_object_container_location",
            "privately_tell_about_object_container_location",
            "privately_tell_about_object_room_location",
            "publicly_tell_about_object_room_location",
        ]
    else:
        assert False, interaction_types_included

    world_building_actions = [
        "move_object_container",
        "update_object_state",
        "privately_tell_about_abstract_topic",
        "publicly_tell_about_abstract_topic",
        "move_object_room",
    ]

    assert all(a in all_allowed_actions for a in differentiating_actions), (
        "Necessary actions should be a subset of all actions"
        + str(world_building_actions)
        + str(differentiating_actions)
    )

    #### Modify output if requested peeking and distracted people setting ###
    peeking_distracted_suffix = "_peeking_distracted"
    world_building_actions.extend(
        [c + peeking_distracted_suffix for c in world_building_actions]
    )

    # special case since move_object_room+asymmetric does not exist
    if interaction_types_included == "tomi+room-changes+asymmetric":
        # add asymmetric actions to the list of all all_allowed_actions
        all_allowed_actions.append("move_object_container" + peeking_distracted_suffix)
    elif "asymmetric" in interaction_types_included:
        # add asymmetric actions to the list of all all_allowed_actions
        for a in differentiating_actions:
            all_allowed_actions.append(a + peeking_distracted_suffix)

        # replace necessary actions to only include stuff
        differentiating_actions = [
            a + peeking_distracted_suffix for a in differentiating_actions
        ]

    assert all(a in all_allowed_actions for a in differentiating_actions), (
        "Necessary actions should be a subset of all actions"
        + str(world_building_actions)
        + str(differentiating_actions)
    )

    return differentiating_actions, all_allowed_actions, world_building_actions


class ChallengingStoryStructureSearcher:
    UPPERBOUND_NUM_NODES_TO_BE_EXPLORED_IN_A_STORY = 5000  # this is just a safety measure to not loop forever, but it should not happen!
    BASE_LOOKAHEAD = 50
    MAX_PEEKING_DISTRACTED_ACTIONS_ALLOWED_PER_STORY = 1

    def __init__(
        self,
        model_call_handler,
        interaction_types_included,
        num_people,
        num_moves,
        num_rooms,
        raw_rooms=False,
        use_custom_names_dict=None,
        **kwargs,
    ):
        self.model_call_handler = model_call_handler
        self.evaluator = ModelAnswerEvaluator(model_call_handler)

        # Search parameters
        self.group_n_next_steps = kwargs.get("group_n_next_steps", 3)
        self.a_star_g_version = kwargs.get("a_star_g_version", "acc")
        self.a_star_h_version = kwargs.get("a_star_h_version", "intstory_w_people")
        self.a_star_h_weight = kwargs.get("a_star_h_weight", 0.1)
        self.a_star_neighbor_priority = kwargs.get(
            "a_star_neighbor_priority", "weight-goal5"
        )
        self.max_neighbors_to_explore = kwargs.get(
            "max_neighbors_to_explore", "1_10-3_0.75-5_0.5"
        )
        self.max_neighbors_to_explore_conditions = (
            self._extract_max_neighbors_to_explore_conditions(
                self.max_neighbors_to_explore
            )
        )

        # Story constraint parameters
        self.num_people, self.num_moves, self.num_rooms = (
            num_people,
            num_moves,
            num_rooms,
        )
        (
            self.people_in_game,
            self.rooms,
            self.objects,
            self.container_names,
            self.topics,
            self.object_states,
        ) = self._initialize_story_context_parameters(use_custom_names_dict, raw_rooms)
        self.interaction_types_included = interaction_types_included
        (
            differentiating_actions,
            all_allowed_actions,
            world_building_actions,
        ) = _setup_config_for_action_sampling(interaction_types_included)
        self.differentiating_actions = differentiating_actions
        self.all_allowed_actions = all_allowed_actions
        self.world_building_actions = world_building_actions

        # Logs & Cache
        self.solved_story_cache = {}
        self.logs = []

        # experiment where we do not actually run search, just one generation to compare the power of searching vs not searching
        self.experiment_to_run = kwargs.get("experiment_to_run", "search")

        # FIXME consider not counting peeking as a separate action!
        self.experiment_variations = kwargs.get(
            "experiment_variations",
            [
                "count_peeking_distracted_as_action",
                "verify_final_story_for_state_updates",
            ],
        )

        MULTIPLIER_FILES = [
            "tomi+room-changes",
            "tomi+room-changes+info-exchange",
            "tomi+fantom+room-changes+info-exchange",
            "all",
            "allbutfantom",
        ]
        MULTIPLIER_FILES.extend([a + "+asymmetric" for a in MULTIPLIER_FILES])
        self.multiplier_neighbor_priority = (
            5 if interaction_types_included in MULTIPLIER_FILES else 1
        )

    ### ****** utils to initialize setting ******
    def _extract_max_neighbors_to_explore_conditions(self, max_neighbors_to_explore):
        max_neighbors_to_explore_conditions = []
        if isinstance(max_neighbors_to_explore, str):
            # if it's a string, it's describing number of neighbors under different circumstances
            # "1_10-3_0.75-5_0.5" means it's 5 neigh if acc <= 0.5 ; 3 if acc <= 0.75 ; 1 if acc <= 10 (to include everything)
            for class_description in max_neighbors_to_explore.split("-"):
                num_neigh, threshold = class_description.split("_")
                max_neighbors_to_explore_conditions.append(
                    (float(threshold), int(num_neigh))
                )
            max_neighbors_to_explore_conditions = sorted(
                max_neighbors_to_explore_conditions
            )
        elif isinstance(max_neighbors_to_explore, int):
            max_neighbors_to_explore_conditions = [(10.0, max_neighbors_to_explore)]
        else:
            assert False
        return max_neighbors_to_explore_conditions

    def _initialize_story_context_parameters(
        self, use_custom_names_dict, raw_rooms=True
    ):
        if use_custom_names_dict:
            people_names = use_custom_names_dict["people_names"]
            container_names = use_custom_names_dict["container_names"]
            rooms = use_custom_names_dict["rooms"]
            objects = use_custom_names_dict["objects"]
            topics = use_custom_names_dict["topics"]
            object_states = use_custom_names_dict.get("class_equivalences", {})

            people_in_game = people_names[: self.num_people]
            container_names = container_names[: self.num_moves]
            rooms = rooms[: self.num_rooms]

        elif raw_rooms:
            people_names = [
                "Person1",
                "Person2",
                "Person3",
                "Person4",
                "Person5",
                "Person6",
                "Person7",
            ]
            container_names = [
                "container1",
                "container2",
                "container3",
                "container4",
                "container5",
                "container6",
            ]
            rooms = ["room1", "room2", "room3"]
            objects = ["object1"]
            topics = ["topic1", "topic2", "topic3"]
            object_states = {}

            people_in_game = random.sample(people_names, self.num_people)
            container_names = random.sample(container_names, self.num_moves)
            rooms = random.sample(rooms, self.num_rooms)

        else:
            people_names = [
                "Anne",
                "Beth",
                "Charles",
                "Diane",
                "Esther",
                "Fanny",
                "George",
            ]
            container_names = ["box", "basket", "chest", "black bin", "crate", "drawer"]
            rooms = ["kitchen", "bedroom"]
            objects = ["apple"]
            topics = []
            object_states = {}

            people_in_game = random.sample(people_names, self.num_people)
            container_names = random.sample(container_names, self.num_moves)
            rooms = random.sample(rooms, self.num_rooms)

        return people_in_game, rooms, objects, container_names, topics, object_states

    ### ****** utils to measure story properties ******
    def _count_actions(self, action_type_counter):
        num_differentiating_actions_performed = sum(
            [
                action_type_counter[k]
                for k in self.differentiating_actions
                if k in action_type_counter
            ]
        )
        num_moves_performed = sum(
            [
                action_type_counter[k]
                for k in self.world_building_actions
                if k in action_type_counter
            ]
        )
        return num_differentiating_actions_performed, num_moves_performed

    def _story_will_never_be_final_wrt_actions(self, action_type_counter):
        """
        Check if a story has too many actions, and thus will never become a final story.
        By construction, it's impossible to have too many people or rooms, as we only keep the exact number needed when building the class.
        """
        # NEW: check if this story has too many secret/distractions
        num_peeking_or_distracted = sum(
            v for k, v in action_type_counter.items() if "_peeking_distracted" in k
        )
        too_many_secrets = (
            num_peeking_or_distracted
            > self.MAX_PEEKING_DISTRACTED_ACTIONS_ALLOWED_PER_STORY
        )

        # if this story has more actions than requested, we should discard
        return (
            sum([action_type_counter[k] for k in self.world_building_actions])
            > self.num_moves
            or too_many_secrets
        )

    def _is_story_final_wrt_actions_and_people(
        self, action_type_counter, num_people_involved, count_differentiating_actions=1
    ):
        (
            num_differentiating_actions_performed,
            num_moves_performed,
        ) = self._count_actions(action_type_counter)

        # 'differentiating actions', actions that make this set special so we really want to include them. There should be at least 1
        differentiating_actions_performed = (
            num_differentiating_actions_performed >= count_differentiating_actions
        )

        # number of canon events (real world/knowledge changes) needs to be exactly X. Easy for generating test sets.
        necessary_num_moves_performed = num_moves_performed == self.num_moves

        # all required people should be involved in the story
        all_people_involved = num_people_involved >= self.num_people

        # at most one secret/distraction per story
        # (in this case, the differentiating actions are only consisting of _peeking_distracted, but we want to make sure there is exactly one)
        num_peeking_or_distracted = sum(
            v for k, v in action_type_counter.items() if "_peeking_distracted" in k
        )
        not_too_many_secrets = (
            num_peeking_or_distracted
            <= self.MAX_PEEKING_DISTRACTED_ACTIONS_ALLOWED_PER_STORY
        )
        return (
            differentiating_actions_performed
            and necessary_num_moves_performed
            and all_people_involved
            and not_too_many_secrets
        )

    def _is_false_belief_story(self, bt):
        return QuestionGenerator(bt).is_false_belief_story()

    def _check_if_node_is_final(
        self, belief_tracker, action_type_counter, cur_a_star_g
    ):
        # 0. stopping conditions
        num_people_involved = (
            len(belief_tracker.people)
            if self.a_star_h_version.startswith("intstory_w_people")
            or self.a_star_h_version.startswith("constant")
            or self.a_star_h_version.startswith("intstorywolook")
            else 1e9
        )
        num_rooms_involved = (
            len(belief_tracker.rooms)
            if self.a_star_h_version.startswith("intstory_w_people")
            or self.a_star_h_version.startswith("constant")
            or self.a_star_h_version.startswith("intstorywolook")
            else 1e9
        )
        assert (
            num_people_involved < 1e9 and num_rooms_involved < 1e9
        ), "We are not using this right now!"

        desirable_a_star_g = cur_a_star_g < 1 - 1e-6
        # this is an experiment to know if finding hard stories is important for training or not
        # here, a_star_g = 1 - actual_accuracy and we want actual_accuracy == 1, thus we want a_star_g = 0
        if self.a_star_h_version == "intstory_w_people_keepeasy":
            desirable_a_star_g = -1e-6 < cur_a_star_g < 1e-6

        # note: has_asymmetry check seems to be redundant
        has_asymmetry = (
            sum(v for k, v in action_type_counter.items() if "_peeking_distracted" in k)
            > 0
            or "asymmetric" not in self.interaction_types_included
        )
        return (
            self._is_story_final_wrt_actions_and_people(
                action_type_counter,
                num_people_involved,
                count_differentiating_actions=1,
            )
            and num_rooms_involved >= self.num_rooms
            and desirable_a_star_g
            and has_asymmetry
        )

    ### ****** utils for computing A* functions ******
    def _compute_a_star_g(self, belief_tracker1, num_nodes_evaluated):
        # 1. Calculate cost of the path so far, including new action
        if self.a_star_g_version in ["acc", "acc2", "acc-int", "acc2-int"]:
            # HEURISTIC: 3 sentences or less ==> accuracy = 1 ; empirically true
            heuristic_always_return_correct = len(belief_tracker1.story_script) <= 3

            keep_only_interesting_qs = self.a_star_g_version.endswith("int")
            responses, tokens_used = self.evaluator.evaluate_story(
                belief_tracker1,
                nth=2 if self.a_star_g_version.startswith("acc2") else 1,
                keep_only_interesting_qs=keep_only_interesting_qs,
                heuristic_always_return_correct=heuristic_always_return_correct,
            )
            num_nodes_evaluated += len(responses) > 0
            is_answer_correct = (
                np.mean([int(r["is_correct"]) for r in responses]) if responses else 2
            )
            a_star_g = np.mean(
                is_answer_correct
            )  # f is in [0, 1], and we prefer lower values
        elif self.a_star_g_version in ["constant", "constant-int"]:
            a_star_g = -7
            responses = []
        else:
            assert False
        return a_star_g, num_nodes_evaluated, responses

    def _compute_a_star_h(self, lookahead_stories, action_type_counter1):
        is_false_belief_story_list = []
        if self.a_star_h_version.startswith("avg_story_len-true_belief_multiplier-"):
            true_belief_story_multiplier = float(self.a_star_h_version.split("-")[-1])
            scores = []
            for bt, _ in lookahead_stories:
                is_false_belief_story = self._is_false_belief_story(bt)
                is_false_belief_story_list.append(is_false_belief_story)
                scores.append(
                    len(bt.story_script)
                    if is_false_belief_story
                    else max_sentences * true_belief_story_multiplier
                )
            a_star_h = (
                np.mean(scores) / max_sentences
            )  # shorter stories are preferred; true belief stories are heavily discouraged
        elif self.a_star_h_version == "avg_story_len":
            a_star_h = avg_story_len
        elif self.a_star_h_version == "constant":
            a_star_h = 0
        elif self.a_star_h_version.startswith("false_belief_ratio"):
            is_false_belief_story_list = []
            for bt, _ in lookahead_stories:
                is_false_belief_story = self._is_false_belief_story(bt)
                is_false_belief_story_list.append(is_false_belief_story)
            a_star_h = 1 - np.mean(
                is_false_belief_story_list
            )  # more false belief is better
        elif self.a_star_h_version.startswith("intstory_w_people"):
            is_ok_story_list = []
            for bt, action_type_counter in lookahead_stories:
                is_ok_story_list.append(
                    self._is_story_final_wrt_actions_and_people(
                        action_type_counter,
                        len(bt.people),
                        count_differentiating_actions=1,
                    )
                )
            a_star_h = 1 - np.mean(is_ok_story_list)  # more ok is better
        else:
            assert False
        return a_star_h, is_false_belief_story_list

    def _compute_average_lookahead_story_length(
        self,
        function_list1,
        belief_tracker1,
        action_type_counter1,
        is_already_final_story_wrt_actions_and_people,
        max_sentences,
    ):
        if is_already_final_story_wrt_actions_and_people:
            # once a story is final (wrt to actions and people) we do not continue it further, hence itself is the only lookahead
            lookahead_stories = [(belief_tracker1, action_type_counter1)]
        else:
            lookahead_stories = []
            for _ in range(self.BASE_LOOKAHEAD):
                (
                    belief_tracker_copy,
                    function_list_copy,
                    action_type_counter_copy,
                ) = copy.deepcopy(
                    [belief_tracker1, function_list1, action_type_counter1]
                )
                (
                    _,
                    belief_tracker_output,
                    action_type_counter_output,
                    _,
                ) = self.complete_story_randomly_and_incrementally(
                    belief_tracker_copy,
                    function_list_copy,
                    action_type_counter_copy,
                    max_sentences,
                    no_retries=True,
                )

                lookahead_stories.append(
                    (belief_tracker_output, action_type_counter_output)
                )
        avg_story_len = (
            np.mean([len(bt.story_script) for bt, _ in lookahead_stories])
            / max_sentences
        )
        return lookahead_stories, avg_story_len

    ### ****** utils for next node selection and prioritization ******
    def _skip_node(
        self,
        belief_tracker,
        action_type_counter,
        num_nodes_evaluated,
        max_nodes_allowed_to_explore,
    ):
        num_people_involved = (
            len(belief_tracker.people)
            if self.a_star_h_version.startswith("intstory_w_people")
            or self.a_star_h_version.startswith("constant")
            or self.a_star_h_version.startswith("intstorywolook")
            else 1e9
        )
        num_rooms_involved = (
            len(belief_tracker.rooms)
            if self.a_star_h_version.startswith("intstory_w_people")
            or self.a_star_h_version.startswith("constant")
            or self.a_star_h_version.startswith("intstorywolook")
            else 1e9
        )

        # [impossible for this to become a final node] this is a node that already fulfills everything but the final accuracy was too high
        # let's not keep expanding this node, since the actions from now on will not have any canon action
        if (
            self._is_story_final_wrt_actions_and_people(
                action_type_counter,
                num_people_involved,
                count_differentiating_actions=1,
            )
            and num_rooms_involved >= self.num_rooms
        ):
            return True

        # [impossible for this to become a final node] too many moves, do not even evaluate
        if self._story_will_never_be_final_wrt_actions(action_type_counter):
            return True

        # do not explore further if we've explored too much, but it's important to check if the remaining nodes in the queue are final nodes
        if num_nodes_evaluated >= max_nodes_allowed_to_explore:
            return True

        return False

    def _list_valid_node_neighbors(
        self,
        belief_tracker,
        function_list,
        action_type_counter,
        max_sentences,
        max_neighbors_to_explore,
    ):
        """
        We list valid node neighbors. If sampling just one action then we just
        enumerate all possible next nodes, otherwise we sample a large quantity (max_neighbors_to_explore * 50).
        """
        valid_next_actions = []
        if self.group_n_next_steps > 1:
            already_considered_narratives = set([])
            for z in range(max_neighbors_to_explore * 50):
                (
                    belief_tracker_copy,
                    function_list_copy,
                    action_type_counter_copy,
                ) = copy.deepcopy([belief_tracker, function_list, action_type_counter])
                max_sentences_for_this_step = min(
                    max_sentences,
                    len(belief_tracker.story_script) + self.group_n_next_steps,
                )
                (
                    function_list1,
                    belief_tracker1,
                    action_type_counter1,
                    function_identifier1,
                ) = self.complete_story_randomly_and_incrementally(
                    belief_tracker_copy,
                    function_list_copy,
                    action_type_counter_copy,
                    max_sentences_for_this_step,
                    no_retries=True,
                )

                # if the procedure added new sentences, include it to the random next steps
                if self._story_will_never_be_final_wrt_actions(action_type_counter1):
                    continue

                narrative = " ".join(belief_tracker1.story_script)
                if (
                    len(belief_tracker1.story_script) > len(belief_tracker.story_script)
                    and narrative not in already_considered_narratives
                ):
                    valid_next_actions.append(
                        [
                            belief_tracker1,
                            function_list1,
                            action_type_counter1,
                            function_identifier1,
                        ]
                    )
                    already_considered_narratives.add(narrative)

                # if self.experiment_to_run == 'baseline':
                #    break  # we need only one valid path for the baseline setting
        else:
            banned_function_identifiers = set()
            valid_next_actions = []
            for action in self.all_allowed_actions:
                function_identifier = True
                while function_identifier is not None:
                    (
                        belief_tracker1,
                        function_list1,
                        action_type_counter1,
                    ) = copy.deepcopy(
                        [belief_tracker, function_list, action_type_counter]
                    )
                    (
                        belief_tracker1,
                        function_list1,
                        action_type_counter1,
                        function_identifier,
                    ) = self._execute_next_sampled_action(
                        belief_tracker1, function_list1, action_type_counter1, action
                    )
                    if (
                        not self._story_will_never_be_final_wrt_actions(
                            action_type_counter1
                        )
                        and function_identifier is not None
                    ):
                        banned_function_identifiers.add(function_identifier)
                        valid_next_actions.append(
                            (
                                belief_tracker1,
                                function_list1,
                                action_type_counter1,
                                function_identifier,
                            )
                        )
            assert all(
                [
                    len(b.story_script_tuple_representation)
                    == len(belief_tracker.story_script_tuple_representation)
                    + 1
                    + ("_peeking" in str(fn) or "_distracted" in str(fn))
                    for b, _, _, fn in valid_next_actions
                ]
            )

        return valid_next_actions

    def _prioritize_node_neighbors_to_analyze(
        self, valid_next_actions, belief_tracker, action_type_counter
    ):
        """
        Given a list of valid next actions, prioritize them according to some predefined criterion.
        """
        # prioritize order in which to analyze next valid actions
        if self.a_star_neighbor_priority == "random":
            random.shuffle(valid_next_actions)
        elif (
            self.a_star_neighbor_priority == "weight-goal4"
            or self.a_star_neighbor_priority == "weight-goal5"
        ):
            #### YOU SHALL NOT SAMPLE SOMETHING THAT DOES NOT ADVANCE THE NARRATIVE

            cur_num_diff_actions, cur_num_moves = self._count_actions(
                action_type_counter
            )
            cur_num_people = len(belief_tracker.people)

            # this computes whether the current story already solved some axes of the requirements, and we'll force the next step to advance the story somehow
            is_unsolved_diff_actions = cur_num_diff_actions < 1
            is_unsolved_moves = cur_num_moves < self.num_moves
            is_unsolved_people = cur_num_people < self.num_people

            # now we compute how close each step would be to our goal in terms of (p, m, r)
            temp_weights = [
                (self._count_actions(ac1), len(bt1.people))
                for bt1, _, ac1, _ in valid_next_actions
            ]
            temp_weights = [
                (a - cur_num_diff_actions)
                * is_unsolved_diff_actions
                * self.multiplier_neighbor_priority
                + (m - cur_num_moves) * is_unsolved_moves
                + (p - cur_num_people) * is_unsolved_people
                for (a, m), p in temp_weights
            ]
            num_non_zero_temp_weights = len([t for t in temp_weights if t > 1e-5])
            if np.sum(temp_weights) < 1e-5:
                valid_next_actions = []
            else:
                temp_weights = (temp_weights / np.sum(temp_weights)).tolist()

                if self.a_star_neighbor_priority == "weight-goal5":
                    temp_weights = [e**2 for e in temp_weights]

                # shuffle valid next actions based on the weights computed above (higher is more likely to be sampled)
                temp_array = list(range(len(temp_weights)))
                shuffled_array = []
                while len(shuffled_array) < num_non_zero_temp_weights:
                    chosen_element_index = random.choices(
                        temp_array, weights=temp_weights, k=1
                    )[0]
                    shuffled_array.append(valid_next_actions[chosen_element_index])
                    index = temp_array.index(chosen_element_index)
                    temp_array.pop(index)
                    temp_weights.pop(index)
                valid_next_actions = shuffled_array
        else:
            assert False
        return valid_next_actions

    def _story_passes_length_regularization(
        self,
        belief_tracker1,
        belief_tracker,
        action_type_counter1,
        avg_story_len,
        cur_avg_story_len,
        max_sentences,
    ):
        constant_better_avg_len = (
            float(self.a_star_h_version.split("better_avg_len-")[-1].split("-")[0])
            if "better_avg_len-" in self.a_star_h_version
            else None
        )

        new_sentences = len(belief_tracker1.story_script) - len(
            belief_tracker.story_script
        )
        assert new_sentences >= 0
        if (
            "better_avg_len-" in self.a_star_h_version
            and avg_story_len
            >= cur_avg_story_len
            + constant_better_avg_len * new_sentences / max_sentences
        ):
            self.logs.append(
                {
                    "node": " ".join(belief_tracker1.story_script),
                    "a_star_g_version": self.a_star_g_version,
                    "a_star_h_version": self.a_star_h_version,
                    "a_star_hv_weight": self.a_star_h_weight,
                    "is_final": "REJECTED",
                    "is_final_without_verification": "REJECTED",
                    "action_type_counter": action_type_counter1.most_common(),
                    "max_sentences": max_sentences,
                    "prev_avg_story_len": cur_avg_story_len,
                    "avg_story_len": avg_story_len,
                }
            )
            return False
        return True

    ### ****** utils for executing next action(s) ******
    def _list_all_peeking_and_distracted(self, random_action):
        extra_args_list = []
        for p1, p2 in itertools.product(
            random.sample(self.people_in_game, len(self.people_in_game)),
            random.sample(self.people_in_game, len(self.people_in_game)),
        ):
            extra_args = {}
            if "_peeking" in random_action:
                extra_args["people_peeking"] = [p1]
            if "_distracted" in random_action:
                extra_args["people_distracted"] = [p2]
            if len(set(p1) & set(p2)) > 0:
                continue  # no one can be distracted and peeking at the same time
            extra_args_list.append(copy.deepcopy(extra_args))

        for p1 in random.sample(self.people_in_game, len(self.people_in_game)):
            extra_args = {}
            if "_peeking" in random_action:
                extra_args["people_peeking"] = [p1]
            if "_distracted" in random_action:
                extra_args["people_distracted"] = []
            extra_args_list.append(copy.deepcopy(extra_args))

        for p2 in random.sample(self.people_in_game, len(self.people_in_game)):
            extra_args = {}
            if "_peeking" in random_action:
                extra_args["people_peeking"] = []
            if "_distracted" in random_action:
                extra_args["people_distracted"] = [p2]
            extra_args_list.append(copy.deepcopy(extra_args))
        return extra_args_list

    def _execute_next_sampled_action(
        self,
        belief_tracker,
        function_list,
        action_type_counter,
        random_action,
        banned_function_identifiers=set(),
    ):
        """
        :param action_type_counter: counter to update how many action types of each type were made
        :param banned_function_identifiers: this allows to explicitly exclude some actions
        """
        cur_function_list = len(function_list)

        if random_action.startswith("enter_room"):
            for p, room_name, extras in itertools.product(
                random.sample(self.people_in_game, len(self.people_in_game)),
                random.sample(self.rooms, len(self.rooms)),
                self._list_all_peeking_and_distracted(random_action),
            ):
                function_identifier = (random_action, p, room_name, str(extras))
                if function_identifier in banned_function_identifiers:
                    continue
                output_flag = belief_tracker.enter_room(p, room_name, **extras)
                if output_flag:
                    function_list.append(
                        lambda o, person=p, room=room_name, extra_args=extras: o.enter_room(
                            person, room, **extra_args
                        )
                    )
                    action_type_counter[random_action] += 1
                    break
        elif random_action.startswith("leave_room"):
            for p, room_name, extras in itertools.product(
                random.sample(self.people_in_game, len(self.people_in_game)),
                random.sample(self.rooms, len(self.rooms)),
                self._list_all_peeking_and_distracted(random_action),
            ):
                function_identifier = (random_action, p, room_name, str(extras))
                if function_identifier in banned_function_identifiers:
                    continue
                output_flag = belief_tracker.leave_room(p, room_name, **extras)
                if output_flag:
                    function_list.append(
                        lambda o, person=p, room=room_name, extra_args=extras: o.leave_room(
                            person, room, **extra_args
                        )
                    )
                    action_type_counter[random_action] += 1
                    break
        elif random_action.startswith("move_object_container"):
            for p, obj1, container1, extras in itertools.product(
                random.sample(self.people_in_game, len(self.people_in_game)),
                random.sample(self.objects, len(self.objects)),
                random.sample(self.container_names, len(self.container_names)),
                self._list_all_peeking_and_distracted(random_action),
            ):
                function_identifier = (random_action, p, obj1, container1, str(extras))
                if function_identifier in banned_function_identifiers:
                    continue
                output_flag = belief_tracker.move_object_container(
                    p, obj1, container1, **extras
                )
                if output_flag:
                    function_list.append(
                        lambda o, person=p, obj=obj1, container=container1, extra_args=extras: o.move_object_container(
                            person, obj, container, **extra_args
                        )
                    )
                    action_type_counter[random_action] += 1
                    break
        elif random_action == "move_object_room":
            for p, obj1, room1 in itertools.product(
                random.sample(self.people_in_game, len(self.people_in_game)),
                random.sample(self.objects, len(self.objects)),
                random.sample(self.rooms, len(self.rooms)),
            ):
                function_identifier = (
                    random_action,
                    p,
                    obj1,
                    room1,
                    "",
                )  # empty extras to match format of other identifiers
                if function_identifier in banned_function_identifiers:
                    continue
                output_flag = belief_tracker.move_object_room(p, obj1, room1)
                if output_flag:
                    function_list.append(
                        lambda o, person=p, obj=obj1, container=room1: o.move_object_room(
                            person, obj, container
                        )
                    )
                    action_type_counter[random_action] += 1
                    break
        elif random_action.startswith("update_object_state"):
            output_flag = False
            for p, obj1, extras in itertools.product(
                random.sample(self.people_in_game, len(self.people_in_game)),
                random.sample(self.objects, len(self.objects)),
                self._list_all_peeking_and_distracted(random_action),
            ):
                filtered_object_states = []
                for k in [
                    "update_object_state-invisible",
                    "update_object_state-visible",
                ] & self.object_states.keys():
                    is_new_state_visible = k == "update_object_state-visible"
                    for elem in self.object_states[k]:
                        # FIXME: this cleanup should not be needed
                        elem_obj_name = elem["object_name"].lower().strip()
                        for start_to_remove in ["a ", "an ", "the "]:
                            elem_obj_name = (
                                elem_obj_name[len(start_to_remove) :]
                                if elem_obj_name.startswith(start_to_remove)
                                else elem_obj_name
                            )
                        if elem_obj_name == obj1:
                            object_state_value = elem["resulting_state"]
                            object_state_type = elem["property_name"]
                            text_description = elem["text_description"].format(
                                person=p, object=elem_obj_name
                            )
                            filtered_object_states.append(
                                (
                                    object_state_value,
                                    object_state_type,
                                    text_description,
                                    is_new_state_visible,
                                )
                            )

                for filtered_state in filtered_object_states:
                    function_identifier = (
                        random_action,
                        p,
                        obj1,
                        *filtered_state,
                        str(extras),
                    )
                    if function_identifier in banned_function_identifiers:
                        continue
                    output_flag = belief_tracker.update_object_state(
                        p, obj1, *filtered_state, **extras
                    )
                    if output_flag:
                        function_list.append(
                            lambda o, person=p, obj=obj1, state=filtered_state, extra_args=extras: o.update_object_state(
                                person, obj, *state, **extra_args
                            )
                        )
                        action_type_counter[random_action] += 1
                        break
                if output_flag:
                    break
        elif random_action.startswith("privately_tell_about_abstract_topic"):
            for p1, p2, topic1, extras in itertools.product(
                random.sample(self.people_in_game, len(self.people_in_game)),
                random.sample(self.people_in_game, len(self.people_in_game)),
                random.sample(self.topics, len(self.topics)),
                self._list_all_peeking_and_distracted(random_action),
            ):
                function_identifier = (random_action, p1, p2, topic1, str(extras))
                if function_identifier in banned_function_identifiers:
                    continue
                output_flag = belief_tracker.private_communication(
                    p1, p2, topic1, **extras
                )
                if output_flag:
                    function_list.append(
                        lambda o, person1=p1, person2=p2, topic=topic1, extra_args=extras: o.private_communication(
                            person1, person2, topic, **extra_args
                        )
                    )
                    action_type_counter[random_action] += 1
                    break
        elif random_action.startswith("publicly_tell_about_abstract_topic"):
            for p1, topic1, extras in itertools.product(
                random.sample(self.people_in_game, len(self.people_in_game)),
                random.sample(self.topics, len(self.topics)),
                self._list_all_peeking_and_distracted(random_action),
            ):
                function_identifier = (random_action, p1, topic1, str(extras))
                if function_identifier in banned_function_identifiers:
                    continue
                output_flag = belief_tracker.broadcast_communication(
                    p1, topic1, **extras
                )
                if output_flag:
                    function_list.append(
                        lambda o, person=p1, topic=topic1, extra_args=extras: o.broadcast_communication(
                            person, topic, **extra_args
                        )
                    )
                    action_type_counter[random_action] += 1
                    break
        elif any(
            random_action.startswith(t)
            for t in [
                "privately_tell_about_object_container_location",
                "privately_tell_about_object_room_location",
            ]
        ):
            cur_action_type = (
                FullBeliefTracker.CONTAINER_LOCATION
                if random_action.startswith(
                    "privately_tell_about_object_container_location"
                )
                else FullBeliefTracker.ROOM_LOCATION
            )

            output_flag = False
            for p1, p2, extras in itertools.product(
                random.sample(self.people_in_game, len(self.people_in_game)),
                random.sample(self.people_in_game, len(self.people_in_game)),
                self._list_all_peeking_and_distracted(random_action),
            ):
                if p1 not in belief_tracker.first_order_beliefs:
                    continue
                candidates = [
                    (k, value)
                    for k, ws in belief_tracker.first_order_beliefs[
                        p1
                    ].world_state.items()
                    for prop, value in ws.items()
                    if prop == cur_action_type and not value.startswith("not(")
                ]
                for entity, value in random.sample(candidates, len(candidates)):
                    topic1 = (entity, cur_action_type, value, True)  # FIXME True
                    function_identifier = (random_action, p1, p2, topic1, str(extras))
                    if function_identifier in banned_function_identifiers:
                        continue
                    output_flag = belief_tracker.private_communication(
                        p1, p2, topic=topic1, **extras
                    )
                    if output_flag:
                        function_list.append(
                            lambda o, person1=p1, person2=p2, topic=topic1, extra_args=extras: o.private_communication(
                                person1, person2, topic, **extra_args
                            )
                        )
                        action_type_counter[random_action] += 1
                        break
                if output_flag:
                    break
        elif any(
            random_action.startswith(t)
            for t in [
                "publicly_tell_about_object_container_location",
                "publicly_tell_about_object_room_location",
            ]
        ):
            cur_action_type = (
                FullBeliefTracker.CONTAINER_LOCATION
                if random_action.startswith(
                    "publicly_tell_about_object_container_location"
                )
                else FullBeliefTracker.ROOM_LOCATION
            )

            output_flag = False
            for p1, extras in itertools.product(
                random.sample(self.people_in_game, len(self.people_in_game)),
                self._list_all_peeking_and_distracted(random_action),
            ):
                if p1 not in belief_tracker.first_order_beliefs:
                    continue
                candidates = [
                    (k, value)
                    for k, ws in belief_tracker.first_order_beliefs[
                        p1
                    ].world_state.items()
                    for prop, value in ws.items()
                    if prop == cur_action_type and not value.startswith("not(")
                ]
                for entity, value in random.sample(candidates, len(candidates)):
                    topic1 = (entity, cur_action_type, value, True)  # FIXME True
                    function_identifier = (random_action, p1, topic1, str(extras))
                    if function_identifier in banned_function_identifiers:
                        continue
                    output_flag = belief_tracker.broadcast_communication(
                        p1, topic=topic1, **extras
                    )
                    if output_flag:
                        function_list.append(
                            lambda o, person1=p1, topic=topic1, extra_args=extras: o.broadcast_communication(
                                person1, topic, **extra_args
                            )
                        )
                        action_type_counter[random_action] += 1
                        break
                if output_flag:
                    break

        if cur_function_list == len(function_list):
            function_identifier = None  # this means that we tried all random_action combinations and failed
        return belief_tracker, function_list, action_type_counter, function_identifier

    def complete_story_randomly_and_incrementally(
        self,
        belief_tracker,
        function_list,
        action_type_counter,
        max_sentences,
        no_retries=False,
    ):
        (
            belief_tracker_orig,
            function_list_orig,
            action_type_counter_orig,
        ) = copy.deepcopy([belief_tracker, function_list, action_type_counter])
        function_identifier = None
        while (
            not self._is_story_final_wrt_actions_and_people(
                action_type_counter,
                len(belief_tracker.people),
                count_differentiating_actions=1,
            )
            and len(function_list) < max_sentences
        ):
            random_action = random.sample(self.all_allowed_actions, 1)[0]
            (
                belief_tracker,
                function_list,
                action_type_counter,
                function_identifier,
            ) = self._execute_next_sampled_action(
                belief_tracker,
                function_list,
                action_type_counter,
                random_action,
            )

        if not no_retries and not self._is_story_final_wrt_actions_and_people(
            action_type_counter,
            len(belief_tracker.people),
            count_differentiating_actions=1,
        ):
            return self.complete_story_randomly_and_incrementally(
                belief_tracker_orig,
                function_list_orig,
                action_type_counter_orig,
                max_sentences,
            )
        return function_list, belief_tracker, action_type_counter, function_identifier

    ### ****** utils for logging ******
    def __build_log_entry(
        self,
        node,
        a_star_g,
        a_star_h,
        responses,
        is_final_without_verification,
        is_final,
        action_type_counter1,
        max_sentences,
        avg_story_len,
        belief_tracker1,
        is_false_belief_story_list,
    ):
        tmp = {
            "node": node,
            "a_star_g": a_star_g,
            "a_star_h": a_star_h,
            "a_star_f": a_star_g + a_star_h * self.a_star_h_weight,
            "model_responses_f": responses,
            "is_final_without_verification": is_final_without_verification,
            "is_final": is_final,
            "action_type_counter": action_type_counter1.most_common(),
            "max_sentences": max_sentences,
            "avg_story_len": avg_story_len,
            "people": list(belief_tracker1.people),
            "is_false_belief_story_list": is_false_belief_story_list,
        }

        return tmp

    def _verify_final_story_structure(self, belief_tracker, function_list):
        """
        Optional verification for the state_update case, where actions might depend from the LLM being of good quality.
        """
        if (
            "verify_final_story_for_state_updates" not in self.experiment_variations
            or "state" not in self.interaction_types_included
        ):
            return True

        from story_structure_infiller import StoryInfillingQualityVerifier

        quality_verifier = StoryInfillingQualityVerifier(
            self.model_call_handler, num_tries_llm_judge_structure=2
        )

        belief_tracker_accum = FullBeliefTracker()

        result = True
        for f in function_list:
            belief_tracker_accum_prev = copy.deepcopy(belief_tracker_accum)
            story_length = len(belief_tracker_accum.story_script)
            a = f(belief_tracker_accum)
            assert a

            new_action_text = " ".join(belief_tracker_accum.story_script[story_length:])
            assert new_action_text

            (
                is_story_structure_valid,
                _,
                _,
            ) = quality_verifier.check_story_structure_with_shuffling_retries(
                belief_tracker_accum_prev,
                belief_tracker_accum,
                new_action_text,
                skip_second_order_qs=True,
            )
            result = result and is_story_structure_valid
        assert belief_tracker.story_script == belief_tracker_accum.story_script

        return result

    def main(self, max_sentences, max_nodes_allowed_to_explore):
        """
        A* SEARCH FOR DIFFICULT STORIES.

        f(). Cost factors of partial story already generated:
            - Model accuracy on partial story (Q calls)
            - Story naturality [as measured by a model, or by perplexity.]

        g(). Lookahead factors:
            - How easy is it to finish this partial story in a meaningful way? For this, we sample N possible endings.
                * How many are false-belief?
                * How long are these endings? Longer stories suggest that it was harder to end the story.
                * Accuracy. We could also include GPT-4o accuracy here (N*Q-calls).
            - ...

        Q = number of questions.

        NOTE: The final nodes are counted as two evaluations if f != constat.
        """
        self.logs = []

        start = (
            [],
            FullBeliefTracker(),
            collections.Counter(),
            [],
        )  # function_list, belief_tracker, action_counter, function_identifiers
        lookahead_stories, avg_story_len = self._compute_average_lookahead_story_length(
            start[0],
            start[1],
            start[2],
            is_already_final_story_wrt_actions_and_people=False,
            max_sentences=max_sentences,
        )  # MSCLARNOTES changed baseline avg_story_len = 1e6

        nodes = [start]
        queue = [(0, 0, avg_story_len, len(nodes) - 1)]  # (h, f) does not matter here
        all_tokens_used = 0
        num_nodes_evaluated = 0

        while (
            queue
            and len(self.logs) < self.UPPERBOUND_NUM_NODES_TO_BE_EXPLORED_IN_A_STORY
        ):

            # for experiment_baseline we want to immediately stop, because it should only do a single path execution
            if (
                self.experiment_to_run == "baseline"
                and num_nodes_evaluated >= max_nodes_allowed_to_explore
            ):
                break

            (_, cur_a_star_g, cur_avg_story_len, current_idx) = heapq.heappop(queue)
            (
                function_list,
                belief_tracker,
                action_type_counter,
                function_list_identifiers,
            ) = nodes[current_idx]
            max_neighbors_to_explore = next(
                iter(
                    val
                    for thr, val in self.max_neighbors_to_explore_conditions
                    if cur_a_star_g <= thr
                )
            )

            if self._check_if_node_is_final(
                belief_tracker, action_type_counter, cur_a_star_g
            ):
                responses, tokens_used = self.evaluator.evaluate_story(
                    belief_tracker,
                    nth=2 if self.a_star_g_version.startswith("acc2") else 1,
                    keep_only_interesting_qs=self.a_star_g_version.endswith("int"),
                )
                num_nodes_evaluated += len(responses) > 0

                a_star_g = (
                    np.mean([int(r["is_correct"]) for r in responses])
                    if responses
                    else 2
                )
                is_final_node_verified_quality = self._verify_final_story_structure(
                    belief_tracker, function_list
                )
                tmp_log = self.__build_log_entry(
                    " ".join(belief_tracker.story_script),
                    a_star_g,
                    0,
                    responses,
                    True,
                    is_final_node_verified_quality,
                    action_type_counter,
                    max_sentences,
                    cur_avg_story_len,
                    belief_tracker,
                    [self._is_false_belief_story(belief_tracker)],
                )
                tmp_log["time"] = time.time()
                self.logs.append(tmp_log)
                return (
                    function_list,
                    belief_tracker,
                    function_list_identifiers,
                    num_nodes_evaluated,
                )
            elif self._skip_node(
                belief_tracker,
                action_type_counter,
                num_nodes_evaluated,
                max_nodes_allowed_to_explore,
            ):
                continue

            if (
                "count_peeking_distracted_as_action" in self.experiment_variations
                and len(belief_tracker.story_script_tuple_representation)
                >= max_sentences
            ):
                break
            # if 'openai' in self.model_call_handler.model_access_method and self.model_call_handler.accum_token_cost() > 1e6 / 15 * 0.1:
            #    break

            valid_next_actions = self._list_valid_node_neighbors(
                belief_tracker,
                function_list,
                action_type_counter,
                max_sentences,
                max_neighbors_to_explore,
            )
            valid_next_actions = self._prioritize_node_neighbors_to_analyze(
                valid_next_actions, belief_tracker, action_type_counter
            )

            # 3. explore (& score) next valid actions
            # capping the number of neighbors to explore, since there are many nodes with accuracy = 2.0, so we might get stuck looping them forever
            num_explored_neighbors = 0
            for (
                belief_tracker1,
                function_list1,
                action_type_counter1,
                function_identifier1,
            ) in valid_next_actions[: max_neighbors_to_explore * 2]:
                if (
                    num_explored_neighbors >= max_neighbors_to_explore
                    or num_nodes_evaluated >= max_nodes_allowed_to_explore
                ):
                    break

                node = " ".join(belief_tracker1.story_script)
                if node in self.solved_story_cache:
                    tmp = self.solved_story_cache[node]
                    tmp["cached"] = True
                    a_star_g, a_star_h, avg_story_len = (
                        tmp["a_star_g"],
                        tmp["a_star_h"],
                        tmp["avg_story_len"],
                    )
                else:
                    # 0. Estimate lookahead value (cost to finish the story well)
                    is_already_final_story_wrt_actions_and_people = (
                        self._is_story_final_wrt_actions_and_people(
                            action_type_counter1,
                            len(belief_tracker1.people),
                            count_differentiating_actions=1,
                        )
                    )
                    (
                        lookahead_stories,
                        avg_story_len,
                    ) = self._compute_average_lookahead_story_length(
                        function_list1,
                        belief_tracker1,
                        action_type_counter1,
                        is_already_final_story_wrt_actions_and_people,
                        max_sentences,
                    )

                    if not self._story_passes_length_regularization(
                        belief_tracker1,
                        belief_tracker,
                        action_type_counter1,
                        avg_story_len,
                        cur_avg_story_len,
                        max_sentences,
                    ):
                        continue

                    a_star_h, is_false_belief_story_list = self._compute_a_star_h(
                        lookahead_stories, action_type_counter1
                    )
                    a_star_g, num_nodes_evaluated, responses = self._compute_a_star_g(
                        belief_tracker1, num_nodes_evaluated
                    )

                    if (
                        self.a_star_h_version == "intstory_w_people_keepeasy"
                        and a_star_g < 1.5
                    ):
                        a_star_g = (
                            1 - a_star_g
                        )  # experiment that prioritizes higher accuracy nodes

                    tmp = self.__build_log_entry(
                        node,
                        a_star_g,
                        a_star_h,
                        responses,
                        False,
                        False,
                        action_type_counter1,
                        max_sentences,
                        avg_story_len,
                        belief_tracker1,
                        is_false_belief_story_list,
                    )
                    tmp["cached"] = False
                    self.solved_story_cache[node] = tmp

                a_star_f = a_star_g + a_star_h * self.a_star_h_weight

                # simply pushing to heap since there are no cycles in this graph
                neighbor = (
                    function_list1,
                    belief_tracker1,
                    copy.copy(action_type_counter1),
                    copy.deepcopy(function_list_identifiers + [function_identifier1]),
                )
                nodes.append(neighbor)
                heapq.heappush(
                    queue, (a_star_f, a_star_g, avg_story_len, len(nodes) - 1)
                )
                self.logs.append(tmp)

                # if no model calls were done (default accuracy is 2) then this counts as a model exploration
                # if we are running the baseline, we always want to increase num_explored_neighbors since we want a single trajectory
                if a_star_g < 1.5 or self.experiment_to_run == "baseline":
                    num_explored_neighbors += 1

        return None, None, None, num_nodes_evaluated


class FullStoryStructureSearcher:
    def __init__(self, model_call_handler, dir_to_write, **kwargs):
        # Search hyper parameters (to pass to the actual searcher)
        self.group_n_next_steps = kwargs.get("group_n_next_steps", 3)
        self.a_star_g_version = kwargs.get("a_star_g_version", "acc")
        self.a_star_h_version = kwargs.get("a_star_h_version", "intstory_w_people")
        self.a_star_h_weight = kwargs.get("a_star_h_weight", 0.1)
        self.a_star_neighbor_priority = kwargs.get(
            "a_star_neighbor_priority", "weight-goal5"
        )
        self.max_neighbors_to_explore = kwargs.get(
            "max_neighbors_to_explore", "1_10-3_0.75-5_0.5"
        )

        self.experiment_variations = kwargs.get(
            "experiment_variations",
            [
                "count_peeking_distracted_as_action",
                "verify_final_story_for_state_updates",
            ],
        )

        self.experiment_to_run = kwargs.get("experiment_to_run", None)
        assert self.experiment_to_run in ["search", "baseline", "easiest"]
        if self.experiment_to_run == "baseline":
            self.a_star_g_version = (
                "constant-int" if self.a_star_g_version.endswith("-int") else "constant"
            )
            self.a_star_h_version = "constant"
            self.a_star_neighbor_priority = "random"
            self.max_neighbors_to_explore = 1
        elif self.experiment_to_run == "easiest":
            self.a_star_g_version = (
                "constant-int" if self.a_star_g_version.endswith("-int") else "constant"
            )
            self.a_star_h_version = "intstory_w_people_keepeasy"
            self.a_star_neighbor_priority = "random"

        # Context parameters
        self.model_generated_contexts_file = kwargs.get(
            "model_generated_contexts_file", None
        )
        self.num_stories_by_context = kwargs.get("num_stories_by_context", None)
        self.model_call_handler = model_call_handler
        self.dir_to_write = dir_to_write

    def _get_filename_without_file_extension(
        self,
        str_filename_params,
        story_type,
        num_people,
        num_moves,
        num_rooms,
        max_sentences,
        budget_per_story,
        num_stories_total,
        max_nodes_allowed_to_explore,
    ):
        if self.group_n_next_steps > 1:
            str_filename_params += f"_groupn_{self.group_n_next_steps}"
        str_filename_params = self.experiment_to_run + str_filename_params
        filename = f"{str_filename_params}_{story_type}_{self.model_call_handler.model_shortname}_n_{num_stories_total}_sent_{max_sentences}_p_{num_people}_m_{num_moves}_r_{num_rooms}_maxnodes_{max_nodes_allowed_to_explore}_maxneigh_{self.max_neighbors_to_explore}_f_{self.a_star_g_version}_g_{self.a_star_h_version}_gweight_{self.a_star_h_weight}_neighpri_{self.a_star_neighbor_priority}_foo"
        filename = os.path.join(self.dir_to_write, filename)
        assert len(filename.split("/")[-1]) <= 256, f"filename is too long: {filename}"
        return filename

    def _dump_all_info_to_pkl(self, all_logs, all_belief_trackers, filename):
        all_belief_trackers = [
            (bt, fn, fn_ids) for bt, fn, fn_ids in all_belief_trackers if bt is not None
        ]
        ctr = 0
        for i in range(len(all_logs)):
            if all_logs[i]["is_final_without_verification"] is True:
                all_logs[i]["belief_tracker"] = all_belief_trackers[ctr][0]
                all_logs[i]["function_list"] = all_belief_trackers[ctr][1]
                all_logs[i]["function_list_identifiers"] = all_belief_trackers[ctr][2]
                ctr += 1
        assert ctr == len(all_belief_trackers), (ctr, len(all_belief_trackers))
        with open(filename, "wb") as f:
            dill.dump(all_logs, f)

    def _load_and_clean_all_story_contexts(self):
        if self.model_generated_contexts_file:
            with open(self.model_generated_contexts_file, "r") as f:
                model_generated_context_dict_list = pd.read_json(
                    self.model_generated_contexts_file, lines=True
                ).to_dict("records")

                random.seed(time.time())
                indices = list(range(len(model_generated_context_dict_list)))
                random.shuffle(indices)
                model_generated_context_dict_list = [
                    model_generated_context_dict_list[i] for i in indices
                ]
                random.seed(0)

                for i in range(len(model_generated_context_dict_list)):
                    for k in ["rooms", "objects", "container_names", "topics"]:
                        model_generated_context_dict_list[i][k] = [
                            elem.lower()
                            for elem in model_generated_context_dict_list[i][k]
                        ]
                        for start_to_remove in ["a ", "an ", "the "]:
                            # assert not any(elem.startswith(start_to_remove) for elem in model_generated_context_dict_list[i][k])
                            model_generated_context_dict_list[i][k] = [
                                elem[len(start_to_remove) :]
                                if elem.startswith(start_to_remove)
                                else elem
                                for elem in model_generated_context_dict_list[i][k]
                            ]
                    for k in ["people_names"]:
                        model_generated_context_dict_list[i][k] = [
                            elem.title()
                            for elem in model_generated_context_dict_list[i][k]
                        ]
            model_generated_context_dict_list = [
                model_generated_context_dict_list[j]
                for j in range(len(model_generated_context_dict_list))
                for i in range(self.num_stories_by_context)
            ]
            str_filename_params = f"_modelctxt_{self.num_stories_by_context}-ctxt"
        else:
            model_generated_context_dict_list = None
            str_filename_params = ""
        return model_generated_context_dict_list, str_filename_params

    def main(
        self,
        story_type,
        num_people,
        num_moves,
        num_rooms,
        max_sentences,
        budget_per_story,
        num_stories_total,
        parallel_idx,
    ):

        max_nodes_allowed_to_explore = budget_per_story * num_stories_total
        (
            model_generated_context_dict_list,
            str_filename_params,
        ) = self._load_and_clean_all_story_contexts()
        assert (
            model_generated_context_dict_list
        ), "For our final experiments we only use this mode."
        filename = self._get_filename_without_file_extension(
            str_filename_params,
            story_type,
            num_people,
            num_moves,
            num_rooms,
            max_sentences,
            budget_per_story,
            num_stories_total,
            max_nodes_allowed_to_explore,
        )
        if os.path.exists(filename + ".jsonl"):
            print("skipping", filename)
            return False

        solved_story_cache = {}
        all_logs = []
        all_belief_trackers = []

        total_node_evaluations = 0
        early_stopping_if_no_progress_is_made, previous_total_node_evaluations = 0, -78
        story_idx = 0
        num_final_stories_found = 0
        while (
            total_node_evaluations < max_nodes_allowed_to_explore
            and early_stopping_if_no_progress_is_made < 10
        ):
            print(
                "story_idxx",
                parallel_idx,
                self.experiment_to_run,
                story_idx,
                total_node_evaluations,
                max_nodes_allowed_to_explore,
                early_stopping_if_no_progress_is_made,
            )
            use_custom_names_dict = (
                model_generated_context_dict_list[story_idx % num_stories_total]
                if model_generated_context_dict_list
                else None
            )

            searcher = ChallengingStoryStructureSearcher(
                self.model_call_handler,
                story_type,
                num_people,
                num_moves,
                num_rooms,
                use_custom_names_dict=use_custom_names_dict,
                experiment_to_run=self.experiment_to_run,
                group_n_next_steps=self.group_n_next_steps,
                a_star_g_version=self.a_star_g_version,
                a_star_h_version=self.a_star_h_version,
                a_star_h_weight=self.a_star_h_weight,
                a_star_neighbor_priority=self.a_star_neighbor_priority,
                max_neighbors_to_explore=self.max_neighbors_to_explore,
                experiment_variations=self.experiment_variations,
            )

            if self.experiment_to_run == "baseline":
                # WE WANT TO GENERATE RANDOM TRAJECTORIES
                # a good workaround to create a single random path is to run A* with max_neighbors_to_explore=1
                # f(x) = acc-int (it runs intermediate estimations but it does not matter)
                # lookeahead function should be non-existent (set to constant)
                num_stories_to_sample = (
                    max_nodes_allowed_to_explore // num_stories_total
                )
                self.a_star_h_version = "constant"

                generations = []
                for idx in range(num_stories_to_sample):
                    (
                        function_list,
                        belief_tracker,
                        function_list_identifiers,
                        num_node_evaluations_performed,
                    ) = searcher.main(
                        max_sentences=max_sentences, max_nodes_allowed_to_explore=1
                    )
                    assert (
                        num_node_evaluations_performed <= 1
                    ), "There should be only the final story evaluation in this setting (assuming a final story was found)"
                    logs = searcher.logs
                    total_node_evaluations += 1

                    if function_list is not None and logs[-1]["a_star_g"] < 1 - 1e-6:
                        assert logs[-1]["is_final_without_verification"]
                        generations.append(
                            (
                                logs[-1]["a_star_g"],
                                function_list,
                                belief_tracker,
                                logs,
                                solved_story_cache,
                                function_list_identifiers,
                            )
                        )
                        num_final_stories_found += 1
                    if (
                        function_list is not None
                        and 1.5 > logs[-1]["a_star_g"] > 1 - 1e-6
                    ):
                        assert logs[-1]["is_final_without_verification"]
                        generations.append(
                            (
                                logs[-1]["a_star_g"],
                                function_list,
                                belief_tracker,
                                logs,
                                solved_story_cache,
                                function_list_identifiers,
                            )
                        )
                        num_final_stories_found += 1

                if generations:
                    accuracies_with_index = zip(
                        [a for a, _, _, _, _, _ in generations], range(len(generations))
                    )
                    accuracies_with_index = sorted(accuracies_with_index)
                    _, best_index = accuracies_with_index[0]
                    (
                        _,
                        function_list,
                        belief_tracker,
                        logs,
                        solved_story_cache,
                        function_list_identifiers,
                    ) = generations[
                        best_index
                    ]  # take the one with lowest accuracy

                    for i, (_, _, _, a, _, _) in enumerate(generations):
                        if i == best_index:
                            continue
                        for elem in a:
                            elem["original_is_final"] = elem["is_final"]
                            elem["original_is_final_without_verification"] = elem[
                                "is_final_without_verification"
                            ]
                            elem["is_final"] = False
                            elem["is_final_without_verification"] = False
                            elem["baseline_path_discarded"] = True
                            logs.append(elem)

                    all_belief_trackers.append(
                        (belief_tracker, function_list, function_list_identifiers)
                    )
                else:
                    logs = []
                    best_index = -1
                    belief_tracker = None
                    function_list = None  # just in case

            elif (
                self.experiment_to_run == "search"
                or self.experiment_to_run == "easiest"
            ):
                (
                    function_list,
                    belief_tracker,
                    function_list_identifiers,
                    num_node_evaluations_performed,
                ) = searcher.main(
                    max_sentences=max_sentences,
                    max_nodes_allowed_to_explore=min(
                        max_nodes_allowed_to_explore // num_stories_total,
                        (max_nodes_allowed_to_explore - total_node_evaluations),
                    ),
                )
                logs = searcher.logs

                if len(logs) >= searcher.UPPERBOUND_NUM_NODES_TO_BE_EXPLORED_IN_A_STORY:
                    num_node_evaluations_performed = max_nodes_allowed_to_explore
                total_node_evaluations += num_node_evaluations_performed
                all_belief_trackers.append(
                    (belief_tracker, function_list, function_list_identifiers)
                )

                if logs[-1]["is_final_without_verification"]:
                    total_node_evaluations -= 1  # we evaluated this one double
                    num_final_stories_found += 1
                assert (
                    total_node_evaluations >= 0
                ), "Nodes evaluated cannot be negative."
                if total_node_evaluations <= previous_total_node_evaluations:
                    early_stopping_if_no_progress_is_made += 1
                else:
                    early_stopping_if_no_progress_is_made = 0
                previous_total_node_evaluations = total_node_evaluations
            else:
                assert False

            for l in logs:
                l["num_story"] = story_idx
                l["use_custom_names_dict"] = use_custom_names_dict
                l["total_node_evaluations"] = total_node_evaluations
                l["total_node_evaluations_version"] = "mult"
            all_logs.extend(logs)
            story_idx += 1

            if (
                num_final_stories_found >= num_stories_total
                and self.a_star_h_version == "intstory_w_people_keepeasy"
            ):
                break  # this is an easier case, it is just an experiment to find easy stories

        _dump_dict_list_to_jsonl(all_logs, filename + ".jsonl")
        self._dump_all_info_to_pkl(
            all_logs, all_belief_trackers, filename + "_dill.pkl"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        required=False,
    )
    parser.add_argument(
        "--model_access_method", type=str, default="vllm-api", required=False
    )

    parser.add_argument(
        "--model_generated_contexts_file",
        type=str,
        default="logs/model_generated_contexts_Llama-3.1-70B-Instruct_n_5_p_6_m_6_r_2_update_object_state_equiv_class_for_v1_dsl_wo_upsampling.jsonl",
        help="Output file from story_context_generatory.py",
    )
    parser.add_argument(
        "--experiment_to_run",
        type=str,
        default="search",
        choices=["search", "baseline", "easiest"],
        help="Experiment to run: search => real exp, baseline => a single path, easiest => search for cases with accuracy 1",
    )
    parser.add_argument("--budget_per_story", type=int, default=50)
    parser.add_argument("--num_stories_total", type=int, default=10)
    parser.add_argument("--num_stories_by_context", type=int, default=1)

    parser.add_argument("--group_n_next_steps", type=int, default=3)
    parser.add_argument("--a_star_g_version", type=str, default="acc")
    parser.add_argument("--a_star_h_version", type=str, default="intstory_w_people")
    parser.add_argument("--a_star_h_weight", type=float, default=0.1)
    parser.add_argument("--a_star_neighbor_priority", type=str, default="weight-goal4")
    parser.add_argument(
        "--max_neighbors_to_explore", type=str, default="1_10-3_0.75-5_0.5"
    )

    parser.add_argument("--generalization_experiment", action="store_true")

    NUM_PARALLEL = 8
    parser.add_argument("--i", type=int, default=None, help="DIY parallelization :)")
    args = parser.parse_args()

    model_call_handler = ModelCallHandler(args.model_name, args.model_access_method)

    all_story_types = [
        "tomi",
        "tomi+object-state",
        "tomi-object-state",
        "tomi+room-changes",
        "fantom-private",
        "fantom-public",
        "tomi+info-exchange",
        "allbutfantom",
        "all",
    ]
    all_story_types.extend([a + "+asymmetric" for a in all_story_types])

    VALID_STORY_GENERATION_TYPES = []
    if args.generalization_experiment:
        for story_type in all_story_types:
            for num_people in range(2, 7):
                for num_moves in range(num_people, 11):
                    for num_rooms in (
                        [2] if "room" in story_type or "all" in story_type else [1]
                    ):
                        for max_sentences in [15]:
                            VALID_STORY_GENERATION_TYPES.append(
                                (
                                    story_type,
                                    num_people,
                                    num_moves,
                                    num_rooms,
                                    max_sentences,
                                )
                            )
    else:
        for story_type in all_story_types:
            for num_people in range(2, 5):
                for num_moves in range(2, 5):
                    for num_rooms in (
                        [2] if "room" in story_type or "all" in story_type else [1]
                    ):
                        for max_sentences in [15]:
                            VALID_STORY_GENERATION_TYPES.append(
                                (
                                    story_type,
                                    num_people,
                                    num_moves,
                                    num_rooms,
                                    max_sentences,
                                )
                            )
    random.seed(0)
    random.shuffle(VALID_STORY_GENERATION_TYPES)
    random.seed(0)

    full_searcher = FullStoryStructureSearcher(
        model_call_handler,
        group_n_next_steps=args.group_n_next_steps,
        a_star_g_version=args.a_star_g_version,
        a_star_h_version=args.a_star_h_version,
        a_star_h_weight=args.a_star_h_weight,
        a_star_neighbor_priority=args.a_star_neighbor_priority,
        max_neighbors_to_explore=args.max_neighbors_to_explore,
        model_generated_contexts_file=args.model_generated_contexts_file,
        num_stories_by_context=args.num_stories_by_context,
        experiment_to_run=args.experiment_to_run,
        experiment_variations=[
            "count_peeking_distracted_as_action"
        ],  # , 'verify_final_story_for_state_updates'],
        dir_to_write="logs_debug_dec4_fixed_restrictions",
    )

    for i, (story_type, num_people, num_moves, num_rooms, max_sentences) in enumerate(
        VALID_STORY_GENERATION_TYPES
    ):
        if (
            args.i is not None and i % NUM_PARALLEL != args.i % NUM_PARALLEL
        ):  # DIY parallelization :)
            continue
        full_searcher.main(
            story_type,
            num_people,
            num_moves,
            num_rooms,
            max_sentences,
            args.budget_per_story,
            args.num_stories_total,
            i,
        )
