# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import ast
import dill
import copy
import time
import random
import argparse
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import _get_model_shortname, ModelAnswerEvaluator
from collections import Counter
from utils import ModelCallHandler
from belief_tracker import FullBeliefTracker, QuestionGenerator
from story_structure_searcher import (
    ChallengingStoryStructureSearcher,
    _setup_config_for_action_sampling,
)

ALL_IDENTIFIER_PARAMS_OF_SETTING = [
    "param=model_generated_context_dict_list_str",
    "param=story_type",
    "param=engine_shortname",
    "param=num_stories_total",
    "param=max_sentences",
    "param=num_people",
    "param=num_moves",
    "param=num_rooms",
    "param=max_nodes_allowed_to_explore",
    "param=max_neighbors_to_explore",
    "param=a_star_g_version",
    "param=a_star_h_version",
    "param=a_star_h_weight",
    "param=a_star_neighbor_priority",
]
ALL_IDENTIFIER_PARAMS_OF_SETTING_WITHOUT_PMR = [  # PMR = people, movement, rooms
    "param=model_generated_context_dict_list_str",
    "param=story_type",
    "param=engine_shortname",
    "param=num_stories_total",
    "param=num_rooms",  # keeping since there is only one option per setting anyways
    "param=max_nodes_allowed_to_explore",
    "param=max_neighbors_to_explore",
    "param=a_star_g_version",
    "param=a_star_h_version",
    "param=a_star_h_weight",
    "param=a_star_neighbor_priority",
]

MODEL_LIST = [
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "gpt-4o",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

# this is used to force an order between action sets when printing a results table
DESIRED_ORDER = [
    "tomi",
    "tomi-object-state",
    "tomi+object-state",
    "tomi+room-changes",
    "tomi+info-exchange",
    "tomi+room-changes+info-exchange",
    "allbutfantom",
    "tomi+fantom+room-changes+info-exchange",
    "all",
    "fantom-private",
    "fantom-public",
]

### **** Beginning Utils ****
def _parse_filename(filename):
    pattern = r"(.*)_(.*)_(.*)_n_(\d+)_sent_(\d+)_p_(\d+)_m_(\d+)_r_(\d+)_maxnodes_([\d.]+)_maxneigh_([\d._-]*)_f_(.*)_g_(.*)_gweight_([\d.]*)_(neighpri_(.*)_|)"
    match = re.match(pattern, filename)
    if match:
        return {
            "param=filename": filename,
            "param=model_generated_context_dict_list_str": match.group(1),
            "param=story_type": match.group(2),
            "param=engine_shortname": match.group(3),
            "param=num_stories_total": int(match.group(4)),
            "param=max_sentences": int(match.group(5)),
            "param=num_people": int(match.group(6)),
            "param=num_moves": int(match.group(7)),
            "param=num_rooms": int(match.group(8)),
            "param=max_nodes_allowed_to_explore": int(float(match.group(9))),
            "param=max_neighbors_to_explore": match.group(10),
            "param=a_star_g_version": match.group(11),
            "param=a_star_h_version": match.group(12),
            "param=a_star_h_weight": match.group(13),
            "param=a_star_neighbor_priority": match.group(15)
            if match.group(15)
            else "random",
        }
    else:
        return None


def _get_full_name_story_type(x):
    # we were using the summarized version of the setting because if not the filename would get too long
    if x == "all":
        return "tomi+fantom+room-changes+info-exchange"
    if x == "all+asymmetric":
        return "tomi+fantom+room-changes+info-exchange+asymmetric"
    if x == "allbutfantom":
        return "tomi+room-changes+info-exchange"
    if x == "allbutfantom+asymmetric":
        return "tomi+room-changes+info-exchange+asymmetric"
    return x


def dump_all_story_structures(
    logs_directory,
    output_filename,
    regex_match_files=r"^search_modelctxt_([\w._-]+).pkl$",
    skip_asymmetric=False,
    only_verified_cases=False,
):

    raw_dataset_list = []
    df_list = []
    for filename in os.listdir(logs_directory):
        if "asymmetric" in filename and skip_asymmetric:
            continue
        if not re.match(regex_match_files, filename):
            continue
        try:
            with open(os.path.join(logs_directory, filename), "rb") as f:
                all_logs = dill.load(f)
        except Exception as e:
            print("issue with filename", filename)
            print(e)
            continue

        if only_verified_cases:
            all_logs = [log for log in all_logs if log["is_final"] is True]
        else:
            all_logs = [
                log for log in all_logs if log["is_final_without_verification"] is True
            ]

        tmp = _parse_filename(filename)
        tmp["num_stories"] = len(all_logs)
        tmp["accuracy"] = np.mean([elem["a_star_g"] for elem in all_logs])

        is_false_belief_story_list = []
        interesting_q_list = []
        for elem in all_logs:
            bt = elem["belief_tracker"]
            function_list = elem["function_list"]
            question_generator = QuestionGenerator(bt)
            is_false_belief_story = question_generator.is_false_belief_story()
            is_false_belief_story_list.append(is_false_belief_story)
            for nth_order in [-1, 1, 2]:
                if nth_order <= 0:
                    question_list = question_generator.generate_factual_questions(
                        function_list
                    )
                else:
                    question_list = question_generator.main(
                        nth_order, expand_relation_type_info=True
                    )
                for question, expected_answer, entities, metadata in question_list:
                    x = _parse_filename(filename)

                    x.update(
                        {
                            "story": " ".join(bt.story_script),
                            "question": question,
                            "expected_answer": expected_answer,
                            "nth_order": nth_order,
                            "is_false_belief": is_false_belief_story,
                            "is_q_interesting": metadata[0],
                            "relation_type": entities[2],
                            "a_star_g": elem["a_star_g"],
                            "model_response": next(
                                iter(
                                    x["model_response"]
                                    for x in elem["model_responses_f"]
                                    if x["question"] == question
                                ),
                                None,
                            ),
                        }
                    )
                    interesting_q_list.append(metadata[0])
                    raw_dataset_list.append(x)

        tmp["false_belief_ratio"] = np.mean(is_false_belief_story_list)
        tmp["is_q_interesting"] = np.mean(interesting_q_list)
        df_list.append(tmp)

    df = pd.DataFrame(df_list)
    if len(df) == 0:
        return None, None
    print("STATS RAW DATASET")
    print(
        df.groupby(["param=story_type"])
        .agg(
            Count=("num_stories", "size"),
            Sum=("num_stories", "sum"),
            Avg=("accuracy", "mean"),
            AvgInteresting=("is_q_interesting", "mean"),
        )
        .reset_index()
    )
    print(
        df.groupby(["param=a_star_h_version"])
        .agg(
            Count=("num_stories", "size"),
            Sum=("num_stories", "sum"),
            Avg=("false_belief_ratio", "mean"),
        )
        .reset_index()
    )
    raw_dataset_df = pd.DataFrame(raw_dataset_list)
    raw_dataset_df.to_csv(output_filename)
    df.to_csv("stats_" + output_filename)
    return df, raw_dataset_df


def dump_all_infilled_stories(
    required_strings, banned_strings, base_dir, output_filename
):
    import numpy as np
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    dump_list = []
    idx = 0
    num_story_rejections = 0

    num_infilled_steps_rejections, num_infilled_steps_acceptances = 0, 0
    for filename in os.listdir(base_dir):
        if not filename.endswith(".pkl"):
            continue
        print()
        if any(
            e in filename
            for e in [f"_p_{v}_" for v in range(5, 13)]
            + [f"_m_{v}_" for v in range(5, 13)]
        ):
            continue
        if any(r not in filename for r in required_strings):
            continue
        if any(r in filename for r in banned_strings):
            continue

        with open(os.path.join(base_dir, filename), "rb") as f:
            big_pickle_data_list = dill.load(f)
        if not big_pickle_data_list:
            continue

        filename_params = _parse_filename(filename)
        filename_params["param=gitcommit"] = sha
        for test_idx, (
            df_list,
            (belief_tracker, function_list),
            _,
            is_story_accepted,
            responses_raw_script_evaluations,
            responses_infilled_script_evaluations,
            log_infilling,
        ) in enumerate(big_pickle_data_list):
            num_infilled_steps_rejections += sum(
                [elem["next_story_step_accepted"] is False for elem in df_list]
            )
            num_infilled_steps_acceptances += sum(
                [elem["next_story_step_accepted"] is True for elem in df_list]
            )
            full_story_accepted = df_list[-1]["next_story_step_accepted"] is True
            assert full_story_accepted == is_story_accepted
            if not full_story_accepted:
                num_story_rejections += 1
                continue

            question_generator = QuestionGenerator(belief_tracker)

            # print(log_infilling[0])
            # print(log_infilling[1])
            # print(log_infilling[2])

            accuracy_raw = (
                np.mean(
                    [int(r["is_correct"]) for r in responses_raw_script_evaluations]
                )
                if responses_raw_script_evaluations
                else None
            )
            accuracy_infilled = (
                np.mean(
                    [
                        int(r["is_correct"])
                        for r in responses_infilled_script_evaluations
                    ]
                )
                if responses_infilled_script_evaluations
                else None
            )
            is_false_belief_story_first = (
                question_generator.is_false_belief_story_restricted_to_first_order_qs()
            )
            is_false_belief_story = question_generator.is_false_belief_story()
            story_dict = {
                "raw_story": " ".join(df_list[-1]["raw_story"]),
                "infilled_story": df_list[-1]["accumulated_infilled_story"],
            }
            story_params = {
                "sprop=is_false_belief_story_1st": is_false_belief_story_first,
                "sprop=is_false_belief_story_1st_and_2nd": is_false_belief_story,
                "sprop=story_accuracy_1st_raw": accuracy_raw,
                "sprop=story_accuracy_1st_infilled": accuracy_infilled,
                "sprop=idx_in_filename": test_idx,
                "sprop=global_idx": idx,
                "sprop=fullpath": os.path.join(base_dir, filename),
                # 'sprop=infilling_params': [(elem['prompt_story_length'], elem['prompt_infilling_text_type']) for elem in df_list if elem['next_story_step_accepted'] is True],
                "sprop=time_to_generate": (df_list[-1]["time"] - df_list[0]["time"]),
            }
            idx += 1
            for nth_order in [-1, 1, 2]:
                if nth_order <= 0:
                    question_list = question_generator.generate_factual_questions(
                        function_list
                    )
                else:
                    question_list = question_generator.main(
                        nth_order, expand_relation_type_info=True
                    )
                for question, answer, question_params, extra_info in question_list:
                    tmp = copy.copy(story_dict)
                    tmp["question"] = question
                    tmp["expected_answer"] = answer
                    tmp["qprop=params"] = question_params
                    tmp["qprop=nth_order"] = nth_order
                    tmp["qprop=non_unique_mental_state"] = extra_info[0]
                    tmp["is_false_belief"] = is_false_belief_story
                    tmp["is_q_interesting"] = extra_info[0]
                    tmp.update(copy.copy(story_params))
                    tmp.update(copy.copy(filename_params))
                    dump_list.append(tmp)

    df = pd.DataFrame(dump_list)
    df.to_csv(output_filename)
    if len(df) == 0:
        return df
    print(
        "FB Debug",
        df[df["param=story_type"] == "tomi"].shape,
        df[df["raw_story"].str.contains("left")].head()["raw_story"].shape,
    )
    print(df.groupby(["param=story_type"])["sprop=is_false_belief_story_1st"].mean())
    print(
        "Number of stories (accepted, rejected):",
        idx,
        num_story_rejections,
        idx / (idx + num_story_rejections),
    )
    print(
        "Number of sentences infilled (accepted, rejected):",
        num_infilled_steps_acceptances,
        num_infilled_steps_rejections,
        num_infilled_steps_acceptances
        / (num_infilled_steps_acceptances + num_infilled_steps_rejections),
    )
    return df


def compare_infilled_stories_vs_structures_accuracy(df):

    # Comparison table between infilling vs not infilling
    raw_acc = (
        df.groupby(["infilled_story", "param=story_type"])
        .agg(accRaw=("sprop=story_accuracy_1st_raw", "mean"))
        .groupby("param=story_type")
        .agg(accRaw=("accRaw", "mean"))
    )
    infilled_acc = (
        df.groupby(["infilled_story", "param=story_type"])
        .agg(accInfilled=("sprop=story_accuracy_1st_infilled", "mean"))
        .groupby("param=story_type")
        .agg(accInfilled=("accInfilled", "mean"))
    )
    comparison = pd.merge(
        raw_acc, infilled_acc, left_index=True, right_index=True
    ).reset_index()
    comparison["is_asymmetric"] = np.where(
        comparison["param=story_type"].str.endswith("asymmetric"), True, False
    )
    comparison["param=story_type"] = comparison["param=story_type"].str.replace(
        "+asymmetric", ""
    )
    comparison["diff"] = comparison.apply(
        lambda x: x["accInfilled"] - x["accRaw"], axis=1
    )
    comparison["accRaw"] = comparison["accRaw"].apply(lambda x: "{:.2f}".format(x))
    comparison["accInfilled"] = comparison["accInfilled"].apply(
        lambda x: "{:.2f}".format(x)
    )
    comparison["diff"] = comparison["diff"].apply(lambda x: "{:.2f}".format(x))

    # Printing Table 5
    comparison["param=story_type"] = pd.Categorical(
        comparison["param=story_type"], DESIRED_ORDER
    )
    comparison = comparison.sort_values(by=["param=story_type"])
    comparison = comparison[
        ["param=story_type", "is_asymmetric", "accRaw", "accInfilled", "diff"]
    ]
    print(comparison.to_latex())

    print(len(df["infilled_story"].drop_duplicates()))
    print(
        "Accuracy (raw, infilled):",
        df["sprop=story_accuracy_1st_raw"].mean(),
        df["sprop=story_accuracy_1st_infilled"].mean(),
    )
    print("Avg time to infill:", df["sprop=time_to_generate"].mean())
    return df


def _precomputations(df, num_stories):
    # this filtering is done to avoid using data that is meant to be seen only in the generalization experiment
    df = df[
        (df["param=num_people"] < 5)
        & (df["param=num_moves"] < 5)
        & (df["param=max_sentences"] == 15)
        & (df["param=num_rooms"] < 3)
    ]
    x = df[~df["model_response"].isna()]
    x = x[x["param=num_stories_total"] == num_stories]
    x["is_correct_answer"] = x.apply(
        lambda x: x["expected_answer"] in x["model_response"].lower(), axis=1
    )
    x["num_sentences"] = x["story"].apply(lambda x: x.count(". ") + 1)
    return x


def _take_best_stories_and_aggregate(x, num_stories):
    """
    Sometimes our algorithm generates more than num_stories with the allocated budget, so we take the best ones.
    """
    acc_by_story = x.groupby(ALL_IDENTIFIER_PARAMS_OF_SETTING + ["story"]).agg(
        acc=("is_correct_answer", "mean"),
        numqs=("is_correct_answer", "count"),
        avgSentences=("num_sentences", "mean"),
    )
    acc_by_story = acc_by_story.sort_values("acc")
    acc_by_story = acc_by_story.reset_index()
    df_list = []
    for i, rows in acc_by_story.groupby(ALL_IDENTIFIER_PARAMS_OF_SETTING):
        df_list.append(rows[:num_stories])
    acc_by_story_top50 = pd.concat(df_list).reset_index()
    stats = acc_by_story_top50.groupby(ALL_IDENTIFIER_PARAMS_OF_SETTING).agg(
        acc=("acc", "mean"),
        numqs=("numqs", "sum"),
        numStories=("numqs", "count"),
        avgSentences=("avgSentences", "mean"),
    )
    return acc_by_story_top50, stats


def _take_best_stories_and_aggregate_by_interestingness(x, num_stories=50):
    acc_by_story = x.groupby(
        ALL_IDENTIFIER_PARAMS_OF_SETTING + ["story", "is_q_interesting"]
    ).agg(
        acc=("is_correct_answer", "mean"),
        numqs=("is_correct_answer", "count"),
        avgSentences=("num_sentences", "mean"),
    )
    acc_by_story = acc_by_story.sort_values("acc")
    acc_by_story = acc_by_story.reset_index()
    df_list = []
    for i, rows in acc_by_story.groupby(
        ALL_IDENTIFIER_PARAMS_OF_SETTING + ["is_q_interesting"]
    ):
        df_list.append(rows[:num_stories])
    acc_by_story_top50 = pd.concat(df_list).reset_index()
    stats = acc_by_story_top50.groupby(
        ALL_IDENTIFIER_PARAMS_OF_SETTING + ["is_q_interesting"]
    ).agg(
        acc=("acc", "mean"),
        numqs=("numqs", "sum"),
        numStories=("numqs", "count"),
        avgSentences=("avgSentences", "mean"),
    )
    return acc_by_story_top50, stats


### **** Actual Functions ****

# Section 3. A. TRACKTHEMIND finds challenging story structures for frontier models. [[[Figure 3]]]
def plot_accuracy_all_stories_by_moves(df_dict, NUM_STORIES_CONST):
    """
    This assumes that we have generated data for all models to be analyzed in the paper.
    """
    TOP_N_STORIES = NUM_STORIES_CONST

    df_list = []
    for model_name in MODEL_LIST:
        df = df_dict.get(model_name, {}).get("search", {}).get(False, None)
        if df is None:
            continue
        x = _precomputations(df, num_stories=NUM_STORIES_CONST)
        df, stats = _take_best_stories_and_aggregate(x, num_stories=TOP_N_STORIES)
        df_list.append(df)

    print(df_list)
    df = pd.concat(df_list)
    df["param=engine_shortname"] = df["param=engine_shortname"].apply(
        lambda x: x.replace("-v0.1", "").replace("Instruct", "Inst")
    )

    import seaborn as sns

    plt.rc("font", size=18)
    plt.subplots(figsize=(8, 4))
    plt.ylim([0.1, 0.5])
    plt.xticks(range(2, 5))
    sns.lineplot(
        data=df,
        x="param=num_people",
        y="acc",
        hue="param=engine_shortname",
        errorbar="ci",
    )
    plt.xlabel("# People")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(
        f"plots/sns_ci_lineplots_by_number_of_people_n_{NUM_STORIES_CONST}_allfiles_take_topn_stories_{TOP_N_STORIES}.pdf"
    )

    plt.subplots(figsize=(8, 4))
    plt.ylim([0.1, 0.5])
    plt.xticks(range(2, 5))
    sns.lineplot(
        data=df,
        x="param=num_moves",
        y="acc",
        hue="param=engine_shortname",
        errorbar="ci",
    )
    plt.xlabel("# Actions (from $\mathcal{A}'$)")
    plt.ylabel("Accuracy")
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(
        f"plots/sns_ci_lineplots_by_number_of_moves_n_{NUM_STORIES_CONST}_allfiles_take_topn_stories_{TOP_N_STORIES}.pdf"
    )


# Section 3. A. TRACKTHEMIND finds challenging story structures for frontier models. Table 1.
def generate_table_1(df, NUM_STORIES_CONST, BUDGET):
    acc_name = "acc"
    max_nodes = BUDGET * NUM_STORIES_CONST
    sent = 15

    x = _precomputations(df, num_stories=NUM_STORIES_CONST)
    acc_by_story_top10, stats = _take_best_stories_and_aggregate(
        x, num_stories=NUM_STORIES_CONST
    )
    df = acc_by_story_top10
    df = df[
        (df["param=a_star_g_version"] == acc_name)
        & (df["param=max_nodes_allowed_to_explore"] == max_nodes)
        & (df["param=max_sentences"] == sent)
    ]
    if len(df) == 0:
        return
    df.to_csv("analyses/selected_story_structures.csv")

    df["is_asymmetric"] = np.where(
        df["param=story_type"].str.endswith("asymmetric"), True, False
    )
    df["param=story_type"] = df["param=story_type"].str.replace("+asymmetric", "")

    # big table accumulating all (p, m, r)  [no need for a_star_g, use acc directly]
    stats_by_story_type = (
        df.groupby(["param=story_type", "is_asymmetric"])
        .agg(Accuracy=("acc", "mean"), Count=("acc", "count"))
        .reset_index()
    )
    stats_by_story_type["Accuracy"] = stats_by_story_type["Accuracy"].apply(
        lambda x: "{:.2f}".format(x)
    )
    stats_by_story_type["Count"] = stats_by_story_type["Count"].apply(
        lambda x: "{:.0f}".format(x)
    )
    stats_by_story_type["param=story_type"] = pd.Categorical(
        stats_by_story_type["param=story_type"], DESIRED_ORDER
    )
    stats_by_story_type = stats_by_story_type.sort_values(by=["param=story_type"])

    # TABLE 1 IN LATEX FORMAT
    df_pivoted = stats_by_story_type.pivot(
        index="param=story_type", columns="is_asymmetric", values="Accuracy"
    )
    print(df_pivoted.to_latex())

    # Question count in each table 1 cell
    df_pivoted = stats_by_story_type.pivot(
        index="param=story_type", columns="is_asymmetric", values="Count"
    )
    print(df_pivoted.to_latex())


def generate_table_6(df, NUM_STORIES_CONST, BUDGET):
    #### How many interesting questions and what is their accuracy if we use acc and not acc-int?
    acc_name = "acc"
    max_nodes = BUDGET * NUM_STORIES_CONST
    sent = 15

    x = _precomputations(df, num_stories=NUM_STORIES_CONST)
    acc_by_story_top10, stats = _take_best_stories_and_aggregate_by_interestingness(
        x, num_stories=NUM_STORIES_CONST
    )
    df = acc_by_story_top10
    df = df[
        (df["param=a_star_g_version"] == acc_name)
        & (df["param=max_nodes_allowed_to_explore"] == max_nodes)
        & (df["param=max_sentences"] == sent)
    ]
    df["is_asymmetric"] = np.where(
        df["param=story_type"].str.endswith("asymmetric"), True, False
    )
    df["param=story_type"] = df["param=story_type"].str.replace("+asymmetric", "")
    stats_by_story_type = (
        df.groupby(["param=story_type", "is_asymmetric", "is_q_interesting"])
        .agg(Accuracy=("acc", "mean"), Count=("acc", "count"))
        .reset_index()
    )
    stats_by_story_type["Accuracy"] = stats_by_story_type["Accuracy"].apply(
        lambda x: "{:.2f}".format(x)
    )
    stats_by_story_type["Count"] = stats_by_story_type["Count"].apply(
        lambda x: "{:.0f}".format(x)
    )
    stats_by_story_type["param=story_type"] = pd.Categorical(
        stats_by_story_type["param=story_type"], DESIRED_ORDER
    )
    stats_by_story_type = stats_by_story_type.sort_values(
        by=["param=story_type", "is_asymmetric", "is_q_interesting"]
    )

    df_pivoted = stats_by_story_type.pivot(
        index=["param=story_type", "is_asymmetric"],
        columns=["is_q_interesting"],
        values="Accuracy",
    )
    df_pivoted = df_pivoted.reset_index()
    df_pivoted = df_pivoted.rename(columns={True: "AccInt", False: "AccNotInt"})
    df_pivoted = df_pivoted.reset_index()

    df_pivoted_ratio_int = stats_by_story_type.pivot(
        index=["param=story_type", "is_asymmetric"],
        columns=["is_q_interesting"],
        values="Count",
    )
    df_pivoted_ratio_int = df_pivoted_ratio_int.reset_index()
    df_pivoted_ratio_int["ratio_interesting_q"] = df_pivoted_ratio_int.apply(
        lambda x: str(
            int(np.round(int(x[True]) * 100 / (int(x[False]) + int(x[True]))))
        )
        + "\%",
        axis=1,
    )
    df_pivoted_ratio_int.drop(True, axis=1, inplace=True)
    df_pivoted_ratio_int.drop(False, axis=1, inplace=True)
    df_pivoted_ratio_int = df_pivoted_ratio_int.reset_index()

    df = pd.merge(
        df_pivoted, df_pivoted_ratio_int, on=["param=story_type", "is_asymmetric"]
    )
    df = df[
        [
            "param=story_type",
            "is_asymmetric",
            "AccInt",
            "AccNotInt",
            "ratio_interesting_q",
        ]
    ]
    print(df.to_latex())


def evaluate_model_on_data_generated_by_other_models(
    df_dict, model_call_handler, SAMPLE_SIZE
):
    evaluator = ModelAnswerEvaluator(model_call_handler)

    # Evaluate all files with a model
    for model_list_idx, nth_order in itertools.product(range(len(MODEL_LIST)), [1, 2]):
        # model_specific_evaluations = '1_2', where 1 is the index in MODEL_LIST and 2 is the nth order to use
        # model_list_idx, nth_order = args.model_specific_evaluations.split('_')
        # model_list_idx = int(model_list_idx)
        # nth_order = int(nth_order)
        # assert nth_order in [1, 2]
        # SAMPLE_SIZE = 250
        # MODEL_LIST = [MODEL_LIST[model_list_idx]]
        model_engine_dataset = MODEL_LIST[model_list_idx]
        engine_shortname_dataset_to_eval = _get_model_shortname(model_engine_dataset)
        # output_filename_astar = f'analyses/raw_scripts_questions_{engine_shortname_dataset_to_eval}_sept26_adv_n_{NUM_STORIES_CONST}.csv'
        # df_baseline = pd.read_csv(output_filename_astar, index_col=0)
        # df_baseline = df_dict[chatgpt_engine_dataset]["search"]
        df_baseline = (
            df_dict.get(model_engine_dataset, {}).get("search", {}).get(False, None)
        )
        if df_baseline is None:
            continue

        df_baseline_nth = df_baseline[df_baseline["nth_order"] == nth_order]
        if SAMPLE_SIZE > 0:
            output_filename_evaluated = f"analyses/raw_scripts_questions_{engine_shortname_dataset_to_eval}_{search_or_baseline}_n_{NUM_STORIES_CONST}_budget_{BUDGET}_evaluated_with_{model_call_handler.model_shortname}_nth_{nth_order}_sample_{SAMPLE_SIZE}.csv"
            df_baseline_nth = df_baseline_nth.sample(frac=1, random_state=42)[
                :SAMPLE_SIZE
            ]
            df = evaluator.evaluate_file(df_baseline_nth, output_filename_evaluated)
        else:
            output_filename_evaluated = f"analyses/raw_scripts_questions_{engine_shortname_dataset_to_eval}_{search_or_baseline}_n_{NUM_STORIES_CONST}_budget_{BUDGET}_evaluated_with_{model_call_handler.model_shortname}_nth_{nth_order}.csv"
            df = evaluator.evaluate_file(df_baseline_nth, output_filename_evaluated)


# Section 3. C. Story structures found adversarially for a model remain challenging for other models. Table 2.
def summarize_all_cross_models_evaluations(NUM_STORIES_CONST, BUDGET, SAMPLE_SIZE):
    print("ModelName1(DatasetGen)", "ModelName2(Evaluator)", "Accuracy", "WrongRatio")
    for model_name_1, model_name_2 in itertools.product(MODEL_LIST, MODEL_LIST):
        model_name_1 = _get_model_shortname(model_name_1)
        model_name_2 = _get_model_shortname(model_name_2)
        output_filename_evaluated_1 = f"analyses/raw_scripts_questions_{model_name_1}_{search_or_baseline}_n_{NUM_STORIES_CONST}_budget_{BUDGET}_evaluated_with_{model_name_2}_nth_{1}_sample_{SAMPLE_SIZE}.csv"
        output_filename_evaluated_2 = f"analyses/raw_scripts_questions_{model_name_1}_{search_or_baseline}_n_{NUM_STORIES_CONST}_budget_{BUDGET}_evaluated_with_{model_name_2}_nth_{2}_sample_{SAMPLE_SIZE}.csv"

        if not os.path.exists(output_filename_evaluated_1) or not os.path.exists(
            output_filename_evaluated_2
        ):
            continue

        df1 = pd.read_csv(output_filename_evaluated_1)
        df2 = pd.read_csv(output_filename_evaluated_2)
        df = pd.concat([df1, df2])
        no_response_ratio = df["model_response"].isna().mean()
        is_answer_correct_ratio = df["is_answer_correct"].mean()
        is_answer_wrong_ratio = len(
            df[~df["model_response"].isna() & ~df["is_answer_correct"]]
        ) / len(df)
        print(
            model_name_1, model_name_2, is_answer_correct_ratio, is_answer_wrong_ratio
        )


# Section 5, Part 2. Training data biases against theory of mind and its implications.
def randomly_sampling_story_structures(analysis_to_return):
    """
    ** Randomly Sampling ToMi-like Stories to check if they are false belief (i.e., contain any interesting q) or not **
    Used in Section 5, "Training data biases against theory of mind and its implications".
    """
    assert analysis_to_return in [
        "is_false_belief_story",
        "ratio_interesting_q",
        "ratio_false_belief_q",
    ]

    confusion_matrix = {
        10: [[None for _ in range(3)] for _ in range(3)],
        15: [[None for _ in range(3)] for _ in range(3)],
    }
    for max_sentences in [10, 15]:
        for num_people, num_moves, num_rooms in itertools.product(
            [2, 3, 4], [2, 3, 4], [1]
        ):
            (
                false_belief_counter,
                false_belief_counter_first,
                is_interesting_q,
                is_true_belief_q,
            ) = (Counter(), Counter(), Counter(), Counter())

            n_total = 1000
            for _ in range(n_total):
                belief_tracker = FullBeliefTracker()
                story_structure_searcher = ChallengingStoryStructureSearcher(
                    None,
                    "tomi",
                    num_people,
                    num_moves,
                    num_rooms,
                    raw_rooms=False,
                    use_custom_names_dict=None,
                )

                (
                    function_list,
                    belief_tracker,
                    action_type_counter,
                    _,
                ) = story_structure_searcher.complete_story_randomly_and_incrementally(
                    belief_tracker, [], Counter(), max_sentences, no_retries=True
                )

                question_generator = QuestionGenerator(belief_tracker)
                is_false_belief_story = question_generator.is_false_belief_story()
                false_belief_counter[is_false_belief_story] += 1
                any_interesting = False
                for n in [1, 2]:
                    for question, _, relation_type, metadata in question_generator.main(
                        nth_reasoning_order=n, expand_relation_type_info=True
                    ):
                        is_interesting_q[metadata[0]] += 1
                        any_interesting = any_interesting or metadata[0]
                        is_true_belief_q[relation_type[-1].split("-")[-1]] += 1
                assert (
                    any_interesting == is_false_belief_story
                ), f"These should be equivalent computations. {(any_interesting, is_false_belief_story)}"

            confusion_matrix[max_sentences][num_people - 2][num_moves - 2] = (
                false_belief_counter[True] / n_total
            )
            assert false_belief_counter[True] + false_belief_counter[False] == n_total
            assert all(k in ["True", "False"] for k in is_true_belief_q)

            if analysis_to_return == "is_false_belief_story":
                print(
                    max_sentences,
                    num_people,
                    num_moves,
                    "{:.3f}".format(
                        false_belief_counter[True]
                        / (false_belief_counter[True] + false_belief_counter[False])
                    ),
                )
            elif analysis_to_return == "ratio_interesting_q":
                print(
                    max_sentences,
                    num_people,
                    num_moves,
                    "{:.3f}".format(
                        is_interesting_q[True]
                        / (is_interesting_q[True] + is_interesting_q[False])
                    ),
                )
            elif analysis_to_return == "ratio_false_belief_q":
                print(
                    max_sentences,
                    num_people,
                    num_moves,
                    "{:.3f}".format(
                        is_true_belief_q["False"]
                        / (is_true_belief_q["True"] + is_true_belief_q["False"])
                    ),
                )
            else:
                assert False, "Option not implemented"

    print(f"Confusion Matrix for {analysis_to_return}")
    print("MaxSentences=10", confusion_matrix[10])
    print("MaxSentences=15", confusion_matrix[15])


def choosing_best_search_parameters(df, NUM_STORIES_CONST):
    #### Compute accuracy using top 50 answers from each setting
    x = _precomputations(df, num_stories=NUM_STORIES_CONST)
    acc_by_story_top50, stats = _take_best_stories_and_aggregate(
        x, num_stories=NUM_STORIES_CONST
    )
    stats["acc"].unstack("param=max_nodes_allowed_to_explore").to_csv(
        "analyses/choosing_budget.csv"
    )
    stats["acc"].unstack("param=a_star_h_weight").to_csv(
        "analyses/choosing_gweight.csv"
    )
    ####

    ### Split performance between interesting and not interesting questions
    x_from_top50_stories = pd.merge(
        acc_by_story_top50, x, on=ALL_IDENTIFIER_PARAMS_OF_SETTING + ["story"]
    )
    acc_by_story_top50_interesting = x_from_top50_stories.groupby(
        ALL_IDENTIFIER_PARAMS_OF_SETTING + ["story", "is_q_interesting"]
    ).agg(acc=("is_correct_answer", "mean"), numqs=("is_correct_answer", "count"))
    acc_by_story_top50_interesting = acc_by_story_top50_interesting.reset_index()
    stats = acc_by_story_top50_interesting.groupby(
        ALL_IDENTIFIER_PARAMS_OF_SETTING + ["is_q_interesting"]
    ).agg(acc=("acc", "mean"), numqs=("numqs", "sum"), numstories=("numqs", "count"))
    stats["acc"].unstack("is_q_interesting").to_csv(
        "analyses/choosing_is_q_interesting.csv"
    )
    ####


# Section 3. B. A* is a better strategy than over-generation and filtering. Figure 6 + numbers reported.
def compare_astar_vs_baseline(
    df, df_baseline, NUM_STORIES_CONST, BUDGET, setting_suffix
):
    x = _precomputations(df, num_stories=NUM_STORIES_CONST)
    settings = x[ALL_IDENTIFIER_PARAMS_OF_SETTING].drop_duplicates().reset_index()
    settings = settings[
        settings["param=max_nodes_allowed_to_explore"] == NUM_STORIES_CONST * BUDGET
    ]
    settings = (
        settings.groupby(ALL_IDENTIFIER_PARAMS_OF_SETTING_WITHOUT_PMR)
        .agg(numfiles=("param=num_people", "count"))
        .reset_index()
    )
    # print("ASTAR SETTINGS MISSING", settings)

    x_baseline = _precomputations(df_baseline, num_stories=NUM_STORIES_CONST)
    settings = (
        x_baseline[ALL_IDENTIFIER_PARAMS_OF_SETTING].drop_duplicates().reset_index()
    )
    settings = settings[
        settings["param=max_nodes_allowed_to_explore"] == NUM_STORIES_CONST * BUDGET
    ]
    settings = (
        settings.groupby(ALL_IDENTIFIER_PARAMS_OF_SETTING_WITHOUT_PMR)
        .agg(numfiles=("param=num_people", "count"))
        .reset_index()
    )
    # print("BASELINE SETTINGS MISSING", settings)

    _, stats = _take_best_stories_and_aggregate(x, num_stories=NUM_STORIES_CONST)
    _, stats_baseline = _take_best_stories_and_aggregate(
        x_baseline, num_stories=NUM_STORIES_CONST
    )

    stats = stats.reset_index()
    stats["param=story_type"] = stats["param=story_type"].apply(
        lambda x: _get_full_name_story_type(x)
    )
    stats_baseline = stats_baseline.reset_index()
    stats_baseline["param=a_star_g_version"] = stats_baseline[
        "param=a_star_g_version"
    ].apply(lambda x: "acc-int" if x == "constant-int" else "acc")
    stats_baseline["param=story_type"] = stats_baseline["param=story_type"].apply(
        lambda x: _get_full_name_story_type(x)
    )
    stats_baseline = stats_baseline[
        stats_baseline["param=max_nodes_allowed_to_explore"]
        == NUM_STORIES_CONST * BUDGET
    ]

    stats.to_csv(f"analyses/stats_astar_{setting_suffix}.csv")
    stats_baseline.to_csv(f"analyses/stats_baseline_{setting_suffix}.csv")

    ALL_IDENTIFIER_MERGER = [
        "param=story_type",
        "param=engine_shortname",
        "param=num_stories_total",
        "param=max_sentences",
        "param=num_people",
        "param=num_moves",
        "param=num_rooms",
        "param=max_nodes_allowed_to_explore",
        "param=a_star_g_version",
    ]
    stats_baseline.sort_values(by="acc", ascending=False, inplace=True)
    stats_baseline = stats_baseline.drop_duplicates(subset=ALL_IDENTIFIER_MERGER)
    stats.sort_values(by="acc", ascending=True, inplace=True)
    stats = stats.drop_duplicates(subset=ALL_IDENTIFIER_MERGER)
    assert all(
        [
            e == 1
            for e in stats.groupby(ALL_IDENTIFIER_MERGER)
            .count()
            .reset_index()["avgSentences"]
            .tolist()
        ]
    ), "No duplicate settings allowed!"
    assert all(
        [
            e == 1
            for e in stats_baseline.groupby(ALL_IDENTIFIER_MERGER)
            .count()
            .reset_index()["avgSentences"]
            .tolist()
        ]
    ), "No duplicate settings allowed!"

    comparison = pd.merge(
        stats,
        stats_baseline,
        on=ALL_IDENTIFIER_MERGER,
        how="inner",
        suffixes=("_astar", "_baseline"),
    )
    comparison["accDiff"] = comparison.apply(
        lambda x: x["acc_astar"] - x["acc_baseline"], axis=1
    )
    comparison.to_csv(f"analyses/comparison_astar_baseline_{setting_suffix}.csv")
    comparison = comparison.sample(n=162 // 2, random_state=21)
    print(
        "AvgSentencesAstar",
        setting_suffix,
        comparison["avgSentences_astar"].mean(),
        comparison["avgSentences_astar"].count(),
    )
    print(
        "AvgSentencesBaseline",
        setting_suffix,
        comparison["avgSentences_baseline"].mean(),
        comparison["avgSentences_baseline"].count(),
    )
    print(
        "RatioAllStoriesGeneratedAstar",
        len(comparison[comparison["numStories_astar"] == 10]) / len(comparison),
    )
    print(
        "RatioAllStoriesGeneratedBaseline",
        len(comparison[comparison["numStories_baseline"] == 10]) / len(comparison),
    )
    print(
        "Accuracies (Astar, Baseline, Diff)",
        comparison["acc_astar"].mean(),
        comparison["acc_baseline"].mean(),
        comparison["accDiff"].mean(),
    )

    differences = sorted(comparison["accDiff"].tolist())
    bins = 8
    plt.rc("font", size=18)
    plt.figure()
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.hist(differences, bins=bins)
    plt.tight_layout(pad=1.5)
    ax.set_xlabel("Acc. Diff. between A* - and Baseline-generated data")
    ax.set_ylabel("Count")
    plt.savefig(f"analyses/astar_vs_baseline_{setting_suffix}.pdf")


def generate_human_evaluation_file(
    df,
    output_filename,
    MAX_QUESTIONS_TO_DUMP_PER_STORY,
    MAX_STORIES_TO_DUMP,
    story_fieldname_input_df,
):
    MESSAGES_EXTRANEOUS_TO_STORY = ["continue story here", "please continue"]
    df = df.sample(frac=1, random_state=42)
    df = df[df["is_q_interesting"] & df["is_false_belief"]]
    df = df[(df["param=num_people"] < 5) & (df["param=num_moves"] < 5)]
    df = df[
        (df["param=max_sentences"] == 15)
        & (df["param=a_star_h_version"] == "intstory_w_people")
        & (df["param=a_star_h_weight"] == 0.1)
    ]

    df_list = []
    counter = Counter()
    for story, rows_list in df.groupby(story_fieldname_input_df):
        assert not any(
            e in story.lower() for e in MESSAGES_EXTRANEOUS_TO_STORY
        ), "The presence of these message indicates an issue in infilling."
        first_row = rows_list.iloc[0]
        if counter[first_row["param=story_type"]] >= MAX_STORIES_TO_DUMP:
            continue
        ctr = 0
        for i, row in rows_list.iterrows():
            tmp = copy.copy(dict(row))
            tmp["ctr_story_type"] = counter[first_row["param=story_type"]]
            tmp[story_fieldname_input_df] = story
            tmp["story"] = story.replace(". ", ".\n ") if ctr == 0 else "."
            tmp["question_number"] = ctr
            ctr += 1
            df_list.append(tmp)
        counter[first_row["param=story_type"]] += 1
    df = pd.DataFrame(df_list)
    df = df[df["question_number"] < MAX_QUESTIONS_TO_DUMP_PER_STORY]
    df = df.sort_values(["ctr_story_type", story_fieldname_input_df, "question_number"])
    df.to_csv(output_filename)


def dump_generalization_experiment():
    for num_people in ["small", 5, 6]:
        for num_moves in range(num_people, 11):
            model_name = MODEL_LIST[0]  # llama
            only_verified_cases = False
            model_shortname = _get_model_shortname(model_name)
            output_filename = f"analyses/raw_scripts_dec4_questions_{model_shortname}_{search_or_baseline}_verified_{only_verified_cases}_n_{NUM_STORIES_CONST}_budget_{BUDGET}.csv"
            if os.path.exists(output_filename):
                df = pd.read_csv(output_filename, index_col=0)
            else:
                if num_people == "small":
                    regex_match_files = (
                        r"^search_modelctxt_([\d]+)-ctxt_groupn_3([+\w._-]+)_"
                        + model_shortname
                        + r"([+\w._-]*)_n_"
                        + str(NUM_STORIES_CONST)
                        + r"_sent_15_p_(2|3|4)_m_("
                        + str(num_moves)
                        + r")_r_(1|2)_maxnodes_"
                        + str(BUDGET * NUM_STORIES_CONST)
                        + r"_([+\w._-]+)_f_acc_g_intstory_w_people_gweight_0.1_([+\w._-]+)weight-goal4([+\w._-]+).pkl$"
                    )
                else:
                    regex_match_files = (
                        r"^search_modelctxt_([\d]+)-ctxt_groupn_3([+\w._-]+)_"
                        + model_shortname
                        + r"([+\w._-]*)_n_"
                        + str(NUM_STORIES_CONST)
                        + r"_sent_15_p_("
                        + str(num_people)
                        + r")_m_("
                        + str(num_moves)
                        + r")_r_(1|2)_maxnodes_"
                        + str(BUDGET * NUM_STORIES_CONST)
                        + r"_([+\w._-]+)_f_acc_g_intstory_w_people_gweight_0.1_([+\w._-]+)weight-goal4([+\w._-]+).pkl$"
                    )

                df_stats, df = dump_all_story_structures(
                    logs_directory,
                    output_filename=output_filename,
                    regex_match_files=regex_match_files,
                    only_verified_cases=only_verified_cases,
                )
                if df_stats is not None and len(df_stats) > 0:
                    print(
                        "Num stories generated",
                        df_stats["num_stories"].tolist(),
                        df_stats["num_stories"].mean(),
                    )
    exit(0)


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
    parser.add_argument("--evaluate_cross_model_generations", action="store_true")
    parser.add_argument("--human_eval_doc_generator", action="store_true")
    parser.add_argument("--dump_for_huggingface", action="store_true")

    args = parser.parse_args()

    NUM_STORIES_CONST = 10
    BUDGET = 50
    logs_directory = "logs_debug_dec4_fixed_restrictions"

    os.makedirs("analyses", exist_ok=True)
    os.makedirs("stats_analyses", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # dump_generalization_experiment()

    # 0. Turn story structures & infilled stories data into compact mode
    df_dict = {}  # model_name -> search_or_baseline -> df
    for model_name, search_or_baseline, only_verified_cases in itertools.product(
        MODEL_LIST, ["search", "baseline"], [True, False]
    ):
        model_shortname = _get_model_shortname(model_name)
        output_filename = f"analyses/raw_scripts_dec4_questions_{model_shortname}_{search_or_baseline}_verified_{only_verified_cases}_n_{NUM_STORIES_CONST}_budget_{BUDGET}.csv"
        if os.path.exists(output_filename):
            df = pd.read_csv(output_filename, index_col=0)
        else:
            if search_or_baseline == "search":
                regex_match_files = (
                    r"^search_modelctxt_([\d]+)-ctxt_groupn_3([+\w._-]+)_"
                    + model_shortname
                    + r"([+\w._-]*)_n_"
                    + str(NUM_STORIES_CONST)
                    + r"_sent_15_p_(2|3|4)_m_(2|3|4)_r_(1|2)_maxnodes_"
                    + str(BUDGET * NUM_STORIES_CONST)
                    + r"_([+\w._-]+)_f_acc_g_intstory_w_people_gweight_0.1_([+\w._-]+)weight-goal4([+\w._-]+).pkl$"
                )
            else:
                regex_match_files = (
                    r"^baseline_modelctxt_([\d]+)-ctxt_groupn_3([+\w._-]+)_"
                    + model_shortname
                    + r"([+\w._-]*)_n_"
                    + str(NUM_STORIES_CONST)
                    + r"_sent_15_p_(2|3|4)_m_(2|3|4)_r_(1|2)_maxnodes_"
                    + str(BUDGET * NUM_STORIES_CONST)
                    + r"_maxneigh_1_f_constant_g_constant_([+\w._-]+)_neighpri_random_foo_dill.pkl$"
                )

            df_stats, df = dump_all_story_structures(
                logs_directory,
                output_filename=output_filename,
                regex_match_files=regex_match_files,
                only_verified_cases=only_verified_cases,
            )
            if df_stats is not None and len(df_stats) > 0:
                print(
                    "Num stories generated",
                    df_stats["num_stories"].tolist(),
                    df_stats["num_stories"].mean(),
                )
        if df is not None and len(df) > 0:
            if model_name not in df_dict:
                df_dict[model_name] = {}
            if search_or_baseline not in df_dict[model_name]:
                df_dict[model_name][search_or_baseline] = {}
            df_dict[model_name][search_or_baseline][only_verified_cases] = df
    print("Available Models", df_dict.keys())

    # 0. Create/load infilled stories file
    model_shortname = _get_model_shortname(args.model_name)
    infilled_filename = (
        f"analyses/infilled_dec4_all_{model_shortname}_n_{NUM_STORIES_CONST}.csv"
    )
    if os.path.exists(infilled_filename):
        df_infilled = pd.read_csv(infilled_filename, index_col=0)
    else:
        df_infilled = dump_all_infilled_stories(
            ["_v3", model_shortname, f"_n_{NUM_STORIES_CONST}_"]
            if NUM_STORIES_CONST > 0
            else [model_shortname],
            ["_fanlike", "_v2"],
            "infilled_with_llm_judge_w_goals_and_initial_debug_dec4_fixed_restrictions",
            infilled_filename,
        )
        infilled_filename2 = f"analyses/infilled_dec4_all_{model_shortname}_n_{NUM_STORIES_CONST}_fantom_like_data.csv"
        _ = dump_all_infilled_stories(
            ["_fanlike", "_v3", model_shortname, f"_n_{NUM_STORIES_CONST}_"]
            if NUM_STORIES_CONST > 0
            else [model_shortname],
            ["_v2"],
            "infilled_with_llm_judge_w_goals_and_initial_debug_dec4_fixed_restrictions",
            infilled_filename2,
        )

    # Optional: Create human evaluation doc
    if args.human_eval_doc_generator:
        os.makedirs("human_evals", exist_ok=True)
        MAX_QUESTIONS_TO_DUMP_PER_STORY = 10
        MAX_STORIES_TO_DUMP = 10

        df = df_dict["meta-llama/Meta-Llama-3.1-70B-Instruct"]["search"][True]
        df = df.rename(
            columns={"story": "raw_story"}
        )  # the 'story' field will be unusable inside generate_human_evaluation_file()
        generate_human_evaluation_file(
            df,
            "human_evals/human_eval_dec4_raw_1.csv",
            MAX_QUESTIONS_TO_DUMP_PER_STORY,
            MAX_STORIES_TO_DUMP,
            "raw_story",
        )
        generate_human_evaluation_file(
            df_infilled,
            "human_evals/human_eval_dec4_infilled_1.csv",
            MAX_QUESTIONS_TO_DUMP_PER_STORY,
            MAX_STORIES_TO_DUMP,
            "infilled_story",
        )
        exit(0)

    # Section 3. C. Story structures found adversarially for a model remain challenging for other models. Table 2.
    if args.evaluate_cross_model_generations:
        print("Generating Table 2... [run once per model]")
        model_call_handler = ModelCallHandler(args.model_name, args.model_access_method)
        evaluate_model_on_data_generated_by_other_models(
            df_dict, model_call_handler, SAMPLE_SIZE=500
        )
        summarize_all_cross_models_evaluations(
            NUM_STORIES_CONST, BUDGET, SAMPLE_SIZE=500
        )
        exit(0)

    # Section 3. A. TRACKTHEMIND finds challenging story structures for frontier models. Table 1.
    print("Generating Table 1...")
    for model_name in MODEL_LIST:
        print(model_name)
        generate_table_1(
            df_dict[model_name]["search"][False], NUM_STORIES_CONST, BUDGET
        )

    # Section 5, Part 1. LLMs lack robust state tracking skills. Table 6.
    for model_name in MODEL_LIST:
        print(model_name)
        generate_table_6(
            df_dict[model_name]["search"][False], NUM_STORIES_CONST, BUDGET
        )

    # Section 3. A. TRACKTHEMIND finds challenging story structures for frontier models. Figure 3.
    print("Plotting Figure 3...")
    plot_accuracy_all_stories_by_moves(df_dict, NUM_STORIES_CONST)

    # Section 3. B. A* is a better strategy than over-generation and filtering. Figure 6 + numbers reported.
    print("Plotting Figure 6 (A* vs Baseline comparison)...")
    print("Average performance, search vs. baseline without search")
    df = df_dict["meta-llama/Meta-Llama-3.1-70B-Instruct"]["search"][False]
    df_baseline = df_dict["meta-llama/Meta-Llama-3.1-70B-Instruct"]["baseline"][False]
    choosing_best_search_parameters(df, NUM_STORIES_CONST)
    compare_astar_vs_baseline(
        df, df_baseline, NUM_STORIES_CONST, BUDGET, "weight-goal4"
    )

    # Section 3. D. Infilled stories remain challenging. Table 5.
    print("Infilled", len(df_infilled))
    if len(df_infilled) > 0:
        print("Generating Table 5...")
        compare_infilled_stories_vs_structures_accuracy(df_infilled)

    # Section 5, Part 2. Training data biases against theory of mind and its implications.
    print(
        "Computing tables for ``training data biases against theory of mind and its implications"
        "..."
    )
    ## Main paper
    randomly_sampling_story_structures(analysis_to_return="is_false_belief_story")

    ## Appendix (in this order)
    randomly_sampling_story_structures(analysis_to_return="is_false_belief_story")
    randomly_sampling_story_structures(analysis_to_return="ratio_interesting_q")
    randomly_sampling_story_structures(analysis_to_return="ratio_false_belief_q")
