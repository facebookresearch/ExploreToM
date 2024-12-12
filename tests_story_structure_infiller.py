# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from utils import ModelCallHandler
from belief_tracker import FullBeliefTracker
from story_structure_infiller import (
    StoryStructureInfiller,
    StoryInfillingQualityVerifier,
)


def test_infilling_story_with_lack_of_attention(story_structure_infiller):
    """
    Caleb entered the green room.
    Cooper entered the green room.
    Caleb told out loud about the upcoming comedy festivals. While this was happening, Cooper got distracted, without anyone noticing the brief lack of attention.
    Caleb told out loud about the tonight's set list.
    Caleb told out loud about the favorite comedy influences.
    """
    function_list = [
        lambda o: o.enter_room("Caleb", "green room"),
        lambda o: o.enter_room("Cooper", "green room"),
        lambda o: o.broadcast_communication(
            "Caleb", "upcoming comedy festivals", people_distracted=["Cooper"]
        ),
        lambda o: o.broadcast_communication("Caleb", "tonight's set list"),
        lambda o: o.broadcast_communication("Caleb", "favorite comedy influences"),
    ]
    belief_tracker = FullBeliefTracker()
    for f in function_list:
        a = f(belief_tracker)
        assert a, belief_tracker.story_script
    use_custom_names_dict = {}
    use_custom_names_dict["people_names"] = ["Caleb", "Cooper"]
    use_custom_names_dict["people_with_personas"] = [
        "Caleb, a festival producer",
        "Cooper, an intern",
    ]

    (
        df_list,
        (belief_tracker, function_list),
        infilled_story,
        is_story_valid,
        log_infilling,
    ) = story_structure_infiller.main(
        belief_tracker, function_list, use_custom_names_dict
    )
    final_prediction = log_infilling[-1]
    is_story_valid = (
        final_prediction["is_story_structure_valid"]
        and final_prediction["is_story_surface_valid"]
        and final_prediction.get("is_adhoc_checks_valid", True)
    )
    print("infilled", infilled_story)
    print("is_story_valid", is_story_valid)
    return is_story_valid


def test_case_check_verifier_receptacle():
    """
    Test for checking if the verifier can tell that the infilling is wrong (the infiller changed the name of the plastic bin to "receptacle").
    Test case created based on issues found on human evaluation.

    STORY STRUCTURE.
    =========
    Sara entered the activity room.
    Victoria entered the activity room.
    Victoria moved the patient file to the cardboard box, which is also located in the activity room.
    Benjamin entered the activity room.
    Victoria told privately to Peyton that the patient file is in the cardboard box.
    Sara moved the patient file to the plastic bin, which is also located in the activity room.
    """

    function_list = [
        lambda o: o.enter_room("Sara", "activity room"),
        lambda o: o.enter_room("Victoria", "activity room"),
        lambda o: o.move_object_container(
            "Victoria", obj="patient file", container="cardboard box"
        ),
        lambda o: o.enter_room("Benjamin", "activity room"),
        lambda o: o.private_communication(
            "Victoria",
            "Peyton",
            topic=(
                "patient file",
                FullBeliefTracker.CONTAINER_LOCATION,
                "cardboard box",
                True,
            ),
        ),
        lambda o: o.move_object_container(
            "Sara", obj="patient file", container="plastic bin"
        ),
    ]

    model_generated_infilling = [
        "The activity room, with its polished floor and rows of neatly arranged tables, was bathed in the warm glow of morning sunlight streaming through the large windows. The air was filled with the gentle hum of murmured conversations, the soft shuffling of papers, and the faint scent of disinfectant.",
        "The warm morning atmosphere of the activity room was momentarily disrupted as Sara stepped inside, her eyes scanning the space with a purposeful air, followed closely by Victoria, whose sharp gaze swiftly swept the room.",
        "Victoria's eyes darted swiftly across the tables, homing in on a discreet, unobtrusive cardboard box in the corner of the room as she skillfully relocated the patient file, reestablishing a sense of organizational order.",
        "Benjamin's arrival brought a fresh wave of energy to the already bustling activity room, his eyes scanning the space as he sought Sara's attention.",
        "Victoria's voice was low and discreet as she privately informed Peyton, "
        "The misplaced file is safely stored where it won't catch anyone's eye.",
        "Sara smoothly restored order in the activity room by relocating the patient file to the designated receptacle.",
    ]

    belief_tracker = FullBeliefTracker()
    for f in function_list:
        a = f(belief_tracker)
        assert a
    return belief_tracker, function_list, model_generated_infilling


def run_verifier_on_proposed_infilling(
    story_structure_infiller, belief_tracker, function_list, next_story_step_full_list
):
    story_context = None

    (
        grouped_story_script,
        grouped_story_script_indices,
    ) = story_structure_infiller._group_actions_for_infilling(
        belief_tracker.story_script, belief_tracker.story_script_tuple_representation
    )
    grouped_story_script = [" ".join(e) for e in grouped_story_script]

    cur_belief_tracker = FullBeliefTracker()
    assert (
        len(grouped_story_script)
        == len(grouped_story_script_indices)
        == len(next_story_step_full_list) - 1
    )  # the -1 is because of the sampled story context
    for new_information, new_information_indices, next_story_step in zip(
        grouped_story_script,
        grouped_story_script_indices,
        next_story_step_full_list[1:],
    ):
        new_action_function = [function_list[i] for i in new_information_indices]
        cur_story_context = (
            next_story_step
            if story_context is None
            else story_context + " " + next_story_step
        )
        (
            next_story_step_accepted,
            belief_tracker_new,
            tmp_for_log_infilling_clean,
        ) = story_structure_infiller.quality_verifier.main(
            cur_belief_tracker,  # partial belief_tracker
            new_action_function,  # function_list for this specific step
            cur_story_context,  # accumulated infillings
            next_story_step,  # infilled next step
            new_information,  # raw next steps
        )
        if next_story_step_accepted:
            cur_belief_tracker = belief_tracker_new
            story_context = cur_story_context
        else:
            return False
    return True


def __assert_value(test_name, pred_value, expected_value):
    if pred_value == expected_value:
        print(f"Test {test_name} succeeded.")
    else:
        print(f"Test {test_name} failed.")


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
        "--num_elements_by_class",
        type=int,
        default=6,
        help="Number of elements (objects, containers, etc.) of each class to generate",
    )
    parser.add_argument(
        "--num_contexts_to_generate",
        type=int,
        default=5,
        help="Number of contexts to generate",
    )
    args = parser.parse_args()

    assert (
        args.num_elements_by_class <= 100 and args.num_contexts_to_generate <= 100
    ), "Llama refuses to generate a longer list so for simplicity we cap to 100."

    model_call_handler = ModelCallHandler(args.model_name, args.model_access_method)
    story_structure_infiller = StoryStructureInfiller(
        model_call_handler, sample_all_next_step_completions_simultaneously=True
    )

    # test verifier
    (
        belief_tracker,
        function_list,
        model_generated_infilling,
    ) = test_case_check_verifier_receptacle()
    verifier_succeeded = run_verifier_on_proposed_infilling(
        story_structure_infiller,
        belief_tracker,
        function_list,
        model_generated_infilling,
    )
    __assert_value("test_case_check_verifier_receptacle", verifier_succeeded, False)

    # test if infiller eventually gives an infilling or not
    infilling_suceeded = False
    for _ in range(4):
        infilling_suceeded = test_infilling_story_with_lack_of_attention(
            story_structure_infiller
        )
        if infilling_suceeded:
            break
    __assert_value(
        "test_infilling_story_with_lack_of_attention", infilling_suceeded, True
    )
