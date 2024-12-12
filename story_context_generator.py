# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import random
import argparse
import pandas as pd
from collections import Counter
from utils import (
    ModelCallHandler,
    _parse_answer_with_separators,
    _remove_noun_articles,
    _dump_dict_list_to_jsonl,
    _parse_fields,
)
from cached_prompt_outputs import PROMPT_OUTPUT_CACHE


class StoryContextGenerator:
    PROMPT_NAME_LISTING = "List 100 names. Do not include any other text."
    PROMPT_STORY_CONTEXT_LISTING = """Suggest 100 different general contexts in which a story may happen. The context should be able to have several people in the same location easily listening and observing each other.

1. a school
2. a hospital
3. a vet shop
4. a family living room

Follow the format and make the descriptions as short as possible. Do not include any text before the list."""

    PROMPT_STORY_CONTEXT = """Suggest a short context where {num_people} people are together in a room. It should be at most two sentences long, and they should be able to observe each other. Later in the story, characters are going to move around and store objects, so your context should be plausible under those constraints. Do not explicitly include that they can all see each other, it should be clear from context. The room could be in a house, work environment, etc.

Here's an example for three people. Follow the same format.

LIST CHARACTERS' NAMES:
1. Emily, a meticulous office manager.
2. Jason, a tech-savvy intern.
3. Karen, a diligent accountant.

GIVE SHORT STORY CONTEXT:
Emily, Jason, and Karen gathered around the central table in the sleek office's conference room, discussing the upcoming audit. As they strategized, the shelves and storage compartments lining the walls around them held the tools and documents they would soon need to organize and pack away.

ROOM IN WHICH THIS STORY BEGINS:

NAME ONE REASONABLE ALTERNATIVE ROOM THEY COULD MOVE TO:

NAME ONE OBJECT TO BE MOVED BY A PERSON DURING THE STORY:

LIST {num_containers} REASONABLE OPAQUE CONTAINERS THAT COULD CONTAIN THIS OBJECT:

LIST {num_topics} DISTINCT AND REASONABLE TOPICS THEY COULD BE CHATTING ABOUT:

To get inspired, make this context happen in {sampled_location}. Suggested names are {sampled_names}, but feel free to come up with your own names if it would suit the story better. Be direct with your answers: do not include parentheses or clarifications beyond the responses requested. Do not refer to plural objects or give options if a singular thing is requested. The object could be anything--an apple, a pen, a spoon, a pair of scissors, a chocolate bar, etc.--, be creative! Avoid vases and microphones, or adding too many details to the object's description in general."""  # vases & microphones happened too often
    PROMPT_STORY_CONTEXT_SEPARATORS = [
        ("LIST CHARACTERS' NAMES:", True),
        ("GIVE SHORT STORY CONTEXT:", False),
        ("ROOM IN WHICH THIS STORY BEGINS:", False),
        ("NAME ONE REASONABLE ALTERNATIVE ROOM THEY COULD MOVE TO:", False),
        ("NAME ONE OBJECT TO BE MOVED BY A PERSON DURING THE STORY:", False),
        (
            "LIST {num_containers} REASONABLE OPAQUE CONTAINERS THAT COULD CONTAIN THIS OBJECT:",
            True,
        ),
        (
            "LIST {num_topics} DISTINCT AND REASONABLE TOPICS THEY COULD BE CHATTING ABOUT:",
            True,
        ),
    ]

    PROMPT_VERIFY_STORY_CONTEXT_IS_REASONABLE = """Given the elements of a story context, are the following statements true?

CHARACTERS:
{people_with_personas}

{enumeration}

Include the itemized questions and do not add any extra text. Answer all items following the format.
"""

    def __init__(self, model_call_handler):
        self.model_call_handler = model_call_handler
        self.model_shortname = self.model_call_handler.model_shortname

    def _sample_list(self, prompt_list_request, **kwargs):
        top_p = kwargs["top_p"]
        temperature = kwargs["temperature"]
        prompt_output_cache_key = (
            prompt_list_request,
            (top_p, temperature),
            self.model_call_handler.model_shortname,
        )
        if prompt_output_cache_key in PROMPT_OUTPUT_CACHE:
            model_output = PROMPT_OUTPUT_CACHE[prompt_output_cache_key]
        else:
            assert (
                False
            ), "We should already have this cached for the models used in the paper (comment this otherwise)."
            model_output, tokens_used = self.model_call_handler.call_model(
                prompt_list_request, **kwargs
            )
        return [e.split(".")[-1].strip() for e in model_output.split("\n")]

    def _get_simple_story_contexts_output_filename(
        self, num_elements_by_class, num_requested_contexts
    ):
        """
        Filename corresponding to the first stage of story context generation (before adding equivalent actions)
        """
        os.makedirs("logs", exist_ok=True)
        num_people, num_moves, num_rooms = (
            num_elements_by_class,
            num_elements_by_class,
            2,
        )
        filepath = f"logs/model_generated_contexts_{self.model_shortname}_n_{num_requested_contexts}_p_{num_people}_m_{num_moves}_r_{num_rooms}.jsonl"
        return filepath

    def _build_prompt_to_verify_story_context_is_reasonable(
        self, people_with_personas, object_name, container_names, room_names, topics
    ):
        enumeration = []
        for container_name_1 in container_names:
            enumeration.append(
                f"* A {object_name} can reasonably fit in the {container_name_1}: (yes/no/unsure)"
            )
        for container_name_1 in container_names:
            enumeration.append(
                f"* It is impossible to see what's inside a {container_name_1} without interacting with the object: (yes/no/unsure)"
            )
        for room_1 in room_names:
            enumeration.append(
                f"* It is possible that a {object_name} is in a {room_1}: (yes/no/unsure)"
            )
        for topic_1 in topics:
            enumeration.append(
                f"* It is plausible for people in this context to be talking about {topic_1}: (yes/no/unsure)"
            )
        expected_answers = ["yes" for _ in range(len(enumeration))]

        llm_judge_prompt = self.PROMPT_VERIFY_STORY_CONTEXT_IS_REASONABLE.format(
            people_with_personas=people_with_personas,
            enumeration="\n".join(enumeration),
        )
        return llm_judge_prompt, expected_answers

    def sample_names_list(self):
        self.name_list = self._sample_list(
            self.PROMPT_NAME_LISTING, top_p=1.0, temperature=0.0
        )

    def sample_location_list(self):
        self.location_list = self._sample_list(
            self.PROMPT_STORY_CONTEXT_LISTING, top_p=1.0, temperature=0.0
        )

    def sample_story_contexts(self, num_elements_by_class, num_requested_contexts):
        sample_context_dict_list = []
        while len(sample_context_dict_list) < num_requested_contexts:
            print("Attempting to create context #", len(sample_context_dict_list))
            # 1. Sample names and initial location
            sampled_names_list = random.sample(self.name_list, num_elements_by_class)
            sampled_location = random.sample(self.location_list, 1)[0]

            # 2. Sample full context
            sampled_context, tokens_used = self.model_call_handler.call_model(
                self.PROMPT_STORY_CONTEXT.format(
                    num_people=num_elements_by_class,
                    num_containers=num_elements_by_class,
                    num_topics=num_elements_by_class,
                    sampled_location=sampled_location,
                    sampled_names=", ".join(sampled_names_list),
                ),
                max_tokens=1000,
                top_p=1.0,
                temperature=0.0,
            )

            # 3. Parse sampled full context into sections
            parsed_parts = _parse_answer_with_separators(
                sampled_context,
                [
                    (
                        sep.format(
                            num_containers=num_elements_by_class,
                            num_topics=num_elements_by_class,
                        ),
                        is_list,
                    )
                    for sep, is_list in self.PROMPT_STORY_CONTEXT_SEPARATORS
                ],
            )
            if parsed_parts is False:
                continue

            # 4. Parse and clean each field
            try:
                people_with_personas = [elem.split(",") for elem in parsed_parts[0]]
                story_context = parsed_parts[1]
                room = parsed_parts[2].strip(".\n ")
                alternative_room = parsed_parts[3].strip(".\n ")
                rooms = [room, alternative_room]
                object_name = parsed_parts[4].strip(".\n ")
                containers = [elem.strip(".\n ") for elem in parsed_parts[5]]
                topics = [elem.strip(".\n ") for elem in parsed_parts[6]]

                sample_context_dict = {
                    "people_names": [
                        people.strip(".\n ")
                        for people, personas in people_with_personas
                    ],
                    "people_personas": [
                        personas.strip(".\n ")
                        for people, personas in people_with_personas
                    ],
                    "people_with_personas": parsed_parts[0],
                    "story_context": story_context,
                    "rooms": [
                        _remove_noun_articles(room),
                        _remove_noun_articles(alternative_room),
                    ],
                    "objects": [_remove_noun_articles(object_name)],
                    "container_names": [_remove_noun_articles(c) for c in containers],
                    "topics": topics,
                }
            except Exception as e:
                print(e)
                continue

            # 5. Verify reasonability with LLM as a judge
            (
                llm_judge_prompt,
                llm_judge_prompt_expected_answers,
            ) = self._build_prompt_to_verify_story_context_is_reasonable(
                sample_context_dict["people_with_personas"],
                sample_context_dict["objects"][0],
                sample_context_dict["container_names"],
                sample_context_dict["rooms"],
                sample_context_dict["topics"],
            )
            output = self.model_call_handler.verify_using_llm_as_judge(
                {}, llm_judge_prompt, llm_judge_prompt_expected_answers
            )
            if not output["is_equivalent"]:
                continue
            sample_context_dict["checked_reasonability"] = True
            sample_context_dict_list.append(sample_context_dict)
        return sample_context_dict_list

    def main(self, num_elements_by_class, num_requested_contexts):
        self.sample_names_list()
        self.sample_location_list()

        # 1. Sample simple story contexts
        filepath = self._get_simple_story_contexts_output_filename(
            num_elements_by_class, num_requested_contexts
        )
        if os.path.exists(filepath):
            sample_context_dict_list = pd.read_json(filepath, lines=True).to_dict(
                "records"
            )
        else:
            sample_context_dict_list = self.sample_story_contexts(
                num_elements_by_class, num_requested_contexts
            )
            _dump_dict_list_to_jsonl(sample_context_dict_list, filepath)

        self.compute_object_statistics(sample_context_dict_list)
        return filepath

    def compute_object_statistics(self, sample_context_dict_list):
        word_counter = Counter()
        for sample_context_dict in sample_context_dict_list:
            for word in sample_context_dict["objects"][0].split(" "):
                word_counter[word.strip(",. :-").lower()] += 1
        print("Most common objects:", word_counter.most_common(n=20))


class ObjectStateUpdatesGenerator:
    STATE_UPDATE_DESIDERATA = None
    LLM_AS_A_JUDGE_PROMPT_STATE_UPDATE_EQUIVALENCE_EXPECTED_ANSWERS = None
    INVISIBLE_STATE_UPDATE_DESIDERATA = [
        "After the state change, the object has a new property",
        "If you were to describe the object to someone, you may mention this new property (either because it's rare, because it's important given the context, or to recognize the object)",  # making sure the change is important
        "The new property is known to people witnessing the state change happening",
        "People that saw the object before the state change (but did not witness the change) would NEVER notice a difference at first glance",  # equivalent between moving from drawer to box (both invisible)
        # "The change is reversible"  # this might not be needed
    ]
    VISIBLE_STATE_UPDATE_DESIDERATA = [
        "After the state change, the object has a new property",
        "If you were to describe the object to someone, you would surely mention this new property",  # making sure the change is important
        "The new property is known to people witnessing the state change happening",
        "People that saw the object before the state change (but did not witness the change) would clearly notice a difference at first glance",  # equivalent between moving from drawer to box (both invisible)
        # "The change is reversible"  # this might not be needed
    ]

    BASE_PROMPT_STATE_UPDATE_EQUIVALENCE_FIELDS = [
        "ACTION CAUSING AN OBJECT STATE CHANGE:",
        "PROPERTY:",
    ]
    BASE_PROMPT_STATE_UPDATE_EQUIVALENCE = """OBJECT: Jam jar
ACTION CAUSING AN OBJECT STATE CHANGE: Moving the jam jar to the fridge
PROPERTY: object location

OBJECT: Apple
ACTION CAUSING AN OBJECT STATE CHANGE: Poisoning an apple
PROPERTY: is object edible

OBJECT: Book
ACTION CAUSING AN OBJECT STATE CHANGE: Writing a small note in one of its pages in pencil
PROPERTY: object has secret note

Generate examples for the following object. I'll also give you context so that you know what the object might be used for. Give 10 diverse ACTION CAUSING AN OBJECT STATE CHANGE, with the associated property. Give examples of object state changes that fulfill the following properties:

{desiderata}

Do not include any text before the examples.

CONTEXT: {{story_context}}
OBJECT: {{object_name}}

Respect the format shown and do not add any extra text before or as explanation. Do not enumerate. Do not include tracking devices.

ACTION CAUSING AN OBJECT STATE CHANGE: <fill>
PROPERTY: <fill>
"""

    BASE_PROMPT_VERIFY_STATE_UPDATE_EQUIVALENCE = """Given an object and a state change, are the following statements true? A property can be a location, a texture, a color, a distinguishing thing, etc.

OBJECT: {{object_name}}
ACTION CAUSING AN OBJECT STATE CHANGE: {{state_change}}

{desiderata}

Include the itemized questions and do not add any extra text. Answer all items following the format.
"""

    PROMPT_DSL_COMPATIBLE_FORMAT = """Given a state change, write what the resulting state after it will be.

OBJECT NAME: Laptop
STATE CHANGE: Decaling laptop with blue tribal stickers
PARSED ACTION WITH SUBJECT: <person> decaled the <object> with blue tribal stickers.
RESULTING STATE OF THE OBJECT: is decaled with blue tribal stickers

Respect the format shown and do not add any extra text before or as explanation. Include responses to all fields.

OBJECT NAME: {object_name}
STATE CHANGE: {state_change}
PARSED ACTION WITH SUBJECT:
RESULTING STATE OF THE OBJECT:"""
    PROMPT_DSL_COMPATIBLE_FORMAT_FIELDS = [
        "OBJECT NAME:",
        "STATE CHANGE:",
        "PARSED ACTION WITH SUBJECT:",
        "RESULTING STATE OF THE OBJECT:",
    ]

    def __init__(self, model_call_handler):
        self.model_call_handler = model_call_handler
        self.model_shortname = self.model_call_handler.model_shortname

    def _get_filename_state_update_without_dsl_compatible(self, base_filename):
        return (
            base_filename[: -len(".jsonl")] + "_update_object_state_equiv_class.jsonl"
        )

    def _get_filename_state_update_dsl_compatible(self, base_filename):
        return (
            base_filename[: -len(".jsonl")]
            + f"_update_object_state_equiv_class_for_v1_dsl_wo_upsampling.jsonl"
        )

    def sample_object_state_update(
        self, sample_context_dict_list, num_contexts_to_fill=50
    ):
        model_generated_context_dict_list = sample_context_dict_list[
            :num_contexts_to_fill
        ]
        equivalent_actions_list = [
            {} for _ in range(len(model_generated_context_dict_list))
        ]

        # Main: Sample equivalent actions for both visible and invisible changes
        for update_type_to_sample, STATE_UPDATE_DESIDERATA in [
            ("update_object_state-invisible", self.INVISIBLE_STATE_UPDATE_DESIDERATA),
            ("update_object_state-visible", self.VISIBLE_STATE_UPDATE_DESIDERATA),
        ]:

            # 0. Infill prompts that will be required for sampling and vrifying
            desiderata_enum = "\n".join([f"* {e}" for e in STATE_UPDATE_DESIDERATA])
            desiderata_enum_with_qs = "\n".join(
                [f"* {e}: (yes/no/unsure)" for e in STATE_UPDATE_DESIDERATA]
            )
            PROMPT_STATE_UPDATE_EQUIVALENCE = (
                self.BASE_PROMPT_STATE_UPDATE_EQUIVALENCE.format(
                    desiderata=desiderata_enum
                )
            )
            PROMPT_VERIFY_STATE_UPDATE_EQUIVALENCE = (
                self.BASE_PROMPT_VERIFY_STATE_UPDATE_EQUIVALENCE.format(
                    desiderata=desiderata_enum_with_qs
                )
            )
            PROMPT_VERIFY_STATE_UPDATE_EQUIVALENCE_EXPECTED_ANSWERS = ["yes"] * len(
                STATE_UPDATE_DESIDERATA
            )

            for i in range(len(model_generated_context_dict_list)):
                story_context = model_generated_context_dict_list[i]["story_context"]
                object_name = model_generated_context_dict_list[i]["objects"][0]

                # 1. Sample candidates to equivalent object update actions for each existing context
                model_response, tokens_used = self.model_call_handler.call_model(
                    PROMPT_STATE_UPDATE_EQUIVALENCE.format(
                        story_context=story_context, object_name=object_name
                    ),
                    max_tokens=400,
                    top_p=1.0,
                    temperature=1.0,
                )

                # 2. Verify which candidates to keep
                candidates_to_keep = []
                for state_change, property_name in _parse_fields(
                    model_response, self.BASE_PROMPT_STATE_UPDATE_EQUIVALENCE_FIELDS
                ):
                    fields_dict = {
                        "state_change": state_change,
                        "object_name": object_name,
                    }
                    result = self.model_call_handler.verify_using_llm_as_judge(
                        fields_dict,
                        PROMPT_VERIFY_STATE_UPDATE_EQUIVALENCE,
                        PROMPT_VERIFY_STATE_UPDATE_EQUIVALENCE_EXPECTED_ANSWERS,
                    )
                    if result["is_equivalent"]:
                        fields_dict["property_name"] = property_name
                        fields_dict["story_context"] = story_context
                        candidates_to_keep.append(fields_dict)
                equivalent_actions_list[i][update_type_to_sample] = candidates_to_keep

        # 3. Once all equivalent actions have been sampled, update the initial setting
        for i in range(len(equivalent_actions_list)):
            model_generated_context_dict_list[i][
                "class_equivalences"
            ] = equivalent_actions_list[i]
        return model_generated_context_dict_list

    def make_data_dsl_compatible(self, model_generated_context_dict_list):
        """
        Make the sampled object state updates suitable to use with our DSL that requires having the resulting state.

        TOMAKEEFFICIENT: This could've been done in the same call as the object state update generation,
        but I'm keeping the original implementation.
        """

        for i1, e in enumerate(model_generated_context_dict_list):
            for k in e["class_equivalences"]:
                for i2, elem in enumerate(e["class_equivalences"][k]):
                    model_response, _ = self.model_call_handler.call_model(
                        self.PROMPT_DSL_COMPATIBLE_FORMAT.format(
                            object_name=elem["object_name"],
                            state_change=elem["state_change"],
                        ),
                        temperature=0.0,
                        max_tokens=400,
                    )
                    output = _parse_fields(
                        model_response, self.PROMPT_DSL_COMPATIBLE_FORMAT_FIELDS
                    )
                    if len(output) != 1 or len(output[0]) != len(
                        self.PROMPT_DSL_COMPATIBLE_FORMAT_FIELDS
                    ):
                        print("Could not parse.", output)
                        continue
                    output = output[0]  # only one item to be parsed
                    text_description = output[2]
                    resulting_state = output[3]
                    if (
                        "<person>" not in text_description
                        or "<object>" not in text_description
                    ):
                        continue
                    text_description = text_description.replace(
                        "<person>", "{person}"
                    ).replace("<object>", "{object}")
                    model_generated_context_dict_list[i1]["class_equivalences"][k][i2][
                        "text_description"
                    ] = text_description
                    model_generated_context_dict_list[i1]["class_equivalences"][k][i2][
                        "resulting_state"
                    ] = resulting_state

                # filter out failed cases
                model_generated_context_dict_list[i1]["class_equivalences"][k] = [
                    elem
                    for elem in model_generated_context_dict_list[i1][
                        "class_equivalences"
                    ][k]
                    if "text_description" in elem
                ]
        return model_generated_context_dict_list

    def main(self, base_filename, num_contexts_to_fill):
        sample_context_dict_list = pd.read_json(base_filename, lines=True).to_dict(
            "records"
        )

        filepath = self._get_filename_state_update_without_dsl_compatible(base_filename)
        if os.path.exists(filepath):
            sample_context_dict_list = pd.read_json(filepath, lines=True).to_dict(
                "records"
            )
        else:
            sample_context_dict_list = self.sample_object_state_update(
                sample_context_dict_list, num_contexts_to_fill=num_contexts_to_fill
            )
            _dump_dict_list_to_jsonl(sample_context_dict_list, filepath)

        filepath_dsl_compatible = self._get_filename_state_update_dsl_compatible(
            base_filename
        )
        if os.path.exists(filepath_dsl_compatible):
            sample_context_dict_list = pd.read_json(
                filepath_dsl_compatible, lines=True
            ).to_dict("records")
        else:
            sample_context_dict_list = self.make_data_dsl_compatible(
                sample_context_dict_list
            )
            _dump_dict_list_to_jsonl(sample_context_dict_list, filepath_dsl_compatible)
        return filepath_dsl_compatible


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
    story_context_generator = StoryContextGenerator(model_call_handler)
    state_updates_generator = ObjectStateUpdatesGenerator(model_call_handler)
    filepath = story_context_generator.main(
        num_elements_by_class=args.num_elements_by_class,
        num_requested_contexts=args.num_contexts_to_generate,
    )
    filepath = state_updates_generator.main(
        filepath, num_contexts_to_fill=args.num_contexts_to_generate
    )
    print("Final generated context is in", filepath)
