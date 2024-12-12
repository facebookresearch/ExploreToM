# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
import copy
import dill
import random
import argparse
import pandas as pd
from utils import ModelCallHandler, ModelAnswerEvaluator
from belief_tracker import FullBeliefTracker

"""
Pending stuff:
- TODO: story infilling output is probably in a bad output format
"""


class StoryInfillingQualityVerifier:
    """
    Object to verify the story infilling quality across several axes using an LLM as a judge.
    """

    #### ************ STORY STRUCTURE VERIFICATION ************
    JUDGING_STORY_STRUCTURE_PROMPT = """We are reading through a long story. We know the current state of the world and what everyone believes to be true. Then, an action happens. Answer whether each statement is true given the previous world state and the action that occurred. You can only answer "yes", "no", or "unsure" if you do not know the answer. 

THINGS TO NOTE:
* If a person performed an action, then they know about it.
* If an object was moved to a container, it still remains in the same room unless otherwise specified. Thus, the object is still in the room and you should answer 'yes' when asked if it is still in the room.
* Usually, if two people are in the same location then they can see each other unless there is a reason to believe otherwise (e.g. someone is distracted, the location is too large or too crowded, etc.). Similarly, if someone is in a location one can assume they observed if something happened, unless there is a reason to believe otherwise.

PREVIOUS WORLD STATE AND BELIEFS:
{prev_world_state}

STORY ACTION THAT OCCURRED:
{story_action}

ANSWER IF EACH CURRENT WORLD STATE AND BELIEFS ARE CORRECT.
{new_world_state}

Pay special attention to "THINGS TO NOTE" when answering. Include the itemized questions and do not add any extra text, or answer explanation, or any clarification. Answer all items following the format.
"""
    CLARIFYING_ANSWERS_PROMPT_TITLE = "CORRECTED RESPONSES:"
    CLARIFYING_ANSWERS_PROMPT = f'Why did you answer the way you did? Re-read the "THINGS TO NOTE" items and try again. If you feel like you want to correct a response, give the corrected responses after a title named "{CLARIFYING_ANSWERS_PROMPT_TITLE}". Do not include any extra text or answer clarification after "{CLARIFYING_ANSWERS_PROMPT_TITLE}", but feel free to think out loud as much as you need before that. After {CLARIFYING_ANSWERS_PROMPT_TITLE} you should only answer each question with a single word, and NO CLARIFICATIONS. '  # Why did you answer the way you did? Are you happy with your responses? If you feel like you want to correct a response, give the corrected responses after a title named "CORRECTED RESPONSES:".  Do not include any extra text after "CORRECTED RESPONSES", but feel free to think out loud as much as you need before that.

    #### ************ STORY SURFACE VERIFICATION ************
    JUDGING_STORY_SURFACE_FORM_PROMPT = """We are writing a long story and want to make sure that the writing is of high quality.

STORY WRITTEN SO FAR:
"{accum_story}"

NEW STORY STEP:
"{story_action}"

ANSWER WHETHER THE NEW STORY STEP COMPLIES WITH THESE CONDITIONS:
{itemized_questions}

Include the itemized questions and do not add any extra text. Answer all items following the format.
"""
    JUDGING_STORY_SURFACE_FORM_PROMPT_QUESTIONS = [
        "* The next story step is reasonable, coherent English text: (yes/no/unsure)",
        "* The next story step adds writer notes or comments that should not be included in the final text: (yes/no/unsure)",
    ]
    JUDGING_STORY_SURFACE_FORM_PROMPT_EXPECTED_ANSWERS = ["yes", "no"]

    #### ************ AD HOC CHECKS VERIFICATION ************
    JUDGING_STORY_FREQUENT_ERRORS_PROMPT = """We are writing a story and want to make sure that the writing is of high quality. I translated from a sentence from a draft version to a final one.

NEW STORY ACTION DRAFT VERSION:
"{story_action_raw}"

NEW STORY ACTION FINAL VERSION:
"{story_action_infilled}"

ANSWER WHETHER THE NEW STORY ACTION *FINAL* VERSION COMPLIES WITH THESE CONDITIONS:
{itemized_questions}

Include the itemized questions and do not add any extra text. Answer all items following the format. If the hypothetical condition of a question does not apply to the case we're evaluating, then answer "yes" by default.
"""
    JUDGING_STORY_FREQUENT_ERRORS_PROMPT_QUESTIONS = [
        "* If there is a secret witness to the action, it is reasonable to believe that the person stops witnessing at the end of this action: (yes/no/unsure)",
        "* If there is a secret witness to the action, it is reasonable to believe that no one else realize someone was spying: (yes/no/unsure)",
        "* If there is a person distracted during the action, it is reasonable to believe that the person stops being distracted at the end of this action: (yes/no/unsure)",
        "* If there is a person distracted during the action, it is reasonable to believe that no one else realized they were distracted: (yes/no/unsure)",
        "* If there is a person distracted during the action, it is reasonable to believe that they did not realize that the action during their distraction happened at all: (yes/no/unsure)",
        "* If there is an object being mentioned in the draft version of the action, then it also is being mentioned in the final version (answer 'yes' if no object is mentioned): (yes/no/unsure)",
        "* If there is an object being modified in the draft version of the action, then it also is being modified in the final version (answer 'yes' if no object is modified): (yes/no/unsure)",
        "* If a topic is discussed in the draft version of the action, then it is also discussed in the final version (answer 'yes' if no topic is discussed): (yes/no/unsure)",
    ]
    JUDGING_STORY_FREQUENT_ERRORS_PROMPT_EXPECTED_ANSWERS = ["yes"] * len(
        JUDGING_STORY_FREQUENT_ERRORS_PROMPT_QUESTIONS
    )

    def __init__(self, model_call_handler, **kwargs):
        self.model_call_handler = model_call_handler
        self.num_tries_llm_judge_structure = kwargs.get(
            "num_tries_llm_judge_structure", 2
        )
        self.num_tries_llm_judge_surface = kwargs.get("num_tries_llm_judge_surface", 1)
        self.num_tries_llm_judge_adhoc = kwargs.get("num_tries_llm_judge_adhoc", 1)

    def _turn_all_textual_beliefs_into_enumeration(
        self,
        prev_world_state_list,
        prev_first_order_beliefs_list,
        prev_second_order_beliefs_list,
        question_suffix="",
        shuffle_statements=False,
    ):
        if shuffle_statements:
            random.shuffle(prev_world_state_list)
            random.shuffle(prev_first_order_beliefs_list)
            random.shuffle(prev_second_order_beliefs_list)
        prev_world_state = "\n".join(
            [f"* {e}{question_suffix}" for e in prev_world_state_list]
        )
        prev_first_order_beliefs = "\n".join(
            [f"* {e}{question_suffix}" for e in prev_first_order_beliefs_list]
        )
        prev_second_order_beliefs = "\n".join(
            [f"* {e}{question_suffix}" for e in prev_second_order_beliefs_list]
        )
        return prev_world_state, prev_first_order_beliefs, prev_second_order_beliefs

    @staticmethod
    def _parse_llm_as_a_judge_output(cur_world_state, model_response):
        cur_world_state_lines = cur_world_state.split("\n")
        cur_world_state_lines = sorted(
            [
                ws.rstrip("(yes/no/unsure)").lower().replace("?", "")
                for ws in cur_world_state_lines
            ]
        )
        model_response = [
            line.lower().replace("?", "")
            for line in model_response.strip().split("\n")
            if line.startswith("*")
        ]
        model_response = [
            mr_line
            for mr_line in model_response
            if any(mr_line.startswith(ws_line) for ws_line in cur_world_state_lines)
        ]
        model_response = sorted(model_response)
        if len(cur_world_state_lines) != len(model_response):
            return [], False

        responses = []
        for clean_ws, model_response_elem in zip(cur_world_state_lines, model_response):
            model_response_elem = model_response_elem.lower().replace("?", "")
            model_output = model_response_elem[len(clean_ws) :].strip()
            if not model_response_elem.startswith(clean_ws) or model_output not in [
                "yes",
                "no",
                "unsure",
            ]:
                return [], False
            responses.append((clean_ws, model_output))
        return responses, True

    def check_story_structure(
        self,
        prev_belief_tracker,
        cur_belief_tracker,
        shuffle_statements,
        new_action_text,
        skip_second_order_qs=False,
    ):
        (
            prev_world_state_list,
            prev_first_order_beliefs_list,
            prev_second_order_beliefs_list,
        ) = prev_belief_tracker.get_all_beliefs_in_textual_form()
        (
            prev_world_state,
            prev_first_order_beliefs,
            prev_second_order_beliefs,
        ) = self._turn_all_textual_beliefs_into_enumeration(
            prev_world_state_list,
            prev_first_order_beliefs_list,
            prev_second_order_beliefs_list,
        )

        (
            cur_world_state_list,
            cur_first_order_beliefs_list,
            cur_second_order_beliefs_list,
        ) = cur_belief_tracker.get_all_beliefs_in_textual_form()
        (
            cur_world_state,
            cur_first_order_beliefs,
            cur_second_order_beliefs,
        ) = self._turn_all_textual_beliefs_into_enumeration(
            cur_world_state_list,
            cur_first_order_beliefs_list,
            cur_second_order_beliefs_list,
            question_suffix="?: (yes/no/unsure)",
            shuffle_statements=shuffle_statements,
        )

        num_calls = 0
        to_analyze = [
            (prev_world_state, cur_world_state, "world_state"),
            (
                prev_world_state + "\n" + prev_first_order_beliefs,
                cur_first_order_beliefs,
                "first_order",
            ),
            (
                prev_world_state
                + "\n"
                + prev_first_order_beliefs
                + "\n"
                + prev_second_order_beliefs,
                cur_second_order_beliefs,
                "second_order",
            ),
        ]
        if skip_second_order_qs:
            to_analyze = to_analyze[:2]
        for prev, cur, ws_type in to_analyze:
            if not cur:
                continue
            llm_as_a_judge_succeeded, responses = False, []

            all_checked_ok = False
            filled_prompt = self.JUDGING_STORY_STRUCTURE_PROMPT.format(
                prev_world_state=prev, story_action=new_action_text, new_world_state=cur
            )
            partial_chat, model_response = [
                {"role": "user", "content": filled_prompt}
            ], None
            for retry_idx in range(self.num_tries_llm_judge_structure):
                if retry_idx > 0:
                    partial_chat.extend(
                        [
                            {"role": "assistant", "content": model_response},
                            {"role": "user", "content": self.CLARIFYING_ANSWERS_PROMPT},
                        ]
                    )
                model_response, tokens_used = self.model_call_handler.call_model(
                    partial_chat, max_tokens=800, top_p=1.0, temperature=0.0
                )
                model_response = model_response.lower()
                num_calls += 1
                if retry_idx and self.CLARIFYING_ANSWERS_PROMPT_TITLE in model_response:
                    model_response = model_response.split(
                        self.CLARIFYING_ANSWERS_PROMPT_TITLE
                    )[-1]
                responses, llm_as_a_judge_succeeded = self._parse_llm_as_a_judge_output(
                    cur, model_response
                )
                if llm_as_a_judge_succeeded and all(r == "yes" for q, r in responses):
                    all_checked_ok = True
                    break

            if not all_checked_ok:
                if not llm_as_a_judge_succeeded:
                    return False, [], num_calls
                elif any(r != "yes" for q, r in responses):
                    return (
                        False,
                        [(q, r) for q, r in responses if r != "yes"],
                        num_calls,
                    )
        return True, [], num_calls

    def check_story_structure_with_shuffling_retries(
        self,
        cur_belief_tracker,
        next_belief_tracker,
        new_action_text,
        skip_second_order_qs=False,
    ):
        # total: check_story_structure_with_shuffling_retries ** 2 calls

        possible_issues_list = []
        output, possible_issues = False, []
        num_calls_total = 0
        for retry_idx in range(self.num_tries_llm_judge_structure):
            # shuffle statements to answer and see if the issue persists
            output, possible_issues, num_calls = self.check_story_structure(
                cur_belief_tracker,
                next_belief_tracker,
                shuffle_statements=(retry_idx > 0),
                new_action_text=new_action_text,
                skip_second_order_qs=skip_second_order_qs,
            )
            num_calls_total += num_calls
            if output is True:
                possible_issues_list = []
                break
            possible_issues_list.append(possible_issues)
        return output, possible_issues_list, num_calls_total

    def check_story_surface_form_with_retries(self, accum_story, next_story_step):
        assert self.num_tries_llm_judge_surface == 1
        possible_issues_list = []
        output, possible_issues = False, []
        num_calls_total = 0
        for retry_idx in range(self.num_tries_llm_judge_surface):
            questions_ordered_indices = list(
                range(len(self.JUDGING_STORY_SURFACE_FORM_PROMPT_QUESTIONS))
            )
            if retry_idx > 0:
                random.shuffle(questions_ordered_indices)
            qs = "\n".join(
                [
                    self.JUDGING_STORY_SURFACE_FORM_PROMPT_QUESTIONS[i]
                    for i in questions_ordered_indices
                ]
            )
            expected_answers = [
                self.JUDGING_STORY_SURFACE_FORM_PROMPT_EXPECTED_ANSWERS[i]
                for i in questions_ordered_indices
            ]

            model_response, tokens_used = self.model_call_handler.call_model(
                self.JUDGING_STORY_SURFACE_FORM_PROMPT.format(
                    accum_story=accum_story,
                    story_action=next_story_step,
                    itemized_questions=qs,
                ),
                max_tokens=800,
                top_p=1.0,
                temperature=0.0,
            )
            num_calls_total += 1

            answers = [
                e.split(": ")[1] for e in model_response.split("\n") if ": " in e
            ]
            output = answers == expected_answers
            if output is True:
                possible_issues_list = []
                break
            possible_issues_list.append(possible_issues)

        return output, possible_issues_list, num_calls_total

    def adhoc_story_checks_with_shuffling_retries(
        self, story_action_raw, story_action_infilled
    ):
        """
        Some ad-hoc checks are needed based on early annotations.

        These recommendations should also be added to the infilling prompt. [TODO]

        - When you ask 'Victoria believes that the patient file is in the plastic bin',
        the answer is yes even if you use a hyponym (receptacle) since you can only be referring to the plastic bin.
        This however is terrible for annotation, because it changes the label, and it might also confuse human annotators.
        - you need to make sure that it's clear that the person stops spying after the action.

        # * Sara believes that Victoria is in the activity room?: (yes/no/unsure)
        """

        # IDEAS
        # each object is unambiguously mentioned
        # * there are no two copies of the same object (e.g. two different groups of strawberries)
        # it is clear that <INFILLED> is a better description of <RAW>, including A. the action is clear B. it clearly contains all the information that RAW contained (e.g. it is clear where you are moving the obejct to) ---> this should be already checked. [gave 'no' too often]
        # Do not assume a person is in the same room if it has not been made explicit before. Also, if someone was spying, or if they were distracted and did not listen/saw something happen, do not forget to include it! Remember that in this case, it should be clear that the distraction or spying applies only to the action mentioned. Avoid making multiple copies of the same object.

        possible_issues_list = []
        output, possible_issues = False, []
        num_calls_total = 0
        for retry_idx in range(self.num_tries_llm_judge_adhoc):
            questions_ordered_indices = list(
                range(len(self.JUDGING_STORY_FREQUENT_ERRORS_PROMPT_QUESTIONS))
            )
            if retry_idx > 0:
                random.shuffle(questions_ordered_indices)
            qs = "\n".join(
                [
                    self.JUDGING_STORY_FREQUENT_ERRORS_PROMPT_QUESTIONS[i]
                    for i in questions_ordered_indices
                ]
            )
            expected_answers = [
                self.JUDGING_STORY_FREQUENT_ERRORS_PROMPT_EXPECTED_ANSWERS[i]
                for i in questions_ordered_indices
            ]
            model_response, tokens_used = self.model_call_handler.call_model(
                self.JUDGING_STORY_FREQUENT_ERRORS_PROMPT.format(
                    story_action_raw=story_action_raw,
                    story_action_infilled=story_action_infilled,
                    itemized_questions=qs,
                ),
                max_tokens=800,
                top_p=1.0,
                temperature=0.0,
            )
            num_calls_total += 1

            answers = [
                e.split(": ")[1] for e in model_response.split("\n") if ": " in e
            ]
            output = answers == expected_answers
            if output is True:
                possible_issues_list = []
                break
            possible_issues_list.append(possible_issues)
        return output, possible_issues_list, num_calls_total

    def main(
        self,
        cur_belief_tracker,
        next_belief_tracker,
        next_story_context,
        next_story_step,
        next_story_step_raw,
    ):
        """
        Verify if the next step in the story makes sense from a structure, surface, and adhoc checks perspective.

        - cur_belief_tracker does not include the new action, next_* does.
        """
        accum_infilled_story_without_new_step = next_story_context[
            : -len(next_story_step)
        ]
        (
            is_story_surface_valid,
            possible_issues_surface_list,
            num_calls_llm_judge,
        ) = self.check_story_surface_form_with_retries(
            accum_infilled_story_without_new_step,
            next_story_step,
        )

        (
            is_story_structure_valid,
            possible_issues_structure_list,
            num_calls_llm_judge_2,
        ) = self.check_story_structure_with_shuffling_retries(
            cur_belief_tracker,
            next_belief_tracker,
            next_story_step,
            skip_second_order_qs=True,
        )  # skipping second order qs since models are quite bad on them

        (
            is_adhoc_checks_valid,
            possible_issues_adhoc_checks_list,
            num_calls_llm_judge_3,
        ) = self.adhoc_story_checks_with_shuffling_retries(
            next_story_step_raw,
            next_story_step,
        )

        tmp_for_log_infilling = {
            "story_context": "Empty"
            if next_story_context is None
            else next_story_context,
            "next_story_step": next_story_step,
            "is_story_surface_valid": is_story_surface_valid,
            "possible_issues_surface_list": possible_issues_surface_list,
            "is_story_structure_valid": is_story_structure_valid,
            "possible_issues_structure_list": possible_issues_structure_list,
            "is_adhoc_checks_valid": is_adhoc_checks_valid,
            "possible_issues_adhoc_checks_list": possible_issues_adhoc_checks_list,
            "num_calls_llm_judge_surface": num_calls_llm_judge,
            "num_calls_llm_judge_structure": num_calls_llm_judge_2,
            "num_calls_llm_judge_adhoc": num_calls_llm_judge_3,
        }

        next_story_step_accepted = (
            is_story_structure_valid
            and is_story_surface_valid
            and is_adhoc_checks_valid
        )
        return next_story_step_accepted, tmp_for_log_infilling


class StoryStructureInfiller:
    """
    Given a FullBeliefTracker that represents a story structure, infill stories to make them sound natural
    while still maintaining faithfulness.
    """

    SYSTEM_PROMPT_FOR_WRITER = "You are an expert writer that uses simple language, avoiding sounding unnatural or cliché. You are clear, creative, and helpful. You use simple sentence constructions and words so that everyone may understand you."
    PROMPT_INITIAL_STORY_CONTEXT = """Given the following story and knowing the description of the characters involved, write the start of a story. Don't actually describe any actions in the story, just the setting in which the story will happen. Only include the characters that are mentioned in the story.

STORY:
{story_script}

CHARACTERS:
{characters_description}

TWO-SENTENCE STORY BEGINNING THAT DOES NOT INCLUDE OR SUGGEST ANY INFORMATION OF WHAT WILL HAPPEN IN THE STORY. DO NOT MENTION PEOPLE:
"""

    PROMPT_CHARACTER_GOALS = """Given the following story and knowing the description of the characters involved, suggest a reasonable goal for each character. Only include the characters that were mentioned in the story.

STORY:
{story_script}

CHARACTERS:
{characters_description}

CHARACTERS GOALS: <insert here>

Follow the format and do not include any other text. Only include the characters mentioned in the story, and do not even mention the others in your list.
"""
    PROMPT_STORY_INFILLING_VERSIONS = {
        "story_length": ["with a single sentence", "with up to two sentences"],
        "infilling_text_type": [
            "Make the new text be declarative, without including conversations.",
            "Make the new text conversational, using direct quotes to convey the words spoken by a character.",
        ],
    }

    PROMPT_STORY_INFILLING_WITHOUT_GOALS = """Continue the story {story_length}, clearly conveying the action or information below without altering it. Do not contradict any prior information. Avoid repeating the information verbatim, instead naturally (and possibly implicitly, but still unambiguously) conveying the meaning. Do not add characters or actions that were not explicitly described. Do not replace characters even if this would improve flow. Combining actions into a single sentence is OK as long as you do not alter the original information. {infilling_text_type}

Make it a short, yet an interesting story to read. Make the text natural to read as well as each character's speech, so try to avoid e.g. starting all the sentences the same way. The story needs to follow common sense, e.g. do not magically change an object's location without mentioning it. Do not include any notes, comments, parentheses, or any other form of extra text that would not belong in a story.

As a warning, take into account that when someone tells someone privately they might not be in the same location, e.g. they might be sending a text message or making a phone call; they might also be in the same location, in that case they could also communicate through a gesture, a whisper, etc. Do not assume a person is in the same room if it has not been made explicit before. Also, if someone was spying, or if they were distracted and did not listen or saw something happen, do not forget to include it! Remember that in this case, it should be clear that the distraction or spying applies only to the action mentioned and they go back to normal after the action is finished. Avoid making multiple copies of the same object.

WHO ARE THE CHARACTERS: {people_with_personas}

NEW ACTION OR INFORMATION TO INCLUDE: {new_information}

CURRENT SHORT STORY: {story_context}

STORY CONTINUATION:"""

    PROMPT_STORY_INFILLING_WITH_GOALS = """Continue the story {story_length}, clearly conveying the action or information below without altering it. Do not contradict any prior information. Avoid repeating the information verbatim, instead naturally (and possibly implicitly, but still unambiguously) conveying the meaning. Do not add characters or actions that were not explicitly described. Do not replace characters even if this would improve flow. Combining actions into a single sentence is OK as long as you do not alter the original information. {infilling_text_type}

Make it a short, yet an interesting story to read. Make the text exciting to read as well as each character's speech, so try to avoid e.g. starting all the sentences the same way. The story needs to follow common sense, e.g. do not magically change an object's location without mentioning it. Do not include any notes, comments, parentheses, or any other form of extra text that would not belong in a story. Feel free to hint or describe characters' goals and motivations for performing the actions if it would make the story flow better.

As a warning, take into account that when someone tells someone privately they might not be in the same location, e.g. they might be sending a text message or making a phone call; they might also be in the same location, in that case they could also communicate through a gesture, a whisper, etc. Do not assume a person is in the same room if it has not been made explicit before. Also, if someone was spying, or if they were distracted and did not listen or saw something happen, do not forget to include it! Remember that in this case, it should be clear that the distraction or spying applies only to the action mentioned and they go back to normal after the action is finished. Avoid making multiple copies of the same object.

WHO ARE THE CHARACTERS: {people_with_personas}

WHAT ARE THEIR GOALS: {optional_characters_goals}

NEW ACTION OR INFORMATION TO INCLUDE: {new_information}

CURRENT SHORT STORY: {story_context}

Continue the story below. Do not include any notes, comments, parentheses, or any other form of extra text that would not belong in a story.

STORY CONTINUATION:"""
    PROMPT_STORY_INFILLING_SIMULTANEOUS = """Continue the story {story_length}, clearly conveying the action or information below without altering it. Do not contradict any prior information. Avoid repeating the information verbatim, instead naturally (and possibly implicitly, but still unambiguously) conveying the meaning. Do not add characters or actions that were not explicitly described. Do not replace characters even if this would improve flow. Combining actions into a single sentence is OK as long as you do not alter the original information. {infilling_text_type}

Make it a short, yet an interesting story to read. Make the text natural to read as well as each character's speech, so try to avoid e.g. starting all the sentences the same way. The story needs to follow common sense, e.g. do not magically change an object's location without mentioning it. Do not include any notes, comments, parentheses, or any other form of extra text that would not belong in a story.

As a warning, take into account that when someone tells someone privately they might not be in the same location, e.g. they might be sending a text message or making a phone call; they might also be in the same location, in that case they could also communicate through a gesture, a whisper, etc. Do not assume a person is in the same room if it has not been made explicit before. Also, if someone was spying, or if they were distracted and did not listen or saw something happen, do not forget to include it! Remember that in this case, it should be clear that the distraction or spying applies only to the action mentioned and they go back to normal after the action is finished. Avoid making multiple copies of the same object.

Give {num_tries_completions} responses, ensuring to give {num_tries_completions} different phrasings of continuing the story conveying the action. Use very different wordings and sentence structures!

WHO ARE THE CHARACTERS: {people_with_personas}

NEW ACTION OR INFORMATION TO INCLUDE: {new_information}

CURRENT SHORT STORY: {story_context}

Follow the format and do not include any other text. Do not include any text before the list. Do not include any notes, comments, parentheses, or any other form of extra text that would not belong in a story. {infilling_text_type}

STORY CONTINUATION: <fill>

STORY CONTINUATION: <fill>"""

    PROMPT_STORY_INFILLING_WITH_GOALS_SIMULTANEOUS = """Continue the story {story_length}, clearly conveying the action or information below without altering it. Do not contradict any prior information. Avoid repeating the information verbatim, instead naturally (and possibly implicitly, but still unambiguously) conveying the meaning. Do not add characters or actions that were not explicitly described. Do not replace characters even if this would improve flow. Combining actions into a single sentence is OK as long as you do not alter the original information. {infilling_text_type}

Make it a short, yet an interesting story to read. Make the text exciting to read as well as each character's speech, so try to avoid e.g. starting all the sentences the same way. The story needs to follow common sense, e.g. do not magically change an object's location without mentioning it. Do not include any notes, comments, parentheses, or any other form of extra text that would not belong in a story. Feel free to hint or describe characters' goals and motivations for performing the actions if it would make the story flow better.

As a warning, take into account that when someone tells someone privately they might not be in the same location, e.g. they might be sending a text message or making a phone call; they might also be in the same location, in that case they could also communicate through a gesture, a whisper, etc. Do not assume a person is in the same room if it has not been made explicit before. Also, if someone was spying, or if they were distracted and did not listen or saw something happen, do not forget to include it! Remember that in this case, it should be clear that the distraction or spying applies only to the action mentioned and they go back to normal after the action is finished. Avoid making multiple copies of the same object.

Give {num_tries_completions} responses, ensuring to give {num_tries_completions} different phrasings of continuing the story conveying the action. Use very different wordings and sentence structures, but avoid changing object or room names!

WHO ARE THE CHARACTERS: {people_with_personas}

WHAT ARE THEIR GOALS: {optional_characters_goals}

NEW ACTION OR INFORMATION TO INCLUDE: {new_information}

CURRENT SHORT STORY: {story_context}

Follow the format and do not include any other text. Do not include any text before the list. Do not enumerate. Continue the story {story_length}. Avoid repeating the information verbatim, instead naturally (and possibly implicitly, but still unambiguously) conveying the meaning.

STORY CONTINUATION: <fill>

STORY CONTINUATION: <fill>"""

    def __init__(self, model_call_handler, **kwargs):
        self.model_call_handler = model_call_handler
        self.evaluator = ModelAnswerEvaluator(
            model_call_handler
        )  # this is just to evaluate the generated story here and have all answers in the logs
        self.quality_verifier = StoryInfillingQualityVerifier(
            model_call_handler, **kwargs
        )

        self.SAMPLE_CHARACTERS_GOALS = kwargs.get(
            "sample_characters_goals", True
        )  # include character goals to story prompt
        self.SAMPLE_INITIAL_STORY_CONTEXT = kwargs.get(
            "sample_initial_story_context", True
        )  # sample context before talking about actions
        self.num_tries_completions = kwargs.get("num_tries_completions", 5)
        self.sample_all_next_step_completions_simultaneously = kwargs.get(
            "sample_all_next_step_completions_simultaneously", False
        )  # this adds diversity
        self.infilling_grouping_type = kwargs.get(
            "infilling_grouping_type", "dst1"
        )  # this groups actions for extra naturalness
        self.experiment_variations = kwargs.get("experiment_variations", [])

        self.log_infilling = []

    def _sample_characters_goals(self, story_script, characters_description):
        if not self.SAMPLE_CHARACTERS_GOALS:
            return None

        full_prompt = self.PROMPT_CHARACTER_GOALS.format(
            story_script=story_script, characters_description=characters_description
        )
        characters_goals, tokens_used = self.model_call_handler.call_model(
            self.PROMPT_CHARACTER_GOALS.format(
                story_script=story_script, characters_description=characters_description
            ),
            max_tokens=300,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            system_prompt=self.SYSTEM_PROMPT_FOR_WRITER,
        )
        characters_goals = characters_goals.strip()
        characters_goals = (
            characters_goals[len("CHARACTERS GOALS:") :]
            if characters_goals.startswith("CHARACTERS GOALS:")
            else characters_goals
        )
        characters_goals = "\n" + characters_goals.strip()
        self.log_infilling.append(
            {
                "characters_goals": characters_goals,
                "full_prompt": full_prompt,
            }
        )
        return characters_goals

    def _sample_initial_infilling_context(self, story_script, characters_description):
        if not self.SAMPLE_INITIAL_STORY_CONTEXT:
            return None
        full_prompt = self.PROMPT_INITIAL_STORY_CONTEXT.format(
            story_script=story_script, characters_description=characters_description
        )
        story_context, tokens_used = self.model_call_handler.call_model(
            self.PROMPT_INITIAL_STORY_CONTEXT.format(
                story_script=story_script, characters_description=characters_description
            ),
            max_tokens=300,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            system_prompt=self.SYSTEM_PROMPT_FOR_WRITER,
        )
        self.log_infilling.append(
            {
                "initial_story_context": story_context,
                "full_prompt": full_prompt,
            }
        )
        return story_context

    def _select_prompt_variations(self, relevant_personas, cur_story_context):
        people_with_personas = " ".join(relevant_personas)
        story_context = "Empty" if cur_story_context is None else cur_story_context

        if "fantom_like_data" in self.experiment_variations:
            prompt_story_length = "with at least five sentences"
            prompt_infilling_text_type = "Make the new text conversational, using direct quotes to convey the words spoken by a character."
        else:
            prompt_story_length = self.PROMPT_STORY_INFILLING_VERSIONS["story_length"][
                random.randint(
                    0, len(self.PROMPT_STORY_INFILLING_VERSIONS["story_length"]) - 1
                )
            ]
            prompt_infilling_text_type = self.PROMPT_STORY_INFILLING_VERSIONS[
                "infilling_text_type"
            ][
                random.randint(
                    0,
                    len(self.PROMPT_STORY_INFILLING_VERSIONS["infilling_text_type"])
                    - 1,
                )
            ]
        return (
            prompt_story_length,
            prompt_infilling_text_type,
            people_with_personas,
            story_context,
        )

    def _single_sample_next_story_steps_for_infilling(
        self,
        relevant_personas,
        new_information,
        cur_story_context,
        characters_goals=None,
    ):
        (
            prompt_story_length,
            prompt_infilling_text_type,
            people_with_personas,
            story_context,
        ) = self._select_prompt_variations(relevant_personas, cur_story_context)
        if self.SAMPLE_CHARACTERS_GOALS:
            prompt = self.PROMPT_STORY_INFILLING_WITH_GOALS.format(
                people_with_personas=people_with_personas,
                new_information=new_information,
                story_context=story_context,
                story_length=prompt_story_length,
                infilling_text_type=prompt_infilling_text_type,
                optional_characters_goals=characters_goals,
            )
        else:
            prompt = self.PROMPT_STORY_INFILLING_WITHOUT_GOALS.format(
                people_with_personas=people_with_personas,
                new_information=new_information,
                story_context=story_context,
                story_length=prompt_story_length,
                infilling_text_type=prompt_infilling_text_type,
            )
        next_story_step, tokens_used = self.model_call_handler.call_model(
            prompt,
            max_tokens=300,
            top_p=1.0,
            temperature=1.0,
            repetition_penalty=1.0,
            system_prompt=self.SYSTEM_PROMPT_FOR_WRITER,
        )
        log_info_regardless_of_retry_idx = {
            "story_context": story_context,
            "story_length": prompt_story_length,
            "new_information": new_information,
            "people_with_personas": people_with_personas,
            "infilling_text_type": prompt_infilling_text_type,
        }
        return next_story_step, tokens_used, log_info_regardless_of_retry_idx

    def _simultaneously_sample_next_story_steps_for_infilling(
        self,
        relevant_personas,
        new_information,
        cur_story_context,
        characters_goals=None,
    ):
        # simultaneously get num_tries_completions generations. We try many times in case the model yielded less generations

        (
            prompt_story_length,
            prompt_infilling_text_type,
            people_with_personas,
            story_context,
        ) = self._select_prompt_variations(relevant_personas, cur_story_context)
        if self.SAMPLE_CHARACTERS_GOALS:
            prompt = self.PROMPT_STORY_INFILLING_WITH_GOALS_SIMULTANEOUS.format(
                people_with_personas=people_with_personas,
                new_information=new_information,
                story_context=story_context,
                story_length=prompt_story_length,
                infilling_text_type=prompt_infilling_text_type,
                optional_characters_goals=characters_goals,
                num_tries_completions=self.num_tries_completions,
            )
        else:
            prompt = self.PROMPT_STORY_INFILLING_SIMULTANEOUS.format(
                people_with_personas=people_with_personas,
                new_information=new_information,
                story_context=story_context,
                story_length=prompt_story_length,
                infilling_text_type=prompt_infilling_text_type,
                num_tries_completions=self.num_tries_completions,
            )

        log_info_regardless_of_retry_idx = {
            "prompt_story_length": prompt_story_length,
            "prompt_infilling_text_type": prompt_infilling_text_type,
            "people_with_personas": people_with_personas,
            "story_context": story_context,
            "new_information": new_information,
            "num_tries_completions": self.num_tries_completions,
        }

        all_tokens_used = 0
        next_story_step_list = []
        for num_try_idx in range(self.num_tries_completions):
            next_story_step, tokens_used = self.model_call_handler.call_model(
                prompt,
                max_tokens=300 * self.num_tries_completions,
                top_p=1.0,
                temperature=1.0,
                repetition_penalty=1.0,
                system_prompt=self.SYSTEM_PROMPT_FOR_WRITER,
            )
            all_tokens_used += tokens_used
            next_story_step_list = [e.strip() for e in next_story_step.split("\n\n")]
            next_story_step_list = [
                e[len("STORY CONTINUATION:") :].strip()
                if e.startswith("STORY CONTINUATION:")
                else e
                for e in next_story_step_list
                if e
            ]

            tmp_log_info = copy.copy(log_info_regardless_of_retry_idx)
            tmp_log_info.update(
                {
                    "num_try_idx": num_try_idx,
                    "next_story_step_list": next_story_step_list,
                }
            )
            self.log_infilling.append(tmp_log_info)

            next_story_step_list = [
                e for e in next_story_step_list if "\n" not in e
            ]  # do not want a sentence infilling that has multiple paragraphs
            if any(["story continuation" in e.lower() for e in next_story_step_list]):
                next_story_step_list = []  # there might be some parsing error
            if any(["1." in e.lower() for e in next_story_step_list]):
                next_story_step_list = []  # there might be some parsing error
            if len(next_story_step_list) >= self.num_tries_completions:
                break  # we have enough generations!
        random.shuffle(next_story_step_list)
        return next_story_step_list, all_tokens_used, log_info_regardless_of_retry_idx

    def _group_actions_for_infilling(
        self, story_script, story_script_tuple_representation
    ):
        """
        Two statements can be grouped if they only differ on one thing.

        If there is an asymmetric statement, it gets grouped with the one before.
        """
        if self.infilling_grouping_type == "dst1":
            groups = [[story_script[0]]]
            groups_indices = [[0]]
            ctr = 0
            for i in range(1, len(story_script_tuple_representation)):
                prev = story_script_tuple_representation[i - 1]
                cur = story_script_tuple_representation[i]
                if (
                    len(prev) == len(cur)
                    and prev[0] == cur[0]
                    and sum(1 for e1, e2 in zip(prev, cur) if e1 == e2) >= len(prev) - 1
                ):
                    groups[-1].append(story_script[i])
                    ctr += 1
                    groups_indices[-1].append(ctr)
                elif (
                    "peeking_distracted" in cur[0]
                ):  # this means that it is a modifier of the previous action and should be added
                    groups[-1].append(story_script[i])
                else:
                    groups.append([story_script[i]])
                    ctr += 1
                    groups_indices.append([ctr])
            return groups, groups_indices
        elif self.infilling_grouping_type == "none":
            assert not any(
                "peeking_distracted" in cur[0]
                for cur in story_script_tuple_representation
            ), "Did not fix this for asymmetry. Currently not in use."
            return [[e] for e in story_script], [[i] for i in range(len(story_script))]
        else:
            assert False, "Unsuported infilling type"

    def main(self, belief_tracker, function_list, use_custom_names_dict):
        assert use_custom_names_dict is not None, "use_custom_names_dict is None"

        story_script = " ".join(belief_tracker.story_script)
        characters_description_enumeration = "\n".join(
            [f"* {e}" for e in use_custom_names_dict["people_with_personas"]]
        )

        characters_goals = self._sample_characters_goals(
            story_script, characters_description_enumeration
        )
        story_context = self._sample_initial_infilling_context(
            story_script, characters_description_enumeration
        )

        (
            grouped_story_script,
            grouped_story_script_indices,
        ) = self._group_actions_for_infilling(
            belief_tracker.story_script,
            belief_tracker.story_script_tuple_representation,
        )
        grouped_story_script = [" ".join(e) for e in grouped_story_script]

        cur_belief_tracker = FullBeliefTracker()
        df_list = []
        for new_information, new_information_indices in zip(
            grouped_story_script, grouped_story_script_indices
        ):
            new_action_function = [function_list[i] for i in new_information_indices]
            relevant_personas = [
                p
                for p_n, p in zip(
                    use_custom_names_dict["people_names"],
                    use_custom_names_dict["people_with_personas"],
                )
                if p_n in new_information
            ]

            if self.sample_all_next_step_completions_simultaneously:
                (
                    next_story_step_list,
                    tokens_used,
                    log_info_regardless_of_retry_idx,
                ) = self._simultaneously_sample_next_story_steps_for_infilling(
                    relevant_personas,
                    new_information,
                    story_context,
                    characters_goals,
                )

            next_story_step_accepted = True
            for retry_idx in range(self.num_tries_completions):
                if self.sample_all_next_step_completions_simultaneously:
                    if retry_idx >= len(next_story_step_list):
                        break
                    next_story_step = next_story_step_list[retry_idx]
                else:
                    (
                        next_story_step,
                        tokens_used,
                        log_info_regardless_of_retry_idx,
                    ) = self._single_sample_next_story_steps_for_infilling(
                        relevant_personas,
                        new_information,
                        story_context,
                        characters_goals,
                    )

                next_story_context = (
                    next_story_step
                    if story_context is None
                    else story_context + " " + next_story_step
                )

                next_belief_tracker = copy.deepcopy(cur_belief_tracker)
                for fn in new_action_function:
                    output_flag = fn(next_belief_tracker)
                assert " ".join(next_belief_tracker.story_script).endswith(
                    new_information
                ), f"{next_belief_tracker.story_script} does not end with {new_information}"

                (
                    next_story_step_accepted,
                    tmp_for_log_infilling_clean,
                ) = self.quality_verifier.main(
                    cur_belief_tracker,
                    next_belief_tracker,
                    next_story_context,
                    next_story_step,
                    new_information,
                )

                tmp_for_log_infilling = copy.copy(tmp_for_log_infilling_clean)
                tmp_for_log_infilling.update(log_info_regardless_of_retry_idx)
                tmp_for_log_infilling.update(
                    {
                        "num_tries_completions": self.num_tries_completions,
                        "retry_idx": retry_idx,
                    }
                )
                self.log_infilling.append(tmp_for_log_infilling)
                tmp_log = {
                    "new_information": new_information,
                    "new_information_indices": new_information_indices,
                    "next_story_step": next_story_step,
                    "infilling_grouping_type": self.infilling_grouping_type,
                    "accumulated_infilled_story": next_story_context,
                    "raw_story": belief_tracker.story_script,
                    "next_story_step_accepted": next_story_step_accepted,
                    "time": time.time(),
                    "characters_goals": characters_goals,
                }
                tmp_log.update(tmp_for_log_infilling_clean)
                tmp_log.update(log_info_regardless_of_retry_idx)
                df_list.append(tmp_log)

                if next_story_step_accepted:  # update the accumulated variables
                    cur_belief_tracker = next_belief_tracker
                    story_context = next_story_context
                    break
                else:
                    continue
            if not next_story_step_accepted:
                break  # we could not find a good infilling for the last action

        if next_story_step_accepted:
            print(f"****** STORY SUCCEEDED ******")
            assert (
                belief_tracker.story_script == cur_belief_tracker.story_script
            ), f"{belief_tracker.story_script} != {cur_belief_tracker.story_script}"
        else:
            print(f"****** STORY FAILED ******")
            responses_raw_script = []
            responses_infilled_script = []

        # evaluating infilled stories with first order questions to assess the difficulty differences
        responses_raw_script_evaluations, tokens_used = self.evaluator.evaluate_story(
            belief_tracker, story=None, nth=1
        )
        (
            responses_infilled_script_evaluations,
            tokens_used,
        ) = self.evaluator.evaluate_story(belief_tracker, story=story_context, nth=1)

        return (
            df_list,
            (belief_tracker, function_list),
            story_context,
            next_story_step_accepted,
            responses_raw_script_evaluations,
            responses_infilled_script_evaluations,
            self.log_infilling,
        )


class FullStoryStructureInfiller:
    def __init__(self, model_call_handler, **kwargs):
        self.model_call_handler = model_call_handler

        self.SAMPLE_CHARACTERS_GOALS = kwargs.get("sample_characters_goals", True)
        self.SAMPLE_INITIAL_STORY_CONTEXT = kwargs.get(
            "sample_initial_story_context", True
        )
        self.num_tries_completions = kwargs.get("num_tries_completions", 5)
        self.sample_all_next_step_completions_simultaneously = kwargs.get(
            "sample_all_next_step_completions_simultaneously", True
        )
        self.infilling_grouping_type = kwargs.get("infilling_grouping_type", "dst1")
        self.experiment_variations = kwargs.get("experiment_variations", [])

        self.infiller = StoryStructureInfiller(
            model_call_handler,
            sample_characters_goals=self.SAMPLE_CHARACTERS_GOALS,
            sample_initial_story_context=self.SAMPLE_INITIAL_STORY_CONTEXT,
            num_tries_completions=self.num_tries_completions,
            sample_all_next_step_completions_simultaneously=self.sample_all_next_step_completions_simultaneously,
            infilling_grouping_type=self.infilling_grouping_type,
            experiment_variations=self.experiment_variations,
        )

    def _get_base_directory(self, logs_directory):
        if self.SAMPLE_CHARACTERS_GOALS and self.SAMPLE_INITIAL_STORY_CONTEXT:
            base_directory = "infilled_with_llm_judge_w_goals_and_initial"
        elif not self.SAMPLE_CHARACTERS_GOALS and not self.SAMPLE_INITIAL_STORY_CONTEXT:
            base_directory = "infilled_with_llm_judge"
        else:
            assert False
        base_directory += logs_directory[len("logs") :]
        return base_directory

    def _build_output_filename(self, filename, len_logs):
        filename = filename.split("/")[-1]
        filename = filename[: -len(".pkl")]
        prefix_filename_to_write = f"{filename}-{self.infilling_grouping_type}"
        if self.sample_all_next_step_completions_simultaneously:
            prefix_filename_to_write += "_smlt"
        if "fantom_like_data" in self.experiment_variations:
            prefix_filename_to_write += "_fanlike"
        prefix_filename_to_write += "_v3"
        full_filename_to_write_pkl = f"{prefix_filename_to_write}_n_{len_logs}.pkl"
        return prefix_filename_to_write, full_filename_to_write_pkl

    def main(
        self,
        logs_directory,
        NUM_STORIES,
        NUM_PARALLEL,
        parallel_idx=None,
        only_verified_cases=True,
    ):
        base_directory = self._get_base_directory(logs_directory)
        os.makedirs(base_directory, exist_ok=True)
        filenames_to_solve = [
            os.path.join(logs_directory, f)
            for f in os.listdir(logs_directory)
            if f.startswith("search")
            and f.endswith(".pkl")
            and self.model_call_handler.model_shortname in f
            and "dill" in f
            and "weight-goal4" in f
            and f"_n_{NUM_STORIES}_" in f
        ]
        if "fantom_like_data" in self.experiment_variations:
            filenames_to_solve = [
                f
                for f in filenames_to_solve
                if "+fantom+" in f or "_fantom-" in f or "_all_" in f
            ]

        random.seed(1)
        random.shuffle(filenames_to_solve)

        for i, filename in enumerate(filenames_to_solve):
            if (
                parallel_idx is not None
                and i % NUM_PARALLEL != parallel_idx % NUM_PARALLEL
            ):  # DIY parallelization hahaha
                continue
            print("f", filename)

            # 0. Select stories to infill (the final stories, which are the only ones with a belief tracker)
            try:
                with open(filename, "rb") as f:
                    all_logs = dill.load(f)
            except Exception as e:
                print("ISSUE WITH FILE", filename)
                print(e)
                continue

            if only_verified_cases:
                all_logs = [log for log in all_logs if log["is_final"] is True]
            else:
                all_logs = [
                    log
                    for log in all_logs
                    if log["is_final_without_verification"] is True
                ]
            if len(all_logs) > NUM_STORIES:
                x = [(elem["a_star_g"], i) for i, elem in enumerate(all_logs)]
                all_logs = [all_logs[i] for _, i in sorted(x)][:NUM_STORIES]

            (
                prefix_filename_to_write,
                full_filename_to_write_pkl,
            ) = self._build_output_filename(filename, len(all_logs))
            if os.path.exists(full_filename_to_write_pkl):
                print("0-skipping")
                continue

            big_pickle_data_list = []
            for log_idx, log in enumerate(all_logs):
                (
                    df_list,
                    (belief_tracker, function_list),
                    story_context,
                    next_story_step_accepted,
                    responses_raw_script_evaluations,
                    responses_infilled_script_evaluations,
                    self.log_infilling,
                ) = self.infiller.main(
                    log["belief_tracker"],
                    log["function_list"],
                    log["use_custom_names_dict"],
                )
                pd.DataFrame(df_list).to_csv(
                    os.path.join(
                        base_directory, f"{prefix_filename_to_write}_idx-{log_idx}.csv"
                    )
                )
                big_pickle_data_list.append(
                    [
                        df_list,
                        (belief_tracker, function_list),
                        story_context,
                        next_story_step_accepted,
                        responses_raw_script_evaluations,
                        responses_infilled_script_evaluations,
                        self.log_infilling,
                    ]
                )
            with open(
                os.path.join(base_directory, full_filename_to_write_pkl), "wb"
            ) as f:
                dill.dump(big_pickle_data_list, f)


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
    parser.add_argument("--generate_fantom_like_data", action="store_true")
    parser.add_argument("--num_stories_total", type=int, default=10)

    NUM_PARALLEL = 8
    parser.add_argument("--i", type=int, default=None, help="DIY parallelization :)")
    args = parser.parse_args()

    # Prerequisite: find challenging story structures and dump them in logs_debug_nov7_fixed_restrictions
    model_call_handler = ModelCallHandler(args.model_name, args.model_access_method)

    if args.generate_fantom_like_data:
        full_infiller = FullStoryStructureInfiller(
            model_call_handler,
            sample_all_next_step_completions_simultaneously=True,
            experiment_variations=["fantom_like_data"],
        )
    else:
        full_infiller = FullStoryStructureInfiller(
            model_call_handler, sample_all_next_step_completions_simultaneously=True
        )
    full_infiller.main(
        logs_directory="logs_debug_dec4_fixed_restrictions",
        NUM_STORIES=args.num_stories_total,
        NUM_PARALLEL=NUM_PARALLEL,
        parallel_idx=args.i,
    )
