# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import re
import time
import json
import openai
import pandas as pd
from belief_tracker import QuestionGenerator


def _get_model_shortname(model_name):
    engine_shortname = model_name.split("/")[-1]
    engine_shortname = (
        engine_shortname[: -len("-Turbo")]
        if engine_shortname.endswith("-Turbo")
        else engine_shortname
    )
    engine_shortname = (
        engine_shortname[len("Meta-") :]
        if engine_shortname.startswith("Meta-")
        else engine_shortname
    )
    return engine_shortname


class ModelCallHandler:
    def __init__(self, model_name, model_access_method, lora_model_path=None):
        self.model_name = model_name
        self.model_shortname = _get_model_shortname(model_name)
        self.model_access_method = model_access_method
        assert model_access_method in [
            "vllm-python",
            "vllm-python-lora",
            "vllm-api",
            "openai-azure-api",
            "openai-api",
            "huggingface-4bit",
            "huggingface",
        ]
        if model_access_method == "vllm-python":
            self.load_vllm_python()
        if model_access_method == "vllm-python-lora":
            self.load_vllm_python_with_lora_request(
                lora_model_path
            )  # model_name = base model ; lora_model_path = fine-tuned model with LORA

        if model_access_method == "vllm-api":
            assert ModelCallHandler.is_vllm_api_available()
            self.load_vllm_api_client()
        if model_access_method == "openai-azure-api":
            self.load_openai_azure_api_client()
        if model_access_method == "openai-api":
            self.load_openai_api_client()

        if model_access_method == "huggingface-4bit":
            self.load_huggingface_model_4bit()
        assert (
            model_access_method != "huggingface"
        ), f"model_access_method={model_access_method} not implemented."

    @staticmethod
    def is_vllm_api_available():
        import requests

        try:
            output = requests.get("http://0.0.0.0:8000")
            assert (
                output.status_code == 404
            ), "I do not know a context in which there would be another status code output"
            return True
        except Exception as e:
            print(e)
            return False

    def load_vllm_python(self):
        import torch
        from vllm import LLM
        from transformers import AutoTokenizer

        self.vllm_llm = LLM(
            model=self.model_name, tensor_parallel_size=torch.cuda.device_count()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lora_request = None

    def load_vllm_python_with_lora_request(self, sql_lora_path):
        import torch
        from vllm import LLM
        from transformers import AutoTokenizer
        from vllm.lora.request import LoRARequest

        base_model_name = self.model_name
        self.vllm_llm = LLM(
            model=base_model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            enable_lora=True,
            max_num_batched_tokens=4000,
            max_model_len=4000,
            max_lora_rank=64,
        )
        self.lora_request = LoRARequest("sql_adapter", 1, sql_lora_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_vllm_api_client(self):
        from openai import OpenAI

        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    def load_openai_azure_api_client(self):
        from openai import AzureOpenAI

        OPENAI_AZURE_API_KEY = os.environ.get("OPENAI_AZURE_API_KEY")
        OPENAI_AZURE_ENDPOINT = os.environ.get("OPENAI_AZURE_ENDPOINT")
        self.client = AzureOpenAI(
            api_version="2024-06-01",
            api_key=OPENAI_AZURE_API_KEY,
            azure_endpoint=OPENAI_AZURE_ENDPOINT,
        )

    def load_openai_api_client(self):
        from openai import OpenAI

        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        OPENAI_API_ORGANIZATION = os.environ.get("OPENAI_API_ORGANIZATION")
        OPENAI_API_PROJECT = os.environ.get("OPENAI_API_PROJECT")
        self.client = OpenAI(
            organization=OPENAI_API_ORGANIZATION,
            project=OPENAI_API_PROJECT,
        )

    def load_huggingface_model_4bit(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=nf4_config,
            attn_implementation="flash_attention_2",
            cache_dir=cache_dir,
        )

    def call_vllm_python_model(self, message_list, top_p, temperature, **kwargs):
        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, **kwargs)
        input_text = self.tokenizer.apply_chat_template(
            message_list, tokenize=False, add_generation_prompt=True
        )
        if self.lora_request:
            output = self.vllm_llm.generate(
                input_text, sampling_params, lora_request=self.lora_request
            )[0]
        else:
            output = self.vllm_llm.generate(input_text, sampling_params)[0]
        return output.outputs[0].text, 0

    def call_openai_api_retry_loop(
        self, message_list, max_tokens=10, top_p=1.0, temperature=1.0, **kwargs
    ):
        try:
            sample_output = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_list,
                max_tokens=max_tokens,
                top_p=top_p,
                temperature=temperature,
                extra_body=kwargs,  # VLLM-supported params that OpenAI does not support
            )
            generation = sample_output.choices[0].message.content
            tokens_used = sample_output.usage.total_tokens
        except (
            openai.RateLimitError,
            openai.APIError,
            openai.InternalServerError,
            openai.UnprocessableEntityError,
            openai.APITimeoutError,
        ) as e:
            print(e)
            time.sleep(0.50)
            return self.call_openai_api_retry_loop(
                message_list, max_tokens, top_p, temperature, **kwargs
            )
        return generation, tokens_used

    def call_huggingface_local_model(
        self, message_list, max_tokens, top_p, temperature
    ):
        input_ids = self.tokenizer.apply_chat_template(
            message_list, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        if temperature > 0:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=False,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = outputs[0][input_ids.shape[-1] :]
        decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
        return decoded_response, 0

    def call_model(
        self,
        prompt,
        max_tokens=50,
        top_p=1.0,
        temperature=1.0,
        system_prompt=None,
        **kwargs,
    ):

        # 1. build message list for a chat model
        if isinstance(prompt, list):
            message_list = prompt
        elif isinstance(prompt, str):
            message_list = [{"role": "user", "content": prompt}]
        else:
            assert False, "prompt needs to be list or str"

        # 2. include system prompt (except for o1-preview, that does not allow it)
        if system_prompt is None:
            message_list = [
                {"role": "system", "content": "You are a helpful assistant."}
            ] + message_list
        elif not self.model_name.startswith("o1-"):
            message_list = [{"role": "system", "content": system_prompt}] + message_list
        assert (
            sum([1 for m in message_list if m["role"] == "system"]) < 2
        ), f"There should not be two system messages: {str(message_list)}"

        if self.model_access_method.startswith("huggingface"):
            return self.call_huggingface_local_model(
                message_list, max_tokens, top_p, temperature
            )

        if self.model_access_method.startswith("vllm-python"):
            return self.call_vllm_python_model(
                message_list, top_p, temperature, **kwargs
            )

        if (
            self.model_access_method.startswith("openai")
            or self.model_access_method == "vllm-api"
        ):
            return self.call_openai_api_retry_loop(
                message_list, max_tokens, top_p, temperature, **kwargs
            )

        assert False, f"Unsupported model access method {self.model_access_method}"

    def verify_using_llm_as_judge(
        self, parsed_fields, llm_judge_prompt, llm_judge_prompt_answers
    ):
        model_response, tokens_used = self.call_model(
            llm_judge_prompt.format(**parsed_fields),
            max_tokens=800,
            top_p=1.0,
            temperature=0.0,
        )
        answers = [e.split(": ")[1] for e in model_response.split("\n") if ":" in e]
        # print(answers, llm_judge_prompt_answers)
        return {
            "is_equivalent": len(answers) == len(llm_judge_prompt_answers)
            and all(a == b for a, b in zip(answers, llm_judge_prompt_answers)),
            "model_response": model_response,
            "parsed_model_responses": answers,
        }


class ModelAnswerEvaluator:
    FULL_PROMPT = "{story}\n{question} Short Answer."
    CHECK_ANSWERS_PROMPT = """Given a sentence, answer if it mentions each topic or state mentioned.

SENTENCE: "{sentence}"

Does it mention the person knows about each of the following topics?
{desiderata}

Include the itemized questions and do not add any extra text. Answer all items following the format.
"""

    def __init__(self, model_call_handler):
        self.model_call_handler = model_call_handler

    def verify_answer_to_question(self, answer, model_response, num_tries=1):
        if isinstance(answer, str):
            return answer in model_response
        elif isinstance(answer, tuple):
            correct_descriptions, incorrect_descriptions = answer
            parsed_fields = {
                "sentence": model_response,
                "desiderata": "\n".join(
                    [
                        f"* {e}: (yes/no)"
                        for e in correct_descriptions + incorrect_descriptions
                    ]
                ),
            }
            llm_judge_prompt_answers = ["yes"] * len(correct_descriptions) + [
                "no"
            ] * len(incorrect_descriptions)

            output_list = []
            for idx_try in range(num_tries):
                if idx_try > 0:
                    shuffled_indices = list(range(len(llm_judge_prompt_answers)))
                    random.shuffle(shuffled_indices)
                    llm_judge_prompt_answers = [
                        llm_judge_prompt_answers[i] for i in shuffled_indices
                    ]
                    parsed_fields["desiderata"] = [
                        parsed_fields["desiderata"][i] for i in shuffled_indices
                    ]

                output = self.model_call_handler.verify_using_llm_as_judge(
                    parsed_fields,
                    ModelEvaluator.CHECK_ANSWERS_PROMPT,
                    llm_judge_prompt_answers,
                )
                output_list.append(output)
                if any(output["is_equivalent"] for output in output_list):
                    return True  # early stopping :)
            return any(output["is_equivalent"] for output in output_list)
        else:
            assert False, "Not Implemented"

    def evaluate_question(self, story, question, expected_answer, num_tries=1):
        model_response, tokens_used = self.model_call_handler.call_model(
            self.FULL_PROMPT.format(story=story, question=question),
            max_tokens=40,
            top_p=1.0,
            temperature=0.0,
        )
        model_response = model_response.lower()

        response = {
            "story": story,
            "model_response": model_response,
            "question": question,
            "answer": expected_answer,
            "is_correct": self.verify_answer_to_question(
                expected_answer, model_response, num_tries=num_tries
            ),
        }
        return response, tokens_used

    def evaluate_story(
        self,
        belief_tracker,
        story=None,
        nth=1,
        num_tries=1,
        keep_only_interesting_qs=False,
        heuristic_always_return_correct=False,
    ):
        all_tokens_used = 0
        responses = []

        story = " ".join(belief_tracker.story_script) if story is None else story
        first_order_questions = QuestionGenerator(belief_tracker).main(
            nth_reasoning_order=nth
        )
        for question, answer, _, metadata in first_order_questions:
            if keep_only_interesting_qs and metadata[0] is False:
                continue
            if heuristic_always_return_correct:
                response = {"is_correct": True, "heuristic_always_return_correct": True}
                tokens_used = 0
            else:
                response, tokens_used = self.evaluate_question(
                    story, question, answer, num_tries=1
                )
            responses.append(response)
            all_tokens_used += tokens_used
        return responses, all_tokens_used

    def evaluate_file(self, df, output_filename_evaluated):
        if os.path.exists(output_filename_evaluated):
            return pd.read_csv(output_filename_evaluated, index_col=0)
        model_response, is_answer_correct = [], []
        start_time = time.time()
        ctr = 0
        for i, row in df.iterrows():
            oupu, _ = self.evaluate_question(
                row["story"], row["question"], row["expected_answer"], num_tries=1
            )
            model_response.append(oupu["model_response"])
            is_answer_correct.append(oupu["is_correct"])

            if ctr % 100 == 0:
                print("Done", ctr, len(df), time.time() - start_time)
            ctr += 1
        df["model_response"] = model_response
        df["is_answer_correct"] = is_answer_correct
        df.to_csv(output_filename_evaluated)
        return df


"""
******** General utils ********
"""
# used in StoryContextGenerator & ObjectStateUpdatesGenerator
def _dump_dict_list_to_jsonl(dict_list, filepath):
    with open(filepath, "w") as f:
        for item in dict_list:
            json.dump(item, f)
            f.write("\n")


"""
******** Utils on model output parsing ********
"""

# used in StoryContextGenerator
parser_numbered_list = re.compile(r"\d+\.\s")  # Matches the pattern of "number. "


def _parse_numbered_list(text):
    return [elem.strip("\n ") for elem in parser_numbered_list.split(text)[1:]]


# used in StoryContextGenerator
def _parse_answer_with_separators(text, separators):
    result = []
    for sep, _ in separators:
        split_text = text.split(sep)
        if len(split_text) != 2:
            print("T", text, sep)
            return False
        result.append(split_text[0].strip("\n "))
        text = split_text[1]
    result.append(text)

    return [
        _parse_numbered_list(parsed_text) if is_numbered_list else parsed_text
        for parsed_text, (_, is_numbered_list) in zip(result[1:], separators)
    ]


# used in StoryContextGenerator
def _remove_noun_articles(text):
    for prefix in ["a ", "an ", "the "]:
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
    return text


# used in ObjectStateUpdatesGenerator
def _parse_fields(model_response, fields):
    suggestions = []
    for elem in model_response.split("\n\n"):
        tmp = elem.split("\n")
        if len(tmp) != len(fields):
            continue
        if not all(tmp[i].startswith(k) for i, k in enumerate(fields)):
            continue
        parsed_fields = [tmp[i][len(k) :].strip() for i, k in enumerate(fields)]
        suggestions.append(parsed_fields)
    return suggestions
