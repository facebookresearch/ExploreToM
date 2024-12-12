# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from belief_tracker import FullBeliefTracker, QuestionGenerator, ChainOfThoughtGenerator


def test1():
    function_list = [
        lambda o: o.enter_room("David", "meeting room"),
        lambda o: o.move_object_container("David", "project prototype", "wooden crate"),
        lambda o: o.private_communication(
            "David",
            "Mark",
            topic=(
                "project prototype",
                FullBeliefTracker.CONTAINER_LOCATION,
                "wooden crate",
                True,
            ),
        ),
    ]
    return function_list


def test2():
    function_list = [
        lambda o: o.enter_room("Maya", "break room"),
        lambda o: o.move_object_container("Maya", "laptop", "wooden desk drawer"),
        lambda o: o.move_object_container("Maya", "laptop", "black canvas tote bag"),
        lambda o: o.move_object_room("Maya", "laptop", "conference room"),
        lambda o: o.enter_room("Maya", "garden"),
    ]
    return function_list


def test3():
    function_list = [
        lambda o: o.enter_room("Maya", "break room"),
        lambda o: o.enter_room("Anne", "break room"),
        lambda o: o.move_object_container("Maya", "laptop", "wooden desk drawer"),
        lambda o: o.enter_room("Maya", "garden"),
        lambda o: o.enter_room("John", "break room"),
    ]
    return function_list


def test_fantom():
    people_name_list = ["Anne", "Beth", "Charly"]
    room_name = "room"

    # FANToM story script without the implicitness of each step
    function_list = [
        lambda o: o.enter_room(people_name_list[0], room_name),
        lambda o: o.enter_room(people_name_list[1], room_name),
        lambda o: o.enter_room(people_name_list[2], room_name),
        lambda o: o.broadcast_communication(people_name_list[0], topic="topic1"),
        lambda o: o.leave_room(people_name_list[0], room_name),
        lambda o: o.broadcast_communication(people_name_list[1], topic="topic2"),
        lambda o: o.enter_room(people_name_list[0], room_name),
        lambda o: o.broadcast_communication(people_name_list[1], topic="topic3"),
    ]
    return function_list


def test_ice_cream_van_story():
    people_name_list = ["Anne", "Beth", "Charly"]
    room_name = "garden"

    # Ice Cream Van test
    A, B, C = people_name_list[0], people_name_list[1], people_name_list[2]
    function_list = [
        lambda o: o.location_declaration(
            A, room_name
        ),  # A = friend of the one getting ice cream
        lambda o: o.location_declaration(B, room_name),  # B = person getting ice cream
        lambda o: o.location_declaration("truck", room_name, is_thinking_entity=False),
        lambda o: o.location_declaration(C, room_name),  # C = ice cream van truck guy
        lambda o: o.leave_room(B, room_name),  # B leaves
        lambda o: o.move_object_room(C, obj="truck", new_room="church"),
        # C moves the object to the church
        lambda o: o.private_communication(
            C, A, topic=("truck", FullBeliefTracker.ROOM_LOCATION, "church", True)
        ),
        lambda o: o.private_communication(
            C, A, topic=(C, FullBeliefTracker.ROOM_LOCATION, "church", True)
        ),
        lambda o: o.private_communication(
            C, B, topic=("truck", FullBeliefTracker.ROOM_LOCATION, "church", True)
        ),
        lambda o: o.private_communication(
            C, B, topic=(C, FullBeliefTracker.ROOM_LOCATION, "church", True)
        ),
    ]
    return function_list


def test_tomi_story():
    people_name_list = ["Anne", "Beth", "Charly"]
    room_name = "living room"
    function_list = [
        lambda o: o.enter_room(people_name_list[0], room_name),
        lambda o: o.enter_room(people_name_list[1], room_name),
        lambda o: o.move_object_container(
            people_name_list[1], obj="ball", container="box"
        ),
        lambda o: o.leave_room(people_name_list[1], room_name),
        lambda o: o.move_object_container(
            people_name_list[0], obj="ball", container="basket"
        ),
        lambda o: o.enter_room(
            people_name_list[1], room_name
        ),  # person_1 should not magically know the basket is in the box
    ]
    return function_list


def test_apple():
    people_name_list = ["Anne", "Beth", "Charly"]
    room_name = "living room"
    function_list = [
        lambda o: o.enter_room(people_name_list[0], room_name),
        lambda o: o.enter_room(people_name_list[1], room_name),
        lambda o: o.update_object_state(
            people_name_list[1],
            "apple",
            "poisoned",
            "<is_edible>",
            f"{people_name_list[1]} poisoned the apple.",
            False,
        ),
        lambda o: o.leave_room(people_name_list[1], room_name),
        lambda o: o.update_object_state(
            people_name_list[0],
            "apple",
            "peeled",
            "<is_peeled>",
            f"{people_name_list[0]} peeled the apple.",
            True,
        ),
        lambda o: o.update_object_state(
            people_name_list[0],
            "apple",
            "covered in chocolate",
            "<is_covered_in_chocolate>",
            f"{people_name_list[0]} covered the {'apple'} in chocolate.",
            True,
        ),
        lambda o: o.enter_room(people_name_list[2], room_name),
        lambda o: o.move_object_container(people_name_list[2], "apple", "fridge"),
        lambda o: o.move_object_container(people_name_list[2], "banana", "fridge"),
        lambda o: o.enter_room(people_name_list[1], room_name),
    ]
    return function_list


def test_debug_interesting_q():
    """
    Benjamin entered the activity room.
    Benjamin moved the patient file to the file cabinet, which is also located in the activity room.
    Sara entered the activity room.
    Sara moved the patient file to the plastic bin, which is also located in the activity room.
    Sara left the activity room.
    Benjamin left the activity room.
    Sara entered the activity room.
    Sara moved the patient file to the cardboard box, which is also located in the activity room.
    """
    people_name_list = ["Benjamin", "Sara"]
    room_name = "activity room"
    object_name = "patient file"
    containers = ["file cabinet", "plastic bin", "cardboard box"]

    function_list = [
        lambda o: o.enter_room("Benjamin", room_name),
        lambda o: o.move_object_container("Benjamin", object_name, "file cabinet"),
        lambda o: o.enter_room("Sara", room_name),
        lambda o: o.move_object_container("Sara", object_name, "plastic bin"),
        lambda o: o.leave_room("Sara", room_name),
        lambda o: o.leave_room("Benjamin", room_name),
        lambda o: o.enter_room("Sara", room_name),
        lambda o: o.move_object_container("Sara", object_name, "cardboard box"),
    ]
    return function_list


def test_asymmetric():
    """
    Sara entered the activity room.
    Benjamin entered the activity room.
    Benjamin moved the patient file to the file cabinet, which is also located in the activity room.
    While this last action was happening, John was peeking through the window.
    While this last action was happening, Sara had fallen asleep.  # we should group these modifiers
    """
    people_name_list = ["Benjamin", "Sara"]
    room_name = "activity room"
    object_name = "patient file"
    containers = ["file cabinet", "plastic bin", "cardboard box"]

    function_list = [
        lambda o: o.enter_room("Benjamin", room_name),
        lambda o: o.enter_room("Sara", room_name),
        lambda o: o.move_object_container(
            "Benjamin",
            object_name,
            "file cabinet",
            people_peeking=["John"],
            people_distracted=["Sara"],
        ),
    ]
    return function_list


def _check_that_questions_are_not_repeated(belief_tracker):
    question_generator = QuestionGenerator(bt)
    first_order_questions = [
        question for question, answer, entities, metadata in question_generator.main(1)
    ]
    second_order_questions = [
        question for question, answer, entities, metadata in question_generator.main(2)
    ]
    assert len(first_order_questions) == len(
        set(first_order_questions)
    ), "There are repeated first-order questions"
    assert len(second_order_questions) == len(
        set(second_order_questions)
    ), "There are repeated second-order questions"


if __name__ == "__main__":
    for function_list in [
        test_asymmetric(),
        test_debug_interesting_q(),
        test_apple(),
        test_tomi_story(),
        test_ice_cream_van_story(),
        test_fantom(),
        test3(),
        test2(),
        test1(),
    ]:
        bt = FullBeliefTracker()
        for i, f in enumerate(function_list):
            output_flag = f(bt)
            assert output_flag, i
        print("Story", bt.story_script)

        question_generator = QuestionGenerator(bt)
        first_order_qs = question_generator.main(1)
        second_order_qs = question_generator.main(2)
        _check_that_questions_are_not_repeated(bt)

        chain_of_thought_generator = ChainOfThoughtGenerator()
        for nth_reasoning_order in [1, 2]:
            for question, answer, entities, metadata in QuestionGenerator(bt).main(
                nth_reasoning_order=nth_reasoning_order
            ):
                # print('-' * 50)
                cot = chain_of_thought_generator.main(
                    bt.story_script_tuple_representation,
                    question,
                    entities,
                    answer,
                    nth_reasoning_order=nth_reasoning_order,
                )
                # if cot:
                #    print('')
                #    print('QUESTION:', question, '|', answer, '|', entities)
                #    print('CHAIN OF THOUGHT:')
                #    print(cot)
