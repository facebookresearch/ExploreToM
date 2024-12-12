# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
TrackTheMind's domain-specific language for Theory of Mind.

TODO question generation output format (especially metadata should be improved! but postponing until after everything else is ready)

TODO Ideas to clean FullBeliefTracker:
    - abstract regular preconditions
    - abstract regular belief updates wherever possible
    - abstract peeking and distracted double loop for second order
    - build desired ground truths for tests

"""
import copy
import collections


class WorldState:
    def __init__(self):
        self.world_state = {}  # [entity/thing -> properties]

    def __str__(self):
        return str(self.world_state)

    def update(self, entity, property_name, new_property_value, replaces_previous=True):
        if entity not in self.world_state:
            self.world_state[entity] = {}
        if property_name not in self.world_state[entity]:
            self.world_state[entity][property_name] = set(
                []
            )  # assuming property might not replace previous

        if replaces_previous:
            old_property_value = self.world_state[entity][property_name]
            self.world_state[entity][property_name] = new_property_value
            return old_property_value != new_property_value
        else:
            old_property_value = copy.copy(self.world_state[entity][property_name])
            self.world_state[entity][property_name].add(new_property_value)
            return new_property_value not in old_property_value

    def get(self, entity, property_name):
        return self.world_state.get(entity, {}).get(property_name, None)

    def get_all_properties(self, entity):
        return self.world_state.get(entity, {})

    def keys(self):
        return self.world_state.keys()


class FullBeliefTracker:
    """
    Domain-Specific language for Theory of Mind reasoning.

    All actions return a boolean that represents whether the action was successfully performed ot not.
    All actions have the same three-step structure (sometimes in different order):
        0. Checking the preconditions to execute the action (e.g. we cannot move an object if it is already there)
        1. Update the story script and update object variables (e.g. self.objects)
        2. Update the world state and the beliefs of all action witnesses

    Preconditions need to be checked before any world state or belief update, as there is no way to roll back an update.

    Communication can be about abstract topics (knowledge) or about to communicate something about
    the current world state. This has to be knowledge that the speaker already possesses and needs to match
    the ground truth to simplify and avoid considerations about e.g. outdated information.
    """

    CONTAINER_LOCATION = "container_location"
    ROOM_LOCATION = "room_location"
    KNOWLEDGE = "<knowledge>"
    GENERAL_STATE_UPDATE = "<state_update>"
    PROPERTIES_TEXTUAL_TRANSLATION = {
        ROOM_LOCATION: "{} is in the {}",
        CONTAINER_LOCATION: "{} is in the {}",
        KNOWLEDGE: "the {}",
    }

    def __init__(self):
        self.thinking_entities = set([])  # entities with thought
        self.world_state = WorldState()  # [entity/thing -> properties]

        # super simple implementation, we could consider higher levels of recursion
        self.first_order_beliefs = {}  # entity -> WorldState()
        self.second_order_beliefs = {}  # entity -> entity -> WorldState()

        self.story_script = []
        self.story_script_tuple_representation = []

        self.people = set([])
        self.containers = set([])
        self.objects = set([])
        self.topics = set([])
        self.rooms = set([])

        self.visible_property_names = set([self.ROOM_LOCATION])
        self.invisible_property_names = set([self.CONTAINER_LOCATION, self.KNOWLEDGE])

    def __str__(self):
        object_str = ""
        object_str += "Story: " + " ".join(self.story_script) + "\n"
        object_str += f"WorldState: {self.world_state.__str__()}\n"
        for e in self.first_order_beliefs:
            object_str += (
                f"\tFirstOrder[{e}]: {self.first_order_beliefs[e].__str__()}\n"
            )
        for e in self.second_order_beliefs:
            for e2 in self.second_order_beliefs[e]:
                object_str += f"\t\tSecondOrder[{e}][{e2}]: {self.second_order_beliefs[e][e2].__str__()}\n"
        return object_str

    def add_entity(self, entity_with_tom):
        if entity_with_tom not in self.thinking_entities:
            self.thinking_entities.add(entity_with_tom)
            self.first_order_beliefs[entity_with_tom] = WorldState()
            self.second_order_beliefs[entity_with_tom] = {
                e: WorldState() for e in self.thinking_entities
            }
            for e in self.thinking_entities:
                self.second_order_beliefs[e][entity_with_tom] = WorldState()
            self.people.add(entity_with_tom)

    def is_object_visible(self, world_state, obj):
        # Simplification: the only way to be hidden in our world modeling is to be inside a container
        return world_state.get(obj, self.CONTAINER_LOCATION) is None

    def is_property_visible(self, property_name):
        return property_name in self.visible_property_names

    def add_property(self, property_name, is_visible):
        if is_visible:
            self.visible_property_names.add(property_name)
        else:
            self.invisible_property_names.add(property_name)
        assert (
            len(self.visible_property_names & self.invisible_property_names) == 0
        ), "There should not be overlap between visible and invisible properties"

    # **************** UTILS FOR TURNING STUFF INTO TEXT ****************
    def _enumeration_in_text(self, people_list):
        if len(people_list) == 1:
            return people_list[0]
        if len(people_list) == 2:
            return f"{people_list[0]} and {people_list[1]}"
        return f"{', '.join(people_list[:-1])}, and {people_list[-1]}"

    def _textifying_communication_topic(self, topic):
        if isinstance(topic, str):
            return topic
        if isinstance(topic, tuple):
            entity, property_name, new_property_value, replaces_previous = topic
            if (
                property_name == self.ROOM_LOCATION
                or property_name == self.CONTAINER_LOCATION
            ):
                if entity in self.people:
                    return f"{entity} is in the {new_property_value}"  # names do not start with 'the'
                else:
                    return f"the {entity} is in the {new_property_value}"
        assert False, "Unimplemented textification"

    def get_all_beliefs_in_textual_form(self):
        # A. world state beliefs (does not include knowledge since it is only mental)
        prev_world_state_list = []
        for entity, state_dict in self.world_state.world_state.items():
            for location_type, v in state_dict.items():
                if location_type.startswith(f"{self.GENERAL_STATE_UPDATE} "):
                    self.PROPERTIES_TEXTUAL_TRANSLATION[location_type] = "{} {}"
                prev_world_state_list.append(
                    self.PROPERTIES_TEXTUAL_TRANSLATION[location_type].format(entity, v)
                )

        # B. First order beliefs
        prev_first_order_beliefs_list = []
        for entity, state_dict in self.first_order_beliefs.items():
            for entity2, state_dict2 in state_dict.world_state.items():
                if entity == entity2:
                    continue
                for location_type, v in state_dict2.items():
                    if location_type == self.KNOWLEDGE:
                        for e in v:
                            prev_first_order_beliefs_list.append(
                                f"{entity} heard about "
                                + self.PROPERTIES_TEXTUAL_TRANSLATION[
                                    location_type
                                ].format(e)
                            )
                    else:
                        prev_first_order_beliefs_list.append(
                            f"{entity} believes that "
                            + self.PROPERTIES_TEXTUAL_TRANSLATION[location_type].format(
                                entity2, v
                            )
                        )

        # C. Second order beliefs
        prev_second_order_beliefs_list = []
        for entity, state_dict in self.second_order_beliefs.items():
            for entity1, state_dict1 in state_dict.items():
                if entity == entity1:
                    continue
                for entity2, state_dict2 in state_dict1.world_state.items():
                    for location_type, v in state_dict2.items():
                        if location_type == self.KNOWLEDGE:
                            for e in v:
                                prev_second_order_beliefs_list.append(
                                    f"{entity} believes that {entity1} heard about "
                                    + self.PROPERTIES_TEXTUAL_TRANSLATION[
                                        location_type
                                    ].format(e)
                                )
                        else:
                            prev_second_order_beliefs_list.append(
                                f"{entity} believes that {entity1} believes that "
                                + self.PROPERTIES_TEXTUAL_TRANSLATION[
                                    location_type
                                ].format(entity2, v)
                            )

        return (
            prev_world_state_list,
            prev_first_order_beliefs_list,
            prev_second_order_beliefs_list,
        )

    # **************** UTILS FOR PEEKING AND DISTRACTED ****************
    def _add_peeking_and_distracted_story_script_and_tuple_representation(
        self, **kwargs
    ):
        if kwargs.get("people_peeking") or kwargs.get("people_distracted"):
            text = "While this action was happening, "
            if kwargs.get("people_peeking"):
                text += f"{self._enumeration_in_text(kwargs.get('people_peeking'))} witnessed this action in secret (and only this action)."
            if kwargs.get("people_peeking") and kwargs.get("people_distracted"):
                text = text[:-1]
                text += "; also, "
            if kwargs.get("people_distracted"):
                text += f"{self._enumeration_in_text(kwargs.get('people_distracted'))} got distracted and did not realize what happened, without anyone noticing the brief lack of attention, and going back to paying attention immediately after the action was finished."
            self.story_script.append(text)
            self.story_script_tuple_representation.append(
                (
                    "peeking_distracted",
                    kwargs.get("people_peeking", []),
                    kwargs.get("people_distracted", []),
                )
            )

    def _check_basic_peeking_and_distracted_preconditions(
        self, witnesses, person_performing_action=None, **kwargs
    ):
        """
        - if there is a person performing the action, they cannot be peeking
        - if there is a person performing the action, they cannot be distracted
        - the peeker should not be a witness otherwise
        - the distracted should be witnesses if they were paying attention
        - no one can be distracted and peeking at the same time

        Some of these checks may be redundant if person_performing_action is already included in witnesses.
        """
        if (
            person_performing_action is not None
            and person_performing_action in kwargs.get("people_peeking", [])
        ):  # the person_performing_action cannot be peeking
            return False
        if (
            person_performing_action is not None
            and person_performing_action in kwargs.get("people_distracted", [])
        ):  # the person_performing_action cannot be distracted
            return False
        if any(
            p in witnesses for p in kwargs.get("people_peeking", [])
        ):  # the peeker should not be a witness otherwise
            return False
        if any(
            p not in witnesses for p in kwargs.get("people_distracted", [])
        ):  # the distracted should be witnesses if they were paying attention
            return False
        if (
            len(
                set(kwargs.get("people_peeking", []))
                & set(kwargs.get("people_distracted", []))
            )
            > 0
        ):  # no one can be distracted and peeking at the same time
            return False
        for p in kwargs.get("people_peeking", []):
            self.add_entity(p)
        return True

    def _real_witnesses_after_peeking_and_distracted(self, witnesses, **kwargs):
        return set(witnesses + kwargs.get("people_peeking", [])) - set(
            kwargs.get("people_distracted", [])
        )

    # ************************** ACTIONS ********************************
    def location_declaration(self, entity, room, is_thinking_entity=True):
        enter_success = self.enter_room(
            entity,
            room,
            skip_story_script_update=True,
            is_thinking_entity=is_thinking_entity,
        )
        if enter_success:
            self.story_script.append(f"{entity} is in the {room}.")
            self.story_script_tuple_representation.append(
                ("location_declaration", entity, room)
            )
        return enter_success

    def enter_room(
        self,
        entity,
        room,
        skip_story_script_update=False,
        is_thinking_entity=True,
        **kwargs,
    ):
        """
        A person/object enters the room.

        This can be used as a standalone action ("John enters the room") or as a helper in other functions.
        Enter room is the most complicated of belief updates, since when people reenter a room, they subtly gain the information of the new location they enter.

        Args:
            skip_story_script_update (bool): determines whether to update the story script or not (useful when calling this as a helper function).
            is_thinking_entity (bool): determines if the entity entering the room should have its own belief entries in self.first_order_beliefs or not (useful when calling this as a helper function).
        """
        assert (
            is_thinking_entity or skip_story_script_update
        ), "is_thinking_entity=False => skip_story_script_update=True, since objects cannot magically enter a room."

        # 0. Preconditions
        ## 0.a Preconditions about the person & object locations
        if self.world_state.get(entity, self.ROOM_LOCATION) == room:
            return False

        ## 0.b Preconditions about peeking & distracted people (needs to be checked before any world state update)
        witnesses = [
            W
            for W in self.thinking_entities
            if self.world_state.get(W, self.ROOM_LOCATION) == room
        ]  # does not include person yet
        successful_checks = self._check_basic_peeking_and_distracted_preconditions(
            witnesses, entity, **kwargs
        )
        if not successful_checks:
            return False

        # 2. Update world state & beliefs of all witnesses
        ## 2.a Update explicit knowledge gain: the witnesses see the new person coming into the room
        if is_thinking_entity:
            self.add_entity(entity)
        self.rooms.add(room)
        self.world_state.update(entity, self.ROOM_LOCATION, room)
        witnesses = [
            W
            for W in self.thinking_entities
            if self.world_state.get(W, self.ROOM_LOCATION) == room
        ]  # includes person that entered
        for w in self._real_witnesses_after_peeking_and_distracted(witnesses, **kwargs):
            self.first_order_beliefs[w].update(entity, self.ROOM_LOCATION, room)
            for b in witnesses:
                self.second_order_beliefs[w][b].update(entity, self.ROOM_LOCATION, room)

        ## 2.b Implicit knowledge gain that person entering the room and the witnesses gain as a result of the action
        ## (When someone enters a room, they observe what's in the room and what's changed; the witnesses know this)
        ## (It's crucial to distinguish whether the object, and particular properties of an object, are visible or not)
        objects_and_people_in_room = [
            W
            for W in self.world_state.keys()
            if self.world_state.get(W, self.ROOM_LOCATION) == room
        ]
        entity_is_actually_a_person = entity in self.first_order_beliefs
        for e in objects_and_people_in_room:
            is_e_visible = self.is_object_visible(self.world_state, e)
            e_used_to_be_visible_to_person = (
                entity_is_actually_a_person
                and self.is_object_visible(self.first_order_beliefs[entity], e)
            )
            e_is_known_to_person = (
                entity_is_actually_a_person
                and len(self.first_order_beliefs[entity].get_all_properties(e)) > 0
            )

            ### 2.b.1 Knowledge update if an object is not visible
            # The only relevant case is when someone reenters a room and they do not see an object/people they used to see
            # If they didn't use to be able to see the entity (e.g. it was already inside of a container), then they'll assume object permanence and no changes are needed.
            # If they used to be able, knowledge is updated to 'not(???)' to trigger the "not(" detectors.
            if (
                not is_e_visible
                and entity_is_actually_a_person
                and e_used_to_be_visible_to_person
                and e_is_known_to_person
            ):
                self.first_order_beliefs[entity].update(
                    e, self.ROOM_LOCATION, "not(???)"
                )
                # current witnesses know that the person coming back does not know where the object is
                for w in self._real_witnesses_after_peeking_and_distracted(
                    witnesses, **kwargs
                ):
                    self.second_order_beliefs[w][entity].update(
                        e, self.ROOM_LOCATION, "not(???)"
                    )
                for w in witnesses:
                    self.second_order_beliefs[entity][w].update(
                        e, self.ROOM_LOCATION, "not(???)"
                    )

            ### 2.b.2 Knowledge update if an object is visible. Only concerns visible properties of the visible object.
            if is_e_visible:
                visible_properties_of_e = [
                    (prop_type, self.world_state.get(e, prop_type))
                    for prop_type in self.world_state.get_all_properties(e).keys()
                    if self.is_property_visible(prop_type)
                ]

                ### A person reenters the room and sees the visible properties of visible objects
                ### We assume perfect vision for simplicity.
                if entity_is_actually_a_person:
                    for property_name, property_value in visible_properties_of_e:
                        self.first_order_beliefs[entity].update(
                            e, property_name, property_value
                        )

                ### A person reenters the room, and everyone in the room knows that everyone else in the room knows
                ### about the visible properties of an object. Updating here since "witnesses" contains the new person.
                ### One of those visible properties is the object's location, so no special code is needed!
                ### We assume perfect vision for simplicity.
                for w in self._real_witnesses_after_peeking_and_distracted(
                    witnesses, **kwargs
                ):
                    for b in witnesses:
                        for property_name, property_value in visible_properties_of_e:
                            self.second_order_beliefs[w][b].update(
                                e, property_name, property_value
                            )

        # 1. Story script update & story script tuple representation update & update objects/containers/rooms
        if not skip_story_script_update:
            self.story_script.append(f"{entity} entered the {room}.")
            self.story_script_tuple_representation.append(("enter_room", entity, room))
        self._add_peeking_and_distracted_story_script_and_tuple_representation(**kwargs)
        return True

    def _leave_location(
        self,
        entity,
        location,
        skip_story_script_update=False,
        location_type=ROOM_LOCATION,
        **kwargs,
    ):
        """
        An entity (person/object) leaves a location (room/container).
        """

        # 0. Check preconditions
        ## 0.a Preconditions about the person & object locations
        # MSCLARNOTES: removed redundant checks
        if (
            self.world_state.get(entity, location_type) != location
        ):  # Person must not already be in the desired destination location
            return False

        ## X. Witness computation. Even if the update is about a container, witnesses are those in the room.
        # Done here since this affects preconditions about peeking & distracted people
        room_location_entity = self.world_state.get(entity, self.ROOM_LOCATION)
        witnesses = [
            W
            for W in self.thinking_entities
            if self.world_state.get(W, self.ROOM_LOCATION) == room_location_entity
        ]  # includes "entity" on purpose

        ## 0.b Preconditions about peeking & distracted people
        successful_checks = self._check_basic_peeking_and_distracted_preconditions(
            witnesses, entity, **kwargs
        )
        if not successful_checks:
            return False

        # 2. Update world state & beliefs of all witnesses
        not_location_str = f"not({location})"
        self.world_state.update(entity, location_type, not_location_str)
        for w in self._real_witnesses_after_peeking_and_distracted(witnesses, **kwargs):
            self.first_order_beliefs[w].update(entity, location_type, not_location_str)
            for b in witnesses:
                self.second_order_beliefs[w][b].update(
                    entity, location_type, not_location_str
                )

        # 1. Story script update & story script tuple representation update & update objects/containers/rooms
        if not skip_story_script_update:
            self.story_script.append(f"{entity} left the {location}.")
            self.story_script_tuple_representation.append(
                ("leave_room", entity, location)
            )
            self._add_peeking_and_distracted_story_script_and_tuple_representation(
                **kwargs
            )
        return True

    def leave_room(self, entity, location, skip_story_script_update=False, **kwargs):
        return self._leave_location(
            entity, location, skip_story_script_update, self.ROOM_LOCATION, **kwargs
        )

    def leave_container(
        self, entity, location, skip_story_script_update=False, **kwargs
    ):
        return self._leave_location(
            entity,
            location,
            skip_story_script_update,
            self.CONTAINER_LOCATION,
            **kwargs,
        )

    def move_object_container(self, person, obj, container, **kwargs):
        """
        Moving an object to a container.

        We detect the room in which the object is, and assume that everyone in that room can see it being moved.

        Preconditions:
            - The person and the object are initially in the same room
            - The room needs to be an actual location (not "not(kitchen)")
            - The destination room of the object cannot be the current room
        """

        # 0. Check preconditions
        ## 0.a Preconditions about the person & object locations
        room1 = self.world_state.get(obj, self.ROOM_LOCATION)
        room2 = self.world_state.get(person, self.ROOM_LOCATION)
        if (
            room2 is None
        ):  # not enforcing this can be problematic when this sentence is in the middle of a story
            return False
        if room1 is not None and room1 != room2:  # contradicting locations
            return False
        room = room2
        if room.startswith("not("):
            return False
        if container == self.world_state.get(
            obj, self.CONTAINER_LOCATION
        ):  # you cannot move the object if it is already there
            return False

        ## 0.b Preconditions about peeking & distracted people
        witnesses = [
            W
            for W in self.thinking_entities
            if self.world_state.get(W, self.ROOM_LOCATION) == room
        ]  # includes person
        successful_checks = self._check_basic_peeking_and_distracted_preconditions(
            witnesses, person, **kwargs
        )
        if not successful_checks:
            return False

        # 1. Story script update & story script tuple representation update & update objects/containers/rooms
        self.story_script.append(
            f"{person} moved the {obj} to the {container}, which is also located in the {room}."
        )
        self.story_script_tuple_representation.append(
            ("move_object_container", person, obj, container)
        )
        self._add_peeking_and_distracted_story_script_and_tuple_representation(**kwargs)
        self.objects.add(obj)
        self.containers.add(container)
        self.rooms.add(room)

        # 2. Update world state & beliefs of all witnesses
        self.world_state.update(obj, self.CONTAINER_LOCATION, container)
        self.world_state.update(obj, self.ROOM_LOCATION, room)
        self.world_state.update(container, self.ROOM_LOCATION, room)
        for w in self._real_witnesses_after_peeking_and_distracted(witnesses, **kwargs):
            self.first_order_beliefs[w].update(obj, self.CONTAINER_LOCATION, container)
            self.first_order_beliefs[w].update(obj, self.ROOM_LOCATION, room)
            for b in witnesses:
                self.second_order_beliefs[w][b].update(
                    obj, self.CONTAINER_LOCATION, container
                )
                self.second_order_beliefs[w][b].update(obj, self.ROOM_LOCATION, room)
        for b in witnesses:
            for w in witnesses:
                self.second_order_beliefs[b][w].update(
                    obj, self.CONTAINER_LOCATION, container
                )
                self.second_order_beliefs[b][w].update(obj, self.ROOM_LOCATION, room)
        return True

    def move_object_room(
        self, person, obj, new_room, object_leaves_container_behind=True
    ):
        """
        Moving an object to a room.

        We detect the room in which the object is, and assume that everyone in that room can see it being moved.

        Preconditions:
            - The person and the object are initially in the same room
            - The room needs to be an actual location (not "not(kitchen)")
            - The destination room of the object cannot be the current room

        Notes:
            - Moving an object to a new room does not currently support peeking & distracted,
                since we'd need to distinguish if they were peeking in the source or target room.
            - We only tested object_leaves_container_behind=True.
        """

        # 0. Check preconditions
        room1 = self.world_state.get(obj, self.ROOM_LOCATION)
        room2 = self.world_state.get(person, self.ROOM_LOCATION)
        if (
            room1 is None or room2 is None
        ):  # both locations need to have been mentioned for easier understanding
            return False
        if room1 != room2:  # contradicting locations
            return False
        room = room1
        if room.startswith("not("):
            return False
        if room == new_room:  # you cannot move the object if it is already there
            return False

        # 1. Story script update & story script tuple representation update & update objects/containers/rooms
        container = self.world_state.get(obj, self.CONTAINER_LOCATION)
        container = (
            None if container is None or container.startswith("not(") else container
        )
        if container is not None and object_leaves_container_behind:
            self.story_script.append(
                f"{person} moved the {obj} to the {new_room}, leaving the {container} in its original location."
            )
        elif container is not None and not object_leaves_container_behind:
            assert (
                False
            ), "We did not test this variant, although it should be simple to do so."
            self.story_script.append(
                f"{person} moved the {obj} to the {new_room}, still inside {container}."
            )
        else:
            self.story_script.append(f"{person} moved the {obj} to the {new_room}.")
        self.story_script_tuple_representation.append(
            ("move_object_room", person, obj, new_room)
        )
        self.objects.add(obj)
        self.rooms.add(new_room)

        ## 1.b & 2. Room movements of both the object and person.
        # Order is important so that person knows the object is leaving.
        # [Update world state & beliefs of all witnesses happens implicitly in leave_room()]
        # [Story Script Update already happened, hence why skip_story_script_update=True]

        ## Taking the object out of the container before moving it if needed
        # MSCLARNOTES: note that I changed order here, I do not think it affects correctness
        if container is not None and object_leaves_container_behind:
            output_flag = self.leave_container(
                obj, container, skip_story_script_update=True
            )
            assert output_flag

        ## Room departure is ordered like this so that person knows the object is leaving
        output_flag = self.leave_room(obj, room, skip_story_script_update=True)
        assert output_flag
        output_flag = self.leave_room(person, room, skip_story_script_update=True)
        assert output_flag

        ## Room entrance is ordered like this so that person knows the object has entered (they grabbed it)
        output_flag = self.enter_room(person, new_room, skip_story_script_update=True)
        assert output_flag
        output_flag = self.enter_room(
            obj, new_room, skip_story_script_update=True, is_thinking_entity=False
        )
        assert output_flag
        return True

    def update_object_state(
        self,
        person,
        obj,
        object_state_value,
        object_state_type,
        text_description,
        is_new_state_visible,
        **kwargs,
    ):
        """
        Update object state. Similar in mental state update logic to moving an object to a container.

        Limitation: this assumes the state is accumulative, but in some cases, something may stop being visible because of another change.
        """

        # adding a prefix to make these actions' sources identifiable
        object_state_type = self.GENERAL_STATE_UPDATE + " " + object_state_type

        # 0. Preconditions
        ## 0.a Preconditions about the person & object locations (object and person are in same room; state is new)
        room1 = self.world_state.get(obj, self.ROOM_LOCATION)
        room2 = self.world_state.get(person, self.ROOM_LOCATION)
        if room2 is None:
            return False  # not enforcing this can be problematic when this sentence is in the middle of a story
        if room1 is not None and room1 != room2:
            return False  # Contradicting locations
        room = room2
        if room.startswith("not("):
            return False
        # the new state value needs to be new (note that the world state for a specific property can be a list or a string)
        if self.world_state.get(obj, object_state_type) is not None and (
            object_state_value in self.world_state.get(obj, object_state_type)
            or self.world_state.get(obj, object_state_type) == object_state_value
        ):
            return False

        ## 0.b Preconditions about peeking & distracted people
        witnesses = [
            W
            for W in self.thinking_entities
            if self.world_state.get(W, self.ROOM_LOCATION) == room
        ]  # includes person
        successful_checks = self._check_basic_peeking_and_distracted_preconditions(
            witnesses, person, **kwargs
        )
        if not successful_checks:
            return False

        # 2. Update world state & beliefs of all witnesses
        self.objects.add(obj)
        self.rooms.add(room)
        self.world_state.update(obj, self.ROOM_LOCATION, room)
        self.add_property(
            object_state_type, is_new_state_visible
        )  # also tracking negatives as a check
        self.world_state.update(obj, object_state_type, object_state_value)
        for w in self._real_witnesses_after_peeking_and_distracted(witnesses, **kwargs):
            self.first_order_beliefs[w].update(
                obj, object_state_type, object_state_value
            )
            self.first_order_beliefs[w].update(obj, self.ROOM_LOCATION, room)
            for b in witnesses:
                self.second_order_beliefs[w][b].update(
                    obj, object_state_type, object_state_value
                )
                self.second_order_beliefs[w][b].update(obj, self.ROOM_LOCATION, room)

        # 1. Story script update & story script tuple representation update & update objects/containers/rooms
        self.story_script.append(text_description)
        self.story_script_tuple_representation.append(
            (
                object_state_type,
                person,
                obj,
                object_state_value,
                text_description,
                is_new_state_visible,
            )
        )
        self._add_peeking_and_distracted_story_script_and_tuple_representation(**kwargs)

        return True

    def _update_broadcast_communication_world_state_and_beliefs(
        self, topic, witnesses, **kwargs
    ):
        is_topic_abstract_knowledge = isinstance(topic, str)
        is_topic_world_state_communication = isinstance(topic, tuple)
        assert is_topic_abstract_knowledge or is_topic_world_state_communication

        # 2. Update world state & beliefs of all witnesses
        updated_knowledge = False
        if is_topic_abstract_knowledge:
            for w in set(witnesses) - set(kwargs.get("people_distracted", [])):
                updated_knowledge |= self.first_order_beliefs[w].update(
                    "", self.KNOWLEDGE, topic, replaces_previous=False
                )  # w knows about topic
                for b in witnesses:
                    updated_knowledge |= self.second_order_beliefs[w][b].update(
                        "", self.KNOWLEDGE, topic, replaces_previous=False
                    )  # w knows that b knows about topic

            # computing this separately since I do not want to count this for the "updated_knowledge" condition
            # because the action should make sense regardless of the people peeking
            for w in kwargs.get("people_peeking", []):
                self.first_order_beliefs[w].update(
                    "", self.KNOWLEDGE, topic, replaces_previous=False
                )
                for b in witnesses:
                    self.second_order_beliefs[w][b].update(
                        "", self.KNOWLEDGE, topic, replaces_previous=False
                    )

        elif is_topic_world_state_communication:
            entity, property_name, new_property_value, replaces_previous = topic
            for w in set(witnesses) - set(kwargs.get("people_distracted", [])):
                information_is_believable_for_w = True  # See Limitation!
                if information_is_believable_for_w:
                    updated_knowledge |= self.first_order_beliefs[w].update(
                        entity, property_name, new_property_value, replaces_previous
                    )  # w knows about topic
                for b in witnesses:
                    updated_knowledge |= self.second_order_beliefs[w][b].update(
                        entity, property_name, new_property_value, replaces_previous
                    )  # w knows that b knows about topic

            # computing this separately since I do not want to count this for the "updated_knowledge" condition
            # because the action should make sense regardless of the people peeking
            for w in kwargs.get("people_peeking", []):
                information_is_believable_for_w = True  # See Limitation!
                if information_is_believable_for_w:
                    updated_knowledge |= self.first_order_beliefs[w].update(
                        entity, property_name, new_property_value, replaces_previous
                    )  # w knows about topic
                for b in witnesses:
                    updated_knowledge |= self.second_order_beliefs[w][b].update(
                        entity, property_name, new_property_value, replaces_previous
                    )  # w knows that b knows about topic
        return updated_knowledge

    def private_communication(self, person1, person2, topic, **kwargs):
        """
        Communicate information to only one people, which may or may not be in the room (e.g. text message).
        The topic can be abstract or world knowledge update.

        Preconditions:
            - The two people need to be different
            - Usual peeking & distracted preconditions
            - person1 would not tell something to person2 that they believe they know

        Limitation:
            - We always assume that the information is believable.
            - This function is assumed to be called always with true information (i.e. true world state) so that
                there are no potential lies involved in a story. Support for "believable lies" could be added.
        """
        # 0. Preconditions
        ## 0.a The two people involved in the communication need to be different
        if person1 == person2:
            return False

        is_topic_abstract_knowledge = isinstance(topic, str)
        is_topic_world_state_communication = isinstance(topic, tuple)

        ## 0.c Preconditions on reasonable conditions where someone would talk about this
        ## Specifically, we check that person1 does not believe that person2 knows about this
        ## i.e. ban "George told privately to Anne that topic3. Anne told privately to George that topic3."
        if (
            person1 in self.second_order_beliefs
            and person2 in self.second_order_beliefs[person1]
        ):
            if is_topic_abstract_knowledge:
                known_topics = self.second_order_beliefs[person1][person2].get(
                    "", self.KNOWLEDGE
                )
                if known_topics is not None and topic in known_topics:
                    return False
            if is_topic_world_state_communication:
                entity, property_name, new_property_value, replaces_previous = topic
                if (
                    self.second_order_beliefs[person1][person2].get(
                        entity, property_name
                    )
                    == new_property_value
                ):
                    return False

        ## 0.b Preconditions about peeking & distracted people
        successful_checks = self._check_basic_peeking_and_distracted_preconditions(
            [person1, person2], person1, **kwargs
        )
        if not successful_checks:
            return False

        # 2. Update world state & beliefs of all witnesses
        self.add_entity(person1)
        self.add_entity(person2)
        is_topic_abstract_knowledge = isinstance(topic, str)
        is_topic_world_state_communication = isinstance(topic, tuple)
        updated_knowledge = (
            self._update_broadcast_communication_world_state_and_beliefs(
                topic, [person1, person2], **kwargs
            )
        )  # MSCLARNOTES this abstraction was checked visually to be identical
        if updated_knowledge and is_topic_abstract_knowledge:
            self.topics.add(
                topic
            )  # do not add the topics if it's a tuple, since that is info about other objects

        # 1. Story script update & story script tuple representation update & update objects/containers/rooms
        if updated_knowledge:
            if is_topic_abstract_knowledge:
                self.story_script.append(
                    f"{person1} told privately to {person2} about the {self._textifying_communication_topic(topic)}."
                )
            elif is_topic_world_state_communication:
                self.story_script.append(
                    f"{person1} told privately to {person2} that {self._textifying_communication_topic(topic)}."
                )
            else:
                assert False
            self.story_script_tuple_representation.append(
                ("private_communication", person1, person2, topic)
            )  # self._textifying_communication_topic(topic)))
            self._add_peeking_and_distracted_story_script_and_tuple_representation(
                **kwargs
            )
        return updated_knowledge

    def broadcast_communication(self, person, topic, **kwargs):
        """
        Communicate information to all other people in the room. The topic can be abstract or world knowledge update.
        Preconditions:
            - The person's room needs to be known
            - There has to be at least one other person in the room
            - Usual peeking & distracted preconditions
            - The knowledge communicated has to be novel even when ignoring peeking & distracted. This is because the person
              would not communicate something that is trivially true to everyone. Our checks on this may be overly strict.

        Limitation:
            - We always assume that the information is believable.
            - This function is assumed to be called always with true information (i.e. true world state) so that
                there are no potential lies involved in a story. Support for "believable lies" could be added.
        """

        # 0. Preconditions
        ## 0.a Preconditions about the person locations (location must be known and there has to be someone to talk to)
        room = self.world_state.get(person, self.ROOM_LOCATION)
        if room is None or room.startswith("not("):
            return False
        witnesses = [
            W
            for W in self.thinking_entities
            if self.world_state.get(W, self.ROOM_LOCATION) == room
        ]  # does not include peekers ; MSCLARNOTES changed it to thinking entities
        if len(witnesses) == 1:  # why would you speak out loud to yourself?
            return False

        ## 0.b Preconditions about peeking & distracted people
        successful_checks = self._check_basic_peeking_and_distracted_preconditions(
            witnesses, person, **kwargs
        )  # MSCLARNOTES this abstraction added extra checks on people_distracted (good!)
        if not successful_checks:
            return False

        # 2. Update world state & beliefs of all witnesses
        is_topic_abstract_knowledge = isinstance(topic, str)
        is_topic_world_state_communication = isinstance(topic, tuple)
        updated_knowledge = (
            self._update_broadcast_communication_world_state_and_beliefs(
                topic, witnesses, **kwargs
            )
        )

        # 1. Story script update & story script tuple representation update & update objects/containers/rooms
        if updated_knowledge and is_topic_abstract_knowledge:
            self.topics.add(
                topic
            )  # do not add the topics if it's a tuple, since that is info about other objects
        if updated_knowledge:
            if is_topic_abstract_knowledge:
                self.story_script.append(
                    f"{person} told out loud about the {self._textifying_communication_topic(topic)}."
                )
            elif is_topic_world_state_communication:
                self.story_script.append(
                    f"{person} told out loud that {self._textifying_communication_topic(topic)}."
                )
            self.story_script_tuple_representation.append(
                ("broadcast_communication", person, topic)
            )  # self._textifying_communication_topic(topic)))
            self._add_peeking_and_distracted_story_script_and_tuple_representation(
                **kwargs
            )
        return updated_knowledge


class QuestionGenerator:
    """
    Class to generate questions for a given belief tracker.
    This abstraction is simply to separate functionality.

    Currently, the output format is a bit of a mess. relation_type_expanded was used for something that I do not remember anymore. FIXME.
    """

    BELIEF_LOCATION_FIRST_ORDER_QS = ["Where will {} search for the {}?"]
    BELIEF_LOCATION_SECOND_ORDER_QS = [
        "Where does {} think that {} will search for the {}?"
    ]

    BELIEF_ROOM_LOCATION_FIRST_ORDER_QS = ["In which room will {} search for the {}?"]
    BELIEF_ROOM_LOCATION_SECOND_ORDER_QS = [
        "In which room does {} think that {} will search for the {}?"
    ]
    BELIEF_LOCATION_FIRST_ORDER_QS = ["In which container will {} search for the {}?"]
    BELIEF_LOCATION_SECOND_ORDER_QS = [
        "In which container does {} think that {} will search for the {}?"
    ]

    # questions about binary beliefs about object state
    BINARY_BELIEF_FIRST_ORDER_QS = ["Does {} believe that the {} {}? Answer yes or no."]
    BINARY_BELIEF_SECOND_ORDER_QS = [
        "Does {} think that {} believes that the {} {}? Answer yes or no."
    ]

    """
    We only cover a small subset of FANToM's diverse questions:
    1. Factual Q about a topic discussed.
        e.g. "What is Linda's dog breed?", if this fact was mentioned in discussion.
    2. Belief about fact of topic discussed. 
        e.g. "What breed would Kailey think Linda’s dog is?" (multiple choice / open ended)
    3. List all the characters who know the correct answer to this question.
    4. List all the characters who know this information.
    5. "Does David know the correct answer to this question?"
    
    Second-order: What does David think about Kailey’s belief on the breed of Linda’s dog?
    """
    KNOWLEDGE_FIRST_ORDER_Q = ["Does {} know about {}? Answer yes or no."]
    KNOWLEDGE_SECOND_ORDER_Q = [
        "What does {} think about {}'s belief on {}? (knows about it / does not know about it)"
    ]

    FACTUAL_QUESTIONS_ABOUT_EVENTS = {
        FullBeliefTracker.CONTAINER_LOCATION: {
            "memory": "In which container was the {} at the beginning?",
            "ground_truth": "In which container is the {} now?",
            "memory_before_event": "In which container was the {} before {}?",  # FIXME to be added!
        },
        FullBeliefTracker.ROOM_LOCATION: {
            "memory": "In which room was the {} at the beginning?",
            "ground_truth": "In which room is the {} now?",
            "memory_before_event": "In which room was the {} before {}?",  # FIXME to be added!
        },
    }

    def __init__(self, belief_tracker):
        self.belief_tracker = belief_tracker

        self.relation_types_already_covered_first_order = set()
        self.relation_types_already_covered_second_order = set()

        self.reset_questions_dict_cache()

    def reset_questions_dict_cache(self):
        self.questions_dict = {1: [], 2: []}
        self.questions_dict_cache_type = None

    def _iterate_first_order_tracked_beliefs(self):
        result = []
        for entity in self.belief_tracker.first_order_beliefs:
            for thing in self.belief_tracker.first_order_beliefs[entity].keys():
                result.append((entity, thing))
        return result

    def _iterate_second_order_tracked_beliefs(self):
        result = []
        for entity1 in self.belief_tracker.second_order_beliefs:
            for entity2 in self.belief_tracker.second_order_beliefs[entity1].keys():
                for thing in self.belief_tracker.second_order_beliefs[entity1][
                    entity2
                ].keys():
                    result.append((entity1, entity2, thing))
        return result

    def _mark_questions_as_interesting(self):
        # mark questions as interesting if not all mentalizing answers are the same
        # it is important to consider both first-order and second-order combined here, due to the Ice Cream Truck test (June 26th change)
        answers_counter = {}
        answers_ids_mapping = {}
        for question_order, questions in self.questions_dict.items():
            for i, (_, answer, conditions, _) in enumerate(questions):
                assert (
                    not answer.startswith("not(") if isinstance(answer, str) else True
                )
                assert (
                    all(not a.startswith("not(") for a in answer)
                    if isinstance(answer, tuple) and isinstance(answer[0], str)
                    else True
                )  # V1
                assert (
                    all(not e.startswith("not(") for a in answer for e in a)
                    if isinstance(answer, tuple) and isinstance(answer[0], tuple)
                    else True
                )  # V1
                entities, thing, prop = conditions

                prop_wo_expansion = prop
                prop_wo_expansion = (
                    prop_wo_expansion[: -len("-False")]
                    if prop_wo_expansion.endswith("-False")
                    else prop_wo_expansion
                )
                prop_wo_expansion = (
                    prop_wo_expansion[: -len("-True")]
                    if prop_wo_expansion.endswith("-True")
                    else prop_wo_expansion
                )
                if (thing, prop_wo_expansion) not in answers_counter:
                    answers_counter[(thing, prop_wo_expansion)] = collections.Counter()
                    answers_ids_mapping[(thing, prop_wo_expansion)] = []

                # this handles that first & second order knowledge questions have different wording
                simplified_answer = (
                    "yes"
                    if prop_wo_expansion == FullBeliefTracker.KNOWLEDGE
                    and answer == "knows about it"
                    else answer
                )
                simplified_answer = (
                    "no"
                    if prop_wo_expansion == FullBeliefTracker.KNOWLEDGE
                    and simplified_answer == "does not know about it"
                    else simplified_answer
                )

                answers_counter[(thing, prop_wo_expansion)][simplified_answer] += 1
                answers_ids_mapping[(thing, prop_wo_expansion)].append(
                    (question_order, i)
                )

        for thing, prop in answers_counter:
            for question_order, i in answers_ids_mapping[(thing, prop)]:
                self.questions_dict[question_order][i][-1] = [
                    len(answers_counter[(thing, prop)]) > 1
                ]

    def location_qs_first_order(self, expand_relation_type_info):
        for entity, thing in self._iterate_first_order_tracked_beliefs():
            if entity == thing:
                continue  # skipping questions like "Where does Anne think that Anne is?"
            if thing in self.belief_tracker.people:
                continue  # skipping questions like "Where does Anne think that Bob is?"; also

            # avoiding asking about containers' rooms by only asking about the property container_location and not room_location
            for relation_type in [FullBeliefTracker.CONTAINER_LOCATION]:
                container_location_ans = self.belief_tracker.first_order_beliefs[
                    entity
                ].get(thing, relation_type)
                world_state_ans = self.belief_tracker.world_state.get(
                    thing, relation_type
                )
                relation_type_expanded = (
                    f"{relation_type}-{world_state_ans == container_location_ans}"
                    if expand_relation_type_info
                    else relation_type
                )
                if (
                    container_location_ans is not None
                    and not container_location_ans.startswith("not(")
                ):
                    for question_template in (
                        self.BELIEF_LOCATION_FIRST_ORDER_QS
                        if relation_type == FullBeliefTracker.CONTAINER_LOCATION
                        else self.BELIEF_ROOM_LOCATION_FIRST_ORDER_QS
                    ):
                        self.questions_dict[1].append(
                            [
                                question_template.format(entity, thing),
                                container_location_ans,
                                ([entity], thing, relation_type_expanded),
                                [],  # extra info
                            ]
                        )

        self.relation_types_already_covered_first_order.add(
            FullBeliefTracker.CONTAINER_LOCATION
        )
        self.relation_types_already_covered_first_order.add(
            FullBeliefTracker.ROOM_LOCATION
        )  # intentionally skipped

    def location_qs_second_order(self, expand_relation_type_info):
        for entity1, entity2, thing in self._iterate_second_order_tracked_beliefs():
            if entity1 == entity2 or entity2 == thing:
                continue  # skipping questions like "Where does Anne think that Anne thinks the truck is?"
            if thing in self.belief_tracker.people:
                continue  # skipping questions like "Where does Anne think that Bob is?"

            for relation_type in [
                FullBeliefTracker.CONTAINER_LOCATION,
                FullBeliefTracker.ROOM_LOCATION,
            ]:
                container_location_ans = self.belief_tracker.second_order_beliefs[
                    entity1
                ][entity2].get(thing, relation_type)
                world_state_ans = self.belief_tracker.world_state.get(
                    thing, relation_type
                )
                relation_type_expanded = (
                    f"{relation_type}-{world_state_ans == container_location_ans}"
                    if expand_relation_type_info
                    else relation_type
                )
                if (
                    container_location_ans is not None
                    and not container_location_ans.startswith("not(")
                ):
                    for question_template in (
                        self.BELIEF_LOCATION_SECOND_ORDER_QS
                        if relation_type == FullBeliefTracker.CONTAINER_LOCATION
                        else self.BELIEF_ROOM_LOCATION_SECOND_ORDER_QS
                    ):
                        self.questions_dict[2].append(
                            [
                                question_template.format(entity1, entity2, thing),
                                container_location_ans,
                                ([entity1, entity2], thing, relation_type_expanded),
                                [],
                            ]
                        )

        self.relation_types_already_covered_second_order.add(
            FullBeliefTracker.CONTAINER_LOCATION
        )
        self.relation_types_already_covered_second_order.add(
            FullBeliefTracker.ROOM_LOCATION
        )

    def all_communicated_abstract_knowledge(self):
        all_things = [
            self.belief_tracker.first_order_beliefs[entity].get(
                "", FullBeliefTracker.KNOWLEDGE
            )
            for entity in self.belief_tracker.first_order_beliefs
        ]
        all_things = [e for e in all_things if e]
        all_things = set([elem for list_things in all_things for elem in list_things])
        return all_things

    def abstract_knowledge_qs_first_order(self, all_things, expand_relation_type_info):
        for entity in self.belief_tracker.first_order_beliefs:
            known_things = self.belief_tracker.first_order_beliefs[entity].get(
                "", FullBeliefTracker.KNOWLEDGE
            )
            known_things = known_things if known_things else []

            for thing in all_things:
                if entity == thing:
                    continue

                belief_ans = "yes" if thing in known_things else "no"

                relation_type = FullBeliefTracker.KNOWLEDGE
                relation_type_expanded = (
                    f"{relation_type}-{True}"
                    if expand_relation_type_info
                    else relation_type
                )

                for question_template in self.KNOWLEDGE_FIRST_ORDER_Q:
                    self.questions_dict[1].append(
                        [
                            question_template.format(entity, thing),
                            belief_ans,
                            ([entity], thing, relation_type_expanded),
                            [],  # extra info
                        ]
                    )

        self.relation_types_already_covered_first_order.add(FullBeliefTracker.KNOWLEDGE)
        self.relation_types_already_covered_first_order.add(FullBeliefTracker.KNOWLEDGE)

    def abstract_knowledge_qs_second_order(self, all_things, expand_relation_type_info):
        for entity1 in self.belief_tracker.second_order_beliefs:
            for entity2 in self.belief_tracker.second_order_beliefs[entity1].keys():
                if entity1 == entity2:
                    continue  # skipping questions like "Where does Anne think that Anne thinks the truck is?"

                known_things = self.belief_tracker.second_order_beliefs[entity1][
                    entity2
                ].get("", FullBeliefTracker.KNOWLEDGE)
                known_things = known_things if known_things else []

                known_things_entity2 = self.belief_tracker.first_order_beliefs[
                    entity2
                ].get("", FullBeliefTracker.KNOWLEDGE)
                known_things_entity2 = (
                    known_things_entity2 if known_things_entity2 else []
                )
                for thing in all_things:
                    belief_ans = (
                        "knows about it"
                        if thing in known_things
                        else "does not know about it"
                    )
                    world_state_ans = (
                        "knows about it"
                        if thing in known_things_entity2
                        else "does not know about it"
                    )

                    relation_type = FullBeliefTracker.KNOWLEDGE
                    relation_type_expanded = (
                        f"{relation_type}-{world_state_ans == belief_ans}"
                        if expand_relation_type_info
                        else relation_type
                    )

                    for question_template in self.KNOWLEDGE_SECOND_ORDER_Q:
                        self.questions_dict[2].append(
                            [
                                question_template.format(entity1, entity2, thing),
                                belief_ans,
                                ([entity1, entity2], thing, relation_type_expanded),
                                [],  # evaluation function
                            ]
                        )

        self.relation_types_already_covered_second_order.add(
            FullBeliefTracker.KNOWLEDGE
        )
        self.relation_types_already_covered_second_order.add(
            FullBeliefTracker.KNOWLEDGE
        )

    def _get_remaining_believed_things(self):
        # get all properties that are believed by someone
        # first-order knowledge is (John -> apple -> salt -> is salted). Get all the properties (apple, is salted) and ask about them
        believed_things = {}
        for entity, thing in self._iterate_first_order_tracked_beliefs():
            believed_things[(entity, thing)] = set([])
            for relation_type, prop_value in (
                self.belief_tracker.first_order_beliefs[entity]
                .get_all_properties(thing)
                .items()
            ):
                if relation_type in self.relation_types_already_covered_first_order:
                    continue
                believed_things[(entity, thing)].add((relation_type, prop_value))
        for entity, entity2, thing in self._iterate_second_order_tracked_beliefs():
            believed_things[(entity2, thing)] = set([])
            for relation_type, prop_value in (
                self.belief_tracker.second_order_beliefs[entity][entity2]
                .get_all_properties(thing)
                .items()
            ):
                if relation_type in self.relation_types_already_covered_second_order:
                    continue
                believed_things[(entity2, thing)].add((relation_type, prop_value))
        return believed_things

    def general_belief_qs_first_order(self, believed_things):
        """
        - How would a person describe the current state of the object? This requires double checking if it is truly a significant change.
        - Binary questions about the object state. This requires changing strategy because rn we only ask about the current state of affairs, so all answers would be "yes". We could sample [...]
        - FOR FALSE BELIEFS ONLY: If they wanted to interact with the object, what would they think of doing? Check that this is consistent with their false belief, and not the real world state.
        """

        for entity, thing in self._iterate_first_order_tracked_beliefs():
            if entity == thing:
                continue
            for relation_type, prop_value in believed_things[(entity, thing)]:
                assert isinstance(prop_value, str)
                believed_by_entity = self.belief_tracker.first_order_beliefs[
                    entity
                ].get(
                    thing, relation_type
                ) is not None and prop_value in self.belief_tracker.first_order_beliefs[
                    entity
                ].get(
                    thing, relation_type
                )
                for question_template in self.BINARY_BELIEF_FIRST_ORDER_QS:
                    self.questions_dict[1].append(
                        [
                            question_template.format(entity, thing, prop_value),
                            "yes" if believed_by_entity else "no",
                            ([entity], thing, relation_type),
                            [],  # extra info
                        ]
                    )

    def general_belief_qs_second_order(self, believed_things):
        for entity1, entity2, thing in self._iterate_second_order_tracked_beliefs():
            if entity1 == entity2 or entity2 == thing or entity1 == thing:
                continue
            for relation_type, prop_value in believed_things[(entity2, thing)]:
                assert isinstance(prop_value, str)
                believed_by_entity = self.belief_tracker.second_order_beliefs[entity1][
                    entity2
                ].get(
                    thing, relation_type
                ) is not None and prop_value in self.belief_tracker.second_order_beliefs[
                    entity1
                ][
                    entity2
                ].get(
                    thing, relation_type
                )
                for question_template in self.BINARY_BELIEF_SECOND_ORDER_QS:
                    self.questions_dict[2].append(
                        [
                            question_template.format(
                                entity1, entity2, thing, prop_value
                            ),
                            "yes" if believed_by_entity else "no",
                            ([entity1, entity2], thing, relation_type),
                            [],  # extra info
                        ]
                    )

    def generate_factual_questions(self, function_list):
        """
        Factual (memory & ground truth) questions, should be used solely as training data.

        We record the trajectory of how an attribute changes over time. We can then ask "where was X before Y happened?"
        We ignore general <state_update> as they are hard to phrase correctly.
        """

        # 1. Compute all intermediate belief trackers
        belief_tracker = FullBeliefTracker()
        belief_tracker_list = []
        for f in function_list:
            a = f(belief_tracker)
            belief_tracker_list.append(copy.deepcopy(belief_tracker))
            assert a
        assert self.belief_tracker.story_script == belief_tracker.story_script

        # 2. Create trajectories of object changes
        trajectory_object_states = {}
        final_belief_tracker = belief_tracker
        for obj in final_belief_tracker.objects:
            trajectory_object_states[obj] = {}
            for i, bt in enumerate(belief_tracker_list):
                for k, v in bt.world_state.get_all_properties(obj).items():
                    if k not in trajectory_object_states[obj]:
                        trajectory_object_states[obj][k] = [(v, i)]
                    elif trajectory_object_states[obj][k][-1][0] != v:
                        trajectory_object_states[obj][k].append((v, i))

        # 3. keep only properties that actually changed over time
        # We ask about the beginning, ask about state before/after some event
        questions_list = []
        for obj in trajectory_object_states:
            for k in trajectory_object_states[obj]:
                if k.startswith("<state_update>"):
                    continue
                if len(trajectory_object_states[obj][k]) <= 1:
                    continue
                beginning = trajectory_object_states[obj][k][0][0]
                ending = trajectory_object_states[obj][k][-1][0]
                for question_template, question_type, answer in [
                    (
                        self.FACTUAL_QUESTIONS_ABOUT_EVENTS[k]["memory"].format(obj),
                        "memory",
                        beginning,
                    ),
                    (
                        self.FACTUAL_QUESTIONS_ABOUT_EVENTS[k]["ground_truth"].format(
                            obj
                        ),
                        "ground_truth",
                        ending,
                    ),
                ]:
                    if answer.startswith("not("):
                        continue
                    relation_type = f"{question_type}-{k}"
                    questions_list.append(
                        [
                            question_template.format(obj),
                            answer,
                            (None, obj, relation_type),
                            [True],  # extra info
                        ]
                    )

                if len(trajectory_object_states[obj][k]) <= 2:
                    continue
                for event_idx in range(1, len(trajectory_object_states[obj][k])):
                    # where was the object before event_idx happened?
                    _, story_idx = trajectory_object_states[obj][k][event_idx]
                    answer, _ = trajectory_object_states[obj][k][event_idx - 1]
                    story_element = (
                        belief_tracker_list[story_idx]
                        .story_script[-1]
                        .strip(".")
                        .split(", which is also located in")[0]
                    )

                    question_type = "memory_before_event"
                    question_template = self.FACTUAL_QUESTIONS_ABOUT_EVENTS[k][
                        question_type
                    ].format(obj, story_element)
                    if answer.startswith("not("):
                        continue

                    relation_type = f"{question_type}-{k}"
                    questions_list.append(
                        [
                            question_template.format(obj),
                            answer,
                            (None, obj, relation_type),
                            [True],  # extra info
                        ]
                    )
        return questions_list

    def main(self, nth_reasoning_order, expand_relation_type_info=False):
        assert nth_reasoning_order in [1, 2]
        if (
            self.questions_dict[nth_reasoning_order] != []
            and self.questions_dict_cache_type == expand_relation_type_info
        ):
            return self.questions_dict[nth_reasoning_order]
        if (
            self.questions_dict_cache_type in [True, False]
            and self.questions_dict_cache_type != expand_relation_type_info
        ):
            self.reset_questions_dict_cache()

        self.location_qs_first_order(expand_relation_type_info)
        self.location_qs_second_order(expand_relation_type_info)

        all_abstract_knowledge = self.all_communicated_abstract_knowledge()
        self.abstract_knowledge_qs_first_order(
            all_abstract_knowledge, expand_relation_type_info
        )
        self.abstract_knowledge_qs_second_order(
            all_abstract_knowledge, expand_relation_type_info
        )

        believed_things = self._get_remaining_believed_things()
        self.general_belief_qs_first_order(
            believed_things
        )  # FIXME why not expand_relation_type_info ?
        self.general_belief_qs_second_order(
            believed_things
        )  # FIXME why not expand_relation_type_info ?

        self._mark_questions_as_interesting()

        # add evaluation function
        for k in self.questions_dict:
            for j, _ in enumerate(self.questions_dict[k]):
                self.questions_dict[k][j][-1].append(lambda x, y: x in y)
        self.questions_dict_cache_type = expand_relation_type_info
        return self.questions_dict[nth_reasoning_order]

    def is_false_belief_story(self):
        # Check if the story requires theory of mind
        return len([e for n in [1, 2] for e in self.main(n) if e[-1][0]]) > 0

    def is_false_belief_story_restricted_to_first_order_qs(self):
        return len([e for n in [1] for e in self.main(n) if e[-1][0]]) > 0


class ChainOfThoughtGenerator:
    """
    - Idea #2, on giving reasons of why something is or is not relevant through counterfactuals:
        > add X as witness or not witness. If the answer doesn't change, then the action is irrelevant to the question
        > add X as witness and the answer changes => they didn't witness in the original
        > add X as not witness and the answer changes => they were a witness in the original
    """

    COT_START_STR = "Let's think step by step by iteratively analyzing each sentence where the answer to this question would change."

    ACTION_DOES_NOT_AFFECT_ANSWER_STR = (
        "Therefore, this action does not affect the answer to the question"
    )
    ACTION_AFFECTS_ANSWER_STR = (
        "Therefore, this action may affect the answer to the question"
    )

    def __init__(self):
        pass

    def _eval_with_peeking_and_distracted(
        self, bt2, fn2_id, entity_to_add, peeking=False, distracted=False
    ):
        assert (
            not peeking or not distracted
        ), "You cannot peek and be distracted all at once."
        assert (
            "peeking" not in fn2_id[0]
        ), "We currently do not support CoT of asymmetric stories."

        tmp = {}
        if peeking:
            tmp = {"people_peeking": [entity_to_add], "people_distracted": []}
        if distracted:
            tmp = {"people_peeking": [], "people_distracted": [entity_to_add]}

        output_flag = True
        if fn2_id[0] == "move_object_container":
            output_flag = bt2.move_object_container(*fn2_id[1:], **tmp)
        elif fn2_id[0] == "move_object_container":
            output_flag = bt2.move_object_room(*fn2_id[1:], **tmp)
        elif fn2_id[0] == "enter_room":
            output_flag = bt2.enter_room(*fn2_id[1:], **tmp)
        elif fn2_id[0] == "leave_room":
            output_flag = bt2.leave_room(*fn2_id[1:], **tmp)
        elif fn2_id[0] == "broadcast_communication":
            output_flag = bt2.broadcast_communication(*fn2_id[1:], **tmp)
        elif fn2_id[0] == "private_communication":
            output_flag = bt2.private_communication(*fn2_id[1:], **tmp)
        else:
            assert False, "Action not supported."
        return bt2, output_flag

    def person_involvement_type(
        self, person, answers_all_settings_peeking, answers_all_settings_distracted
    ):
        """
        How was a person involved in an action?

        - If we tried adding the person as a peeker and a distracted person and both failed (=False), then the person performed the action.
        - ElseIf we tried adding the person as a peeker and it failed, then the person was already a witness.
        - Else: he was not a witness
        """
        assert not (
            answers_all_settings_peeking[1] and answers_all_settings_distracted[1]
        ), "Schroedinger person?"

        # this ordering seems important, because if they were the one making the action, both cases will equal to False (and they were a witness)
        if (
            answers_all_settings_peeking[1] == False
            and answers_all_settings_distracted[1] == False
        ):
            return f"{person} performed the action"
        if answers_all_settings_peeking[1] == False:
            return f"{person} was a witness"
        if answers_all_settings_distracted[1] == False:
            return f"{person} was not a witness"
        assert False

    def get_each_characters_involvement_in_the_step_str(
        self, question_entities, answers_all_settings
    ):
        reason_list = []
        for q_idx, entity in enumerate(question_entities):
            reason = self.person_involvement_type(
                question_entities[q_idx],
                answers_all_settings[1 + 2 * q_idx],
                answers_all_settings[2 + 2 * q_idx],
            )
            reason_list.append(reason)
        return ". ".join(reason_list)

    def action_affects_final_answer_str(self, current_answer, new_current_answer):
        return (
            self.ACTION_DOES_NOT_AFFECT_ANSWER_STR
            if current_answer == new_current_answer
            else self.ACTION_AFFECTS_ANSWER_STR
        )

    def get_answer_to_question_in_this_setting(
        self, bt, question_to_analyze, nth_reasoning_order
    ):
        filtered = [
            (question, answer, entities, metadata)
            for question, answer, entities, metadata in QuestionGenerator(bt).main(
                nth_reasoning_order=nth_reasoning_order
            )
            if question_to_analyze == question
        ]
        assert len(filtered) <= 1
        return filtered[0][1] if filtered else None

    def _current_answer_clean_text(self, current_answer):
        if current_answer is None:
            return "unknown"
        if isinstance(current_answer, str):
            return f"the {current_answer}"
        assert False

    def main(
        self,
        function_list_identifiers,
        question,
        entities,
        expected_answer,
        nth_reasoning_order=1,
    ):
        if any(
            "peeking" in identifier[0]
            or "<state_update>" in identifier[0]
            or "location_declaration" in identifier[0]
            for identifier in function_list_identifiers
        ):
            return None

        partial_cot = [self.COT_START_STR]
        entities = entities[0]  # actual characters involved in the question

        belief_tracker = FullBeliefTracker()
        current_answer = None
        for i, f_id1 in enumerate(function_list_identifiers):
            len_old_story_script = len(belief_tracker.story_script)

            bt_orig_updated = None
            answers_all_settings = []
            for j in range(1 + 2 * len(entities)):
                bt = copy.deepcopy(belief_tracker)
                f_id = copy.deepcopy(f_id1)

                # 0 = original ; 1 = add person as peeker ; 2 = add person as distracted
                if j == 0:
                    bt, output_flag = self._eval_with_peeking_and_distracted(
                        bt, f_id, entities[0]
                    )
                    bt_orig_updated = bt
                if j > 0 and j % 2 == 1:
                    bt, output_flag = self._eval_with_peeking_and_distracted(
                        bt, f_id, entities[(j - 1) // 2], peeking=True
                    )
                if j > 0 and j % 2 == 0:
                    bt, output_flag = self._eval_with_peeking_and_distracted(
                        bt, f_id, entities[(j - 1) // 2], distracted=True
                    )

                answers_all_settings.append(
                    (
                        self.get_answer_to_question_in_this_setting(
                            bt, question, nth_reasoning_order
                        ),
                        output_flag,
                    )
                )

            belief_tracker = copy.deepcopy(bt_orig_updated)
            action = " ".join(belief_tracker.story_script[len_old_story_script:]).strip(
                "."
            )
            new_current_answer, output_flag = answers_all_settings[0]
            assert output_flag

            partial_cot.append(
                ". ".join(
                    [
                        f"The {'next' if len(partial_cot) > 1 else 'first'} action is ``{action}''",
                        self.get_each_characters_involvement_in_the_step_str(
                            entities, answers_all_settings
                        ),
                        self.action_affects_final_answer_str(
                            current_answer, new_current_answer
                        ),
                        f"At this point the answer would be {self._current_answer_clean_text(new_current_answer)}.",
                    ]
                )
            )
            current_answer = new_current_answer

        partial_cot.append(
            f"Since this is the end of the story, the final answer is: {expected_answer}."
        )
        assert current_answer == expected_answer, (current_answer, expected_answer)
        return "\n".join(partial_cot)
