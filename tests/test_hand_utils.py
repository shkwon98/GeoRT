import numpy as np

from geort.utils.hand_utils import check_contact


class Contact:
    def __init__(self, actor0, actor1, impulse=1.0):
        self.actor0 = actor0
        self.actor1 = actor1
        self.points = [type("Point", (), {"impulse": np.array([impulse, 0.0, 0.0])})()]


class Scene:
    def __init__(self, *contacts):
        self.contacts = contacts

    def get_contacts(self):
        return self.contacts


def test_check_contact_requires_both_contact_actors_in_sets():
    hand_a, hand_b, ground = object(), object(), object()

    assert check_contact(Scene(Contact(hand_a, hand_b)), [hand_a, hand_b], [hand_a, hand_b])
    assert not check_contact(Scene(Contact(hand_a, ground)), [hand_a, hand_b], [hand_a, hand_b])


def test_check_contact_accepts_reversed_actor_order():
    hand, obstacle = object(), object()

    assert check_contact(Scene(Contact(obstacle, hand)), [hand], [obstacle])


def test_check_contact_filters_low_impulse_contacts():
    hand_a, hand_b = object(), object()

    assert not check_contact(Scene(Contact(hand_a, hand_b, impulse=1e-12)), [hand_a], [hand_b])
