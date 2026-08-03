import numpy as np

from geort.utils.hand_utils import check_contact


class Contact:
    def __init__(self, actor0, actor1, impulse=1.0):
        self.bodies = [actor0, actor1]
        self.points = [
            type("Point", (), {"impulse": np.array([impulse, 0.0, 0.0])})()]


class Scene:
    def __init__(self, *contacts):
        self.contacts = contacts

    def get_contacts(self):
        return self.contacts


def test_check_contact_requires_a_real_pair_and_impulse():
    hand_a, hand_b, ground = object(), object(), object()

    assert check_contact(Scene(Contact(hand_a, hand_b)), [
                         hand_a, hand_b], [hand_a, hand_b])
    assert check_contact(Scene(Contact(hand_b, hand_a)), [hand_a], [hand_b])
    assert not check_contact(Scene(Contact(hand_a, ground)), [
                             hand_a, hand_b], [hand_a, hand_b])
    assert not check_contact(
        Scene(Contact(hand_a, hand_b, impulse=1e-12)), [hand_a], [hand_b])
