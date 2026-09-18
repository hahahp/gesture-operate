import unittest

from desktop_core import ActionKind, DesktopGestureEngine, HandObservation, ScreenArea


def hand(x=0.5, y=0.5, index=1.0, middle=1.0, gesture="", landmarks=()):
    return HandObservation(x, y, index, middle, gesture, landmarks=landmarks)


class DesktopGestureEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = DesktopGestureEngine(ScreenArea(-1280, 0, 3200, 1080), active=True)

    def test_maps_camera_to_virtual_desktop_edges(self):
        self.assertEqual(self.engine.map_position((0.07, 0.07)), (-1280, 0))
        right, bottom = self.engine.map_position((0.93, 0.93))
        self.assertAlmostEqual(right, 1919)
        self.assertAlmostEqual(bottom, 1079)

    def test_maps_full_hand_skeleton_and_aligns_index_tip(self):
        landmarks = tuple((0.5, 0.5) for _ in range(21))
        frame = self.engine.update(hand(landmarks=landmarks), 0.0)
        self.assertEqual(len(frame.landmarks), 21)
        self.assertEqual(frame.landmarks[8], frame.position)

    def test_short_index_pinch_clicks_on_release(self):
        self.engine.update(hand(), 0.0)
        self.engine.update(hand(index=0.2), 0.1)
        frame = self.engine.update(hand(), 0.2)
        self.assertEqual([action.kind for action in frame.actions], [ActionKind.PRIMARY_CLICK])
        self.assertEqual(frame.feedback, "primary")

    def test_two_short_pinches_report_double_feedback(self):
        self.engine.update(hand(index=0.2), 0.0)
        self.engine.update(hand(), 0.1)
        self.engine.update(hand(index=0.2), 0.2)
        frame = self.engine.update(hand(), 0.3)
        self.assertEqual(frame.feedback, "double")

    def test_moving_while_pinched_drags(self):
        self.engine.update(hand(), 0.0)
        self.engine.update(hand(index=0.2), 0.1)
        moving = self.engine.update(hand(x=0.75, index=0.2), 0.2)
        kinds = [action.kind for action in moving.actions]
        self.assertEqual(kinds, [ActionKind.DRAG_BEGIN, ActionKind.DRAG_MOVE])
        released = self.engine.update(hand(x=0.75), 0.3)
        self.assertEqual([action.kind for action in released.actions], [ActionKind.DRAG_END])

    def test_middle_pinch_is_secondary_action(self):
        self.engine.update(hand(middle=0.25), 0.0)
        frame = self.engine.update(hand(), 0.1)
        self.assertEqual([action.kind for action in frame.actions], [ActionKind.SECONDARY_CLICK])

    def test_simultaneous_pinch_emits_only_closer_finger_action(self):
        self.engine.update(hand(index=0.2, middle=0.3), 0.0)
        frame = self.engine.update(hand(), 0.1)
        self.assertEqual([action.kind for action in frame.actions], [ActionKind.PRIMARY_CLICK])

    def test_open_palm_hold_toggles_active_once(self):
        self.engine.update(hand(gesture="Open_Palm"), 0.0)
        frame = self.engine.update(hand(gesture="Open_Palm"), 1.1)
        self.assertFalse(frame.active)
        self.assertEqual(
            [action.kind for action in frame.actions],
            [ActionKind.ACTIVE_CHANGED],
        )
        held = self.engine.update(hand(gesture="Open_Palm"), 2.2)
        self.assertFalse(held.active)
        self.assertEqual(held.actions, ())

    def test_lost_hand_releases_active_drag(self):
        self.engine.update(hand(index=0.2), 0.0)
        self.engine.update(hand(x=0.8, index=0.2), 0.1)
        frame = self.engine.update(None, 0.2)
        self.assertEqual([action.kind for action in frame.actions], [ActionKind.DRAG_END])


if __name__ == "__main__":
    unittest.main()
