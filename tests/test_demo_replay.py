import copy
import unittest
from demo.build_data import build_events


class Decoder:
    def decode(self, ids, **kwargs):
        return ["", "A", "A�", "A🙂", "A🙂B"][len(ids)]


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.row = {"token_ids": [1, 2, 3, 4], "text": "A🙂B", "seconds": 1,
            "events": [{"at_s": n / 10, "token_count": n, "token_ids": [n]}
                       for n in range(1, 5)]}

    def test_reconstructs_unicode_and_preserves_timestamps(self):
        events = build_events(self.row, Decoder())
        self.assertEqual([e["at_s"] for e in events], [.1, .2, .3, .4])
        self.assertEqual(events[2]["keep"], 1)
        self.assertEqual(events[3]["keep"], 3)
        text = ""
        for event in events:
            text = text.encode("utf-16-le")[:event["keep"] * 2].decode("utf-16-le") + event["append"]
        self.assertEqual(text, self.row["text"])

    def test_rejects_invalid_capture(self):
        for kind in ("time", "count", "text"):
            row = copy.deepcopy(self.row)
            if kind == "time":
                row["events"][2]["at_s"] = .05
            elif kind == "count":
                row["events"][0]["token_count"] = 99
            else:
                row["text"] = "invented"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                build_events(row, Decoder())
