"""Check that setup errors cannot launch paid cloud work."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import modal_app


class SetupGuardsTest(unittest.TestCase):
    def test_wrong_workspace_stops_before_secret_access_or_app_start(self):
        workspace = SimpleNamespace(name="another-workspace", hydrate=Mock())
        with (
            patch.object(modal_app.modal.Workspace, "from_context", return_value=workspace),
            patch.object(modal_app, "wandb_secret") as secret,
            patch.object(modal_app, "app") as app,
        ):
            with self.assertRaisesRegex(RuntimeError, "Expected Modal workspace"):
                modal_app.main()
            secret.hydrate.assert_not_called()
            app.run.assert_not_called()

    def test_missing_secret_stops_before_app_start(self):
        workspace = SimpleNamespace(name=modal_app.WORKSPACE, hydrate=Mock())
        with (
            patch.object(modal_app.modal.Workspace, "from_context", return_value=workspace),
            patch.object(modal_app, "wandb_secret") as secret,
            patch.object(modal_app, "app") as app,
        ):
            secret.hydrate.side_effect = RuntimeError("Missing W&B secret")
            with self.assertRaisesRegex(RuntimeError, "Missing W&B secret"):
                modal_app.main()
            app.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
