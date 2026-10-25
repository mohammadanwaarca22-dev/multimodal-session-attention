import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class DashboardTests(unittest.TestCase):
    def test_dashboard_loads_with_empty_outputs(self):
        app = AppTest.from_file(APP_PATH).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Multimodal Session Dashboard")
        self.assertTrue(
            any("No activity timeline is available" in item.value for item in app.info)
        )

    def test_synthetic_demo_renders_activity_chart_and_metrics(self):
        app = AppTest.from_file(APP_PATH).run()
        app.checkbox[0].set_value(True).run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Synthetic demonstration data is active" in item.value for item in app.info))
        self.assertGreaterEqual(len(app.metric), 3)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 3)
        self.assertGreaterEqual(len(app.dataframe), 6)
        self.assertGreaterEqual(len(app.expander), 2)


if __name__ == "__main__":
    unittest.main()
