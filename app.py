"""Hugging Face Spaces entry point for Digital Twin Tumor Response Assessment."""
import os
import sys

# Ensure src/ is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Generate demo data if not present (one-time cold start cost)
demo_path = os.path.join(os.path.dirname(__file__), ".cache", "demo.db")
os.makedirs(os.path.dirname(demo_path), exist_ok=True)
if not os.path.exists(demo_path):
    from digital_twin_tumor.data.synthetic import generate_all_demo_data

    generate_all_demo_data(db_path=demo_path, seed=42, verbose=True)
os.environ["DTT_DEMO_DB"] = demo_path

from digital_twin_tumor.ui import create_app  # noqa: E402

app = create_app()
app.launch(server_name="0.0.0.0", server_port=7860)
