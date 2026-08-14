"""Domain services. Heavy ML dependencies are imported lazily inside each
service so the API (and the test suite) boots without torch/ultralytics present.
"""
