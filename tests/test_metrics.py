from evaluation.metrics import classification, detection, ocr


def test_levenshtein_and_error_rates():
    assert ocr.levenshtein("kitten", "sitting") == 3
    assert ocr.character_error_rate("2026", "2O26") == 0.25
    assert ocr.word_error_rate("EXP 12 09 2026", "EXP 12 09 2026") == 0.0


def test_ocr_evaluate_scores_date_parsing():
    samples = [
        {"id": "a", "ocr_text": "EXP 12/09/2026", "truth_text": "EXP 12/09/2026",
         "truth_date": "2026-09-12"},
        {"id": "b", "ocr_text": "BB 20260818", "truth_text": "BB 20260818",
         "truth_date": "2026-08-18"},
        {"id": "c", "ocr_text": "smudged", "truth_text": "", "truth_date": None},
    ]
    result = ocr.evaluate(samples)
    assert result["dates_claimed"] == 2
    assert result["dates_correct"] == 2
    assert result["date_parsing_precision"] == 1.0
    assert result["date_parsing_recall"] == 1.0


def test_classification_metrics_without_sklearn_path():
    y_true = ["fresh", "fresh", "ripening", "spoiled"]
    y_pred = ["fresh", "ripening", "ripening", "spoiled"]
    result = classification._manual_metrics(y_true, y_pred, ["fresh", "ripening", "spoiled"])
    assert result["top1_accuracy"] == 0.75
    assert result["confusion_matrix"][0] == [1, 1, 0]


def test_detection_fallback_precision_recall():
    predictions = [{"boxes": [[0.0, 0.0, 0.2, 0.2]], "labels": [1], "scores": [0.9]}]
    targets = [{"boxes": [[0.01, 0.01, 0.21, 0.21]], "labels": [1]}]
    result = detection._fallback_map(predictions, targets, iou_threshold=0.5)
    assert result["true_positives"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_latency_stats():
    stats = detection.latency_stats([10.0, 20.0, 30.0])
    assert stats["mean_ms"] == 20.0
    assert stats["median_ms"] == 20.0
    assert stats["count"] == 3
