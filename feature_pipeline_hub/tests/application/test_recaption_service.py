from feature_pipeline.application import recaption_service
from feature_pipeline.domain.models import DatasetSample, ImageMetrics


def _sample(sample_id: str, image_path: str, caption: str = "old caption") -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        image_path=image_path,
        caption=caption,
        original_caption=caption,
        metrics=ImageMetrics(
            width=512, height=512, aspect_ratio=1.0, format="PNG", phash="abcd1234"
        ),
    )


def _fake_runner(events):
    def run_recaption(image_paths, detailed, environment=None):
        run_recaption.calls.append({"paths": list(image_paths), "detailed": detailed})
        yield from events

    run_recaption.calls = []
    return run_recaption


def test_captions_are_matched_back_to_their_sample(monkeypatch):
    samples = [_sample("s1", "/data/a.png"), _sample("s2", "/data/b.png")]
    monkeypatch.setattr(
        recaption_service.recaption_runner,
        "run_recaption",
        _fake_runner(
            [
                {"event": "loaded", "device": "cuda", "seconds": 4.5},
                {"event": "caption", "path": "/data/b.png", "caption": "a blue square", "seconds": 2.0},
            ]
        ),
    )

    results = list(recaption_service.recaption_samples(samples, trigger_word=""))

    assert [r.kind for r in results] == ["loaded", "caption"]
    assert results[1].sample_id == "s2"
    assert results[1].caption == "a blue square"


def test_trigger_word_is_injected_into_ai_captions(monkeypatch):
    monkeypatch.setattr(
        recaption_service.recaption_runner,
        "run_recaption",
        _fake_runner([{"event": "caption", "path": "/data/a.png", "caption": "a red car"}]),
    )

    results = list(
        recaption_service.recaption_samples([_sample("s1", "/data/a.png")], trigger_word="sks_style")
    )

    assert results[0].caption == "sks_style, a red car"


def test_trigger_word_is_not_duplicated(monkeypatch):
    monkeypatch.setattr(
        recaption_service.recaption_runner,
        "run_recaption",
        _fake_runner(
            [{"event": "caption", "path": "/data/a.png", "caption": "sks_style, a red car"}]
        ),
    )

    results = list(
        recaption_service.recaption_samples([_sample("s1", "/data/a.png")], trigger_word="sks_style")
    )

    assert results[0].caption == "sks_style, a red car"


def test_a_failed_image_is_reported_without_aborting_the_batch(monkeypatch):
    samples = [_sample("s1", "/data/a.png"), _sample("s2", "/data/b.png")]
    monkeypatch.setattr(
        recaption_service.recaption_runner,
        "run_recaption",
        _fake_runner(
            [
                {"event": "error", "path": "/data/a.png", "message": "OSError: broken image"},
                {"event": "caption", "path": "/data/b.png", "caption": "a blue square"},
            ]
        ),
    )

    results = list(recaption_service.recaption_samples(samples, trigger_word=""))

    assert [r.kind for r in results] == ["error", "caption"]
    assert results[0].sample_id == "s1"
    assert "broken image" in results[0].message


def test_worker_failure_surfaces_as_a_failed_event(monkeypatch):
    monkeypatch.setattr(
        recaption_service.recaption_runner,
        "run_recaption",
        _fake_runner([{"event": "failed", "message": "exited with code 1"}]),
    )

    results = list(
        recaption_service.recaption_samples([_sample("s1", "/data/a.png")], trigger_word="")
    )

    assert results[0].kind == "failed"
    assert "exited with code 1" in results[0].message


def test_unknown_paths_are_ignored(monkeypatch):
    monkeypatch.setattr(
        recaption_service.recaption_runner,
        "run_recaption",
        _fake_runner([{"event": "caption", "path": "/data/ghost.png", "caption": "nothing"}]),
    )

    results = list(
        recaption_service.recaption_samples([_sample("s1", "/data/a.png")], trigger_word="")
    )

    assert results == []


def test_detailed_flag_reaches_the_runner(monkeypatch):
    runner = _fake_runner([])
    monkeypatch.setattr(recaption_service.recaption_runner, "run_recaption", runner)

    list(
        recaption_service.recaption_samples(
            [_sample("s1", "/data/a.png")], trigger_word="", detailed=True
        )
    )

    assert runner.calls[0]["detailed"] is True
    assert runner.calls[0]["paths"] == ["/data/a.png"]


def test_an_empty_batch_never_starts_the_worker(monkeypatch):
    runner = _fake_runner([])
    monkeypatch.setattr(recaption_service.recaption_runner, "run_recaption", runner)

    assert list(recaption_service.recaption_samples([], trigger_word="")) == []
    assert runner.calls == []
