"""CPU-only decoded audio tail/codec-trim contracts for strict paired media."""

import pytest

from test_h3_evaluate import evaluation, valid_probe


def test_missing_decoded_audio_tail_is_rejected_even_with_matching_declared_durations():
    probe = valid_probe()
    probe["frames"].pop()
    with pytest.raises(ValueError, match="decoded audio sample duration"):
        evaluation.media_contract(probe)


def test_extra_contiguous_decoded_audio_tail_is_rejected():
    probe = valid_probe()
    probe["frames"].append(dict(probe["frames"][-1], best_effort_timestamp=4000))
    with pytest.raises(ValueError, match="decoded audio sample duration"):
        evaluation.media_contract(probe)


def test_decoder_applied_partial_last_frame_is_accepted_without_retrimming():
    probe = valid_probe()
    # nb_samples/duration describe the already-trimmed decoded frames. The
    # final frame need not have the encoder's usual packet sample count.
    probe["frames"][3].update(nb_samples=3072, duration=3072)
    probe["frames"][4].update(best_effort_timestamp=3072, nb_samples=928, duration=928)
    contract = evaluation.media_contract(probe)
    assert contract["decoded_audio_samples"] == 4000
    assert contract["audio_frame_samples"] == [3072, 928]
    assert contract["duration_seconds"] == "1/8"


@pytest.mark.parametrize("kind", ["Skip samples", "Skip Samples"])
def test_unresolved_frame_skip_metadata_is_explicitly_unsupported(kind):
    probe = valid_probe()
    probe["frames"][3]["side_data_list"] = [
        {"side_data_type": kind, "skip_samples": 100, "discard_padding": 0}
    ]
    with pytest.raises(ValueError, match="unresolved decoded audio skip/discard"):
        evaluation.media_contract(probe)
