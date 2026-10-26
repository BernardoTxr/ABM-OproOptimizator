import real_lob_inference


def test_public_api_only_names_available_objects() -> None:
    missing = [
        name for name in real_lob_inference.__all__
        if not hasattr(real_lob_inference, name)
    ]

    assert missing == []
