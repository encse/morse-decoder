from dataclasses import asdict
import json
from pathlib import Path

from generate_training_samples import main
from morse_timing.audio_dataset import Stage1DatasetConfig


def test_generates_matching_wav_png_and_json_files(tmp_path: Path) -> None:
    model_json = tmp_path / "model.json"
    output_directory = tmp_path / "preview"
    config = Stage1DatasetConfig(
        min_characters=2,
        max_characters=2,
        noise_only_probability=0.0,
        apply_input_filter=False,
    )
    model_json.write_text(
        json.dumps(
            {
                "dataset_config": asdict(config),
                "experiment": {"seed": 123},
            }
        ),
        encoding="utf-8",
    )

    main(
        [
            str(model_json),
            "2",
            "--output-directory",
            str(output_directory),
        ]
    )

    for index in range(2):
        stem = f"sample-{index:05d}"
        assert (output_directory / f"{stem}.wav").is_file()
        assert (output_directory / f"{stem}.png").is_file()
        metadata = json.loads(
            (output_directory / f"{stem}.json").read_text(encoding="utf-8")
        )
        assert metadata["seed"] == 123
        assert metadata["index"] == index
        assert metadata["wav"] == f"{stem}.wav"
        assert metadata["png"] == f"{stem}.png"
