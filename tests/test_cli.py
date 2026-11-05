import pytest

from music_recommender.cli import main


@pytest.fixture(scope="module")
def csv_path(raw_tracks, tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "tracks.csv"
    raw_tracks.to_csv(path, index=False)
    return path


def run(capsys, csv_path, tmp_path, *args):
    code = main(["--data", str(csv_path), "--model", str(tmp_path / "none.joblib"), *args])
    return code, capsys.readouterr()


def test_search(capsys, csv_path, tmp_path):
    code, out = run(capsys, csv_path, tmp_path, "search", "bohemian")
    assert code == 0 and "Queen" in out.out


def test_similar_and_playlist(capsys, csv_path, tmp_path):
    code, out = run(capsys, csv_path, tmp_path, "similar", "Song 10", "-k", "5")
    assert code == 0 and "Seed: Song 10" in out.out
    code, out = run(
        capsys,
        csv_path,
        tmp_path,
        "playlist",
        "--like",
        "Song 1",
        "--like",
        "Song 2",
        "--dislike",
        "Song 3",
        "-k",
        "5",
        "--diversity",
        "0.3",
    )
    assert code == 0 and "rank" in out.out


def test_mood(capsys, csv_path, tmp_path):
    code, out = run(capsys, csv_path, tmp_path, "mood", "energy=0.9", "tempo=128", "-k", "3")
    assert code == 0


def test_not_found_exit_code(capsys, csv_path, tmp_path):
    code, out = run(capsys, csv_path, tmp_path, "similar", "zzzz qqqq nothing")
    assert code == 2 and "No track found" in out.err


def test_build_then_load(capsys, csv_path, tmp_path):
    model = tmp_path / "m.joblib"
    assert main(["--data", str(csv_path), "build", "--out", str(model)]) == 0
    assert model.exists()
    assert main(["--data", str(csv_path), "--model", str(model), "search", "Song 4"]) == 0


def test_import_and_playlist_from_file(capsys, csv_path, tmp_path, raw_tracks):
    pasted = tmp_path / "pasted.txt"
    song = raw_tracks.loc[raw_tracks["name"] == "Song 3", "track_id"].iloc[0]
    lines = [f"https://open.spotify.com/track/{song}", "Stand by Me - Ben E. King", "nope - nobody"]
    pasted.write_text("\n".join(lines), encoding="utf-8")
    code, out = run(capsys, csv_path, tmp_path, "import", str(pasted))
    assert code == 0 and "Matched 2 of 3" in out.out and "no match" in out.out
    code, out = run(capsys, csv_path, tmp_path, "playlist", "--from-file", str(pasted), "-k", "4")
    assert code == 0 and "Matched 2 of 3" in out.err and "rank" in out.out
    code, out = run(capsys, csv_path, tmp_path, "playlist")
    assert code == 1 and "--like" in out.err
