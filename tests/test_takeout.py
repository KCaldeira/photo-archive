from pathlib import Path

from photosel.takeout import SidecarIndex, album_for_folder, parse_media_name, sidecar_key


def test_sidecar_key_forms():
    assert sidecar_key("IMG_1.JPG.json") == ("IMG_1.JPG", 0)
    assert sidecar_key("IMG_1.JPG(2).json") == ("IMG_1.JPG", 2)
    assert sidecar_key("IMG_1.JPG.supplemental-metadata.json") == ("IMG_1.JPG", 0)
    assert sidecar_key("IMG_1.JPG.supplemental-metadata(1).json") == ("IMG_1.JPG", 1)
    assert sidecar_key("IMG_1.JPG.supplemental-met.json") == ("IMG_1.JPG", 0)
    assert sidecar_key("IMG_1.JPG.s.json") == ("IMG_1.JPG", 0)
    assert sidecar_key("IMG_1.json") == ("IMG_1", 0)


def test_parse_media_name():
    m = parse_media_name("IMG_1-edited(1).jpg")
    assert (m.original, m.n, m.is_edited) == ("IMG_1.jpg", 1, True)
    m = parse_media_name("IMG_1(3).JPG")
    assert (m.original, m.n, m.is_edited) == ("IMG_1.JPG", 3, False)


def _idx(*names):
    return SidecarIndex([Path(n) for n in names])


def test_match_variants():
    idx = _idx("IMG_1.JPG.json", "IMG_1.JPG(1).json", "IMG_2.HEIC.supplemental-metadata.json", "IMG_3.json")
    assert idx.match("IMG_1.JPG").name == "IMG_1.JPG.json"
    assert idx.match("IMG_1(1).JPG").name == "IMG_1.JPG(1).json"
    assert idx.match("IMG_1-edited.JPG").name == "IMG_1.JPG.json"
    assert idx.match("IMG_2.HEIC").name == "IMG_2.HEIC.supplemental-metadata.json"
    assert idx.match("IMG_3.JPG").name == "IMG_3.json"
    assert idx.match("IMG_9.JPG") is None


def test_match_truncated_long_names():
    long = "A_really_long_file_name_from_some_messaging_app_2023.jpg"
    side = (long + ".supplemental-metadata")[:46] + ".json"
    assert len(side) == 51
    assert _idx(side).match(long).name == side
    old_side = long[:46] + ".json"
    assert _idx(old_side).match(long).name == old_side


def test_albums():
    assert album_for_folder("Photos from 2023") is None
    assert album_for_folder("Trash") is None
    assert album_for_folder("Beach Trip") == "Beach Trip"
