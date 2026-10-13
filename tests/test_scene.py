from faultline.scene import Obj, parse_scene, spatial_relation

from fixtures import obj, street_scene


def box(x, y, w, h):
    return Obj("o", "x", None, x, y, w, h, 0.0)


def test_left_right():
    a, b = box(0, 100, 100, 100), box(300, 120, 100, 100)
    assert spatial_relation(a, b, 1000, 1000, 0.02, 0.3) == ("to the left of", 0.2)
    assert spatial_relation(b, a, 1000, 1000, 0.02, 0.3)[0] == "to the right of"


def test_above_below():
    a, b = box(100, 0, 100, 100), box(120, 400, 100, 100)
    assert spatial_relation(a, b, 1000, 1000, 0.02, 0.3)[0] == "above"
    assert spatial_relation(b, a, 1000, 1000, 0.02, 0.3)[0] == "below"


def test_diagonal_and_touching_are_rejected():
    a = box(0, 0, 100, 100)
    assert spatial_relation(a, box(400, 400, 100, 100), 1000, 1000, 0.02, 0.3) is None
    assert spatial_relation(a, box(110, 0, 100, 100), 1000, 1000, 0.02, 0.3) is None  # gap 0.01
    assert spatial_relation(a, box(50, 50, 100, 100), 1000, 1000, 0.02, 0.3) is None  # overlap


def test_parse_normalises_names_and_colors():
    g = {"width": 100, "height": 100, "objects": {
        "1": obj("TV", 0, 0, 10, 10, ["Grey", "shiny"]),
        "2": obj("sofa", 20, 0, 10, 10, ["red", "blue"]),
        "3": obj("ghost", 0, 0, 0, 10),
    }}
    s = parse_scene("9", g)
    assert [o.canon for o in s.objects] == ["television", "couch"]
    assert s.objects[0].colors == ["gray"]
    assert s.objects[1].colors == ["red", "blue"]


def test_absence_is_alias_aware():
    s = parse_scene("1", {"width": 100, "height": 100,
                          "objects": {"1": obj("bike", 0, 0, 10, 10)}})
    assert not s.is_absent("bicycle")
    assert s.is_absent("car")


def test_people_are_not_vocabulary():
    s = parse_scene("1", street_scene())
    men = [o for o in s.objects if o.name == "man"]
    assert men and all(o.canon is None for o in men)
