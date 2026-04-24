from Explore import angle_to

def test_angle_to():
    assert angle_to(0, 0) == 0
    assert angle_to(90, 0) == 90
    assert angle_to(180, 0) == 180
    assert angle_to(270, 0) == -90
    assert angle_to(0, 90) == -90
    assert angle_to(90, 90) == 0
    assert angle_to(180, 90) == 90
    assert angle_to(270, 90) == 180
    assert angle_to(0, 180) == 180
    assert angle_to(90, 180) == -90
    assert angle_to(180, 180) == 0
    assert angle_to(270, 180) == 90
    assert angle_to(0, 270) == 90
    assert angle_to(90, 270) == 180
    assert angle_to(180, 270) == -90
    assert angle_to(270, 270) == 0

