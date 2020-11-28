from models.centernet_2d import create_model


class TestModel:
    def test_create_model(self):
        model = create_model(185, 612, int(185 // 2), int(612 // 2), 5)
