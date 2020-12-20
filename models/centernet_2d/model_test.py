from models.centernet_2d import create_model


class TestModel:
    def test_create_model(self):
        model = create_model(92, 308, int(92 // 2), int(308 // 2), 5)
        model.summary()
