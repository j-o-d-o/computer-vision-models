from models.semseg import create_model


class TestModel:
    def test_create_model(self):
        model = create_model()
        model.summary()
