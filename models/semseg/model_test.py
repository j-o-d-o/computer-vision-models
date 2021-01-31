from models.semseg import create_model
from models.semseg import SemsegParams

class TestModel:
    def test_create_model(self):
        params = SemsegParams()
        model = create_model(params.INPUT_HEIGHT, params.INPUT_WIDTH)
        model.summary()
