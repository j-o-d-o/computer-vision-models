from models.semseg import create_model
from models.semseg import Params

class TestModel:
    def test_create_model(self):
        model = create_model(Params.INPUT_HEIGHT, Params.INPUT_WIDTH)
        model.summary()
