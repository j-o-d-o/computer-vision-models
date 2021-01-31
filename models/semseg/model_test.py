from models.semseg import create_model
from models.semseg import SemsegParams

class TestModel:
    def test_create_model(self):
        model = create_model(SemsegParams.INPUT_HEIGHT, SemsegParams.INPUT_WIDTH)
        model.summary()
