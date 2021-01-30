from models.centernet import create_model
from models.centernet.params import Params
from data.od_spec import OD_CLASS_MAPPING


class TestModel:
    def test_create_model(self):
        model = create_model(Params.INPUT_HEIGHT, Params.INPUT_WIDTH, None, None, len(OD_CLASS_MAPPING))
        model.summary()
