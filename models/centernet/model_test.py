from models.centernet import create_model
from models.centernet.params import Params
from data.od_spec import OD_CLASS_MAPPING


class TestModel:
    def test_create_model(self):
        params = Params(len(OD_CLASS_MAPPING))
        model = create_model(params)
        model.summary()
