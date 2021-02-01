from models.centertracker.model import create_model
from models.centertracker.params import CentertrackerParams
from data.od_spec import OD_CLASS_MAPPING


class TestModel:
    def test_create_model(self):
        params = CentertrackerParams(len(OD_CLASS_MAPPING))
        model = create_model(params)
        model.summary()
