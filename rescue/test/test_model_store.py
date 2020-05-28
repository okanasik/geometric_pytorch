from rescue.models.test_model import TestModel
from rescue.models.model import Model


if __name__ == "__main__":
    model_file_name = "test_model.pth"
    model = TestModel(128, 128)
    model.mysave(model_file_name)

    model2 = Model.load(model_file_name)
    print(model2.name)
    print(model2.num_features)