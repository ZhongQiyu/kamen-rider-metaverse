# main.py

class ModelA:
    def __init__(self):
        print("ModelA initialized")

    def predict(self, input_data):
        return "ModelA prediction for {}".format(input_data)

class ModelB:
    def __init__(self):
        print("ModelB initialized")

    def predict(self, input_data):
        return "ModelB prediction for {}".format(input_data)

if __name__ == "__main__":
    a = ModelA()
    print(a.predict("test data A"))

    b = ModelB()
    print(b.predict("test data B"))
