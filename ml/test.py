from tensorflow.keras.models import load_model

model = load_model("cancer_model.h5")

print("\n===== MODELE =====")
for i, layer in enumerate(model.layers):
    print(i, layer.name, type(layer).__name__)

print("\n===== EFFICIENTNET =====")
base = model.get_layer("efficientnetb0")

for i, layer in enumerate(base.layers[-20:]):
    print(i, layer.name, type(layer).__name__)


#from tensorflow.keras.models import load_model

model = load_model("cancer_model.h5", compile=False)

print(model.summary())

print("\n-----------")

for layer in model.layers:
    print(layer.name, type(layer))

efficientnet = model.get_layer("efficientnetb0")

print("\nTOP =", efficientnet.get_layer("top_conv"))