# Importando as bibliotecas
from PIL import Image

def apresenta_classifica_precissao(probs):

    tags = ["Pet", "People", "Pet and People", "Pet Running Away"]
    probs_simple = [probs[0][0].item(), probs[0][1].item(), probs[0][2].item(), probs[0][3].item()]

    tag = tags[probs_simple.index(max(probs_simple))]

    return tag

def classifica_imagem(url, model, processor):
    image = Image.open(url)

    # classes para classificacao
    classes = ["a photo of a pet", "a photo of a people", "a photo of a pet and people", "a photo of a pet running away"]

    inputs = processor(text=classes, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # score entre a imagem e as tags
    probs = logits_per_image.softmax(dim=1) # obtendo as probabilidades das tags

    result = apresenta_classifica_precissao(probs)

    return result