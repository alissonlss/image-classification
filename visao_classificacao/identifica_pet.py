# Carregar as dependencias
import cv2
import time
import classifica
import numpy as np

# Importando o modelo e manipulação de imagem 
from transformers import CLIPProcessor, CLIPModel

# Instanciando o modelo de IA para classificacao de imagem
model_ia = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Cores das classes
# colors = [(0, 255, 255), (255, 255, 0), (0, 255,0), (255, 0, 0)]
colors = np.random.uniform(0, 255, size=(80, 3))

# Carregar nomes de classes
class_names = []
with open("yolo/coco.names", "r") as f:
    class_names = f.read().strip().split("\n")

# Carregar o modelo YOLO e configuração
net = cv2.dnn.readNet("yolo/yolov4-tiny.weights", "yolo/yolov4-tiny.cfg")

# Iniciar a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Setando os parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# Lendo os frames do video
while True:

    # Captura do frame
    _, frame  = cap.read()

    # Comeco da contagem dos ms
    start = time.time()

    # Deteccao
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # Fim da contagem dos ms
    end = time.time()

    # Percorrer todas as deteccoes
    for (classid, score, box) in zip(classes, scores, boxes):

        # Gerando uma cor para a classe
        color = colors[int(classid) % len(colors)]

        if (class_names[classid] == "dog"):
            # Salva a imagem
            cv2.imwrite("imagem.jpg", frame)

            result = classifica.classifica_imagem("imagem.jpg", model_ia, processor)
            print(result)

        # Pegando o nome da classe pelo id e o seu score acuracia
        label = f"{class_names[classid]} : {score: .2f}"

        # Desenhando a box da deteccao
        cv2.rectangle(frame, box, color, 2)

        # Escrevendo o nome da class em cima do box do objeto
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculando o tempo que levou para fazer a dateccao
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"

    # Escrevendo o fps na imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Mostrando a imagem
    cv2.imshow("Detection", frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()