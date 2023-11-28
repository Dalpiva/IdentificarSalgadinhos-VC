from ultralytics import YOLO
import cv2


def preserva_aspect_ratio_resize(
    imagem, largura=None, altura=None, inter=cv2.INTER_AREA
):
    # Recupera o tamanho da imagem e inicializa as dimensões
    dim = None
    (h, w) = imagem.shape[:2]

    # Retorna a imagem original se não houver necessidade de redimensionar
    if largura is None and altura is None:
        return imagem

    # Redimensiona algura se largura for none
    if largura is None:
        # Calcula o aspecto da altura e constrói as dimensões
        r = altura / float(h)
        dim = (int(w * r), altura)
    else:  # Redimensiona a largura se a altura for none
        # Calcula o aspecto da largura e constrói as dimensões
        r = largura / float(w)
        dim = (largura, int(h * r))

    return cv2.resize(imagem, dim, interpolation=inter)


id_img = 29
PATH = r"E:\Faculdade\5 Semestre\Processamento de Imagens\Trabalho Final\Identificar Salgadinhos\IdentificarSalgadinhos-VC\Imagens\img_{}.jpg".format(
    id_img
)
img = cv2.imread(PATH)

img = cv2.imread(PATH)
img = preserva_aspect_ratio_resize(img, altura=800)

altura, largura = img.shape[:2]

path_treino = r"E:\TRABALHO-VC\runs\detect\train2\weights\best.pt"

modelo_yolo = YOLO(path_treino)

resultado = modelo_yolo(img)[0]

limear = 0.5

for result in resultado.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > limear:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(
            img,
            resultado.names[int(class_id)].upper(),
            (int(x1), int(y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )


cv2.imshow("Salgadinhos Detectados", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
