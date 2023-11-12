import cv2
import numpy as np


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


def ajusta_saturacao(imagem):
    # Ajuste de Saturação
    sat_adj = 3
    val_adj = 1
    img_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(img_hsv)
    s = s * sat_adj
    v = v * val_adj
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    img_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(img_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)


# Setando Variaveis globais
lower = np.array([0, 255, 0])
upper = np.array([178, 255, 255])

vermelho = (0, 0, 255)
verde = (0, 255, 0)
azul = (255, 0, 0)
font = cv2.FONT_HERSHEY_DUPLEX

# Image to detect shapes on below
PATH = r"E:\Faculdade\5 Semestre\Processamento de Imagens\Trabalho Final\Identificar Salgadinhos\IdentificarSalgadinhos-VC\Imagens\shape_5.jpg"
img = cv2.imread(PATH)

img = preserva_aspect_ratio_resize(img, altura=800)

img_rect = img.copy()

img_sat = ajusta_saturacao(img)

hsv = cv2.cvtColor(img_sat, cv2.COLOR_BGR2HSV)

mascara = cv2.inRange(hsv, lower, upper)

# Retrieving outer-edge coordinates in the new threshold image
contours, hierarchy = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterating through each contour to retrieve coordinates of each shape
for i, contour in enumerate(contours):
    if i == 0:
        continue

    # The 2 lines below this comment will approximate the shape we want. The reason being that in certain cases the
    # shape we want might have flaws or might be imperfect, and so, for example, if we have a rectangle with a
    # small piece missing, the program will still count it as a rectangle. The epsilon value will specify the
    # precision in which we approximate our shape.
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Retrieving coordinates of the contour so that we can put text over the shape.
    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(
        x + (w / 3)
    )  # This is an estimation of where the middle of the shape is in terms of the x-axis.
    y_mid = int(
        y + (h / 1.5)
    )  # This is an estimation of where the middle of the shape is in terms of the y-axis.

    # Setting some variables which will be used to display text on the final image
    coords = (x_mid, y_mid)

    # This is the part where we actually guess which shape we have detected. The program will look at the amount of edges
    # the contour/shape has, and then based on that result the program will guess the shape (for example, if it has 3 edges
    # then the chances that the shape is a triangle are very good.)
    #
    # You can add more shapes if you want by checking more lenghts, but for the simplicity of this tutorial program I
    # have decided to only detect 5 shapes.
    area = cv2.contourArea(contour)

    if area > 15000:
        cv2.putText(
            img_rect, "Doritos", coords, font, 1, vermelho, 2
        )  # Text on the image
    elif area > 10000:
        cv2.putText(img_rect, "Cebolitos", coords, font, 1, verde, 2)
    elif area > 5000:
        # If the length is not any of the above, we will guess the shape/contour to be a circle.
        cv2.putText(img_rect, "Cheetos", coords, font, 1, azul, 2)


# Displaying the image with the detected shapes onto the screen
cv2.imshow("shapes_detected", img_rect)
cv2.waitKey(0)
