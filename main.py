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


def id_imagens(id_img):
    # Pathing para as imagens a serem processadas
    PATH = r"E:\Faculdade\5 Semestre\Processamento de Imagens\Trabalho Final\Identificar Salgadinhos\IdentificarSalgadinhos-VC\Imagens\shape_{}.jpg".format(
        id_img
    )
    img = cv2.imread(PATH)

    # Ajusta a altura para que a imagem apareca por inteiro na tela
    # Valor padrao de 800px, caso este valor seja alterado, deve-se
    # analizar novamente a area minima dos salgadinhos na deteccao
    img = preserva_aspect_ratio_resize(img, altura=800)
    img_rect = img.copy()

    # Processa a imagem para criar uma mascara contendo apenas os salgadinhos
    img_sat = ajusta_saturacao(img)
    hsv = cv2.cvtColor(img_sat, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, lower, upper)

    # Retira os contornos da imagem
    contornos = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Passa de contorno em contorno retirando suas informacoes e coordenadas
    for i, contour in enumerate(contornos):
        # O primeiro contorno eh a propria imagem, sendo assim podemos ignoralo
        if i == 0:
            continue

        # Calcula a area dos contornos para definir o tipo do salgadinho
        area = cv2.contourArea(contour)

        # Retrieving coordinates of the contour so that we can put text over the shape.
        x, y, w, h = cv2.boundingRect(contour)

        # Estima o meio do contono no eixo X
        x_mid = int(x + (w / 3))
        # Estima o meio do contono no eixo Y
        y_mid = int(y + (h / 1.5))
        # Pega as coordenadas e salva para escrever na tela
        coords = (x_mid, y_mid)

        if area > 15000:
            cv2.putText(img_rect, "Doritos", coords, font, 1, vermelho, 2)
        elif area > 10000:
            cv2.putText(img_rect, "Cebolitos", coords, font, 1, verde, 2)
        elif area > 5000:
            cv2.putText(img_rect, "Cheetos", coords, font, 1, azul, 2)

    # Mostra na tela as deteccoes
    cv2.imshow("Salgadinhos Detectados", img_rect)
    cv2.waitKey(0)


def id_tempo_real():
    camera = cv2.VideoCapture(0)

    while 1:
        frame = camera.read()[1]
        frame = preserva_aspect_ratio_resize(frame, altura=800)

        frame_copy = frame.copy()

        # Processa a imagem para criar uma mascara contendo apenas os salgadinhos
        img_sat = ajusta_saturacao(frame)
        hsv = cv2.cvtColor(img_sat, cv2.COLOR_BGR2HSV)
        mascara = cv2.inRange(hsv, lower, upper)

        # Retira os contornos da imagem
        contornos = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        # Passa de contorno em contorno retirando suas informacoes e coordenadas
        for i, contour in enumerate(contornos):
            # O primeiro contorno eh a propria imagem, sendo assim podemos ignoralo
            if i == 0:
                continue

            # Calcula a area dos contornos para definir o tipo do salgadinho
            area = cv2.contourArea(contour)

            # Retrieving coordinates of the contour so that we can put text over the shape.
            x, y, w, h = cv2.boundingRect(contour)

            # Estima o meio do contono no eixo X
            x_mid = int(x + (w / 3))
            # Estima o meio do contono no eixo Y
            y_mid = int(y + (h / 1.5))
            # Pega as coordenadas e salva para escrever na tela
            coords = (x_mid, y_mid)

            if area > 15000:
                cv2.putText(frame_copy, "Doritos", coords, font, 1, vermelho, 2)
            elif area > 10000:
                cv2.putText(frame_copy, "Cebolitos", coords, font, 1, verde, 2)
            elif area > 5000:
                cv2.putText(frame_copy, "Cheetos", coords, font, 1, azul, 2)

        # Mostra a imagem
        cv2.imshow("Identificador de Salgadinhos", frame_copy)

        # Espera mais tempo para evitar que a taxa de atualização seja maior que o framerate de videos
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


# Setando Variaveis globais
lower = np.array([0, 255, 0])
upper = np.array([178, 255, 255])

vermelho = (0, 0, 255)
verde = (0, 255, 0)
azul = (255, 0, 0)
font = cv2.FONT_HERSHEY_DUPLEX

entrada = input("Camera ou Imagem (C/I):").strip().lower()

if entrada == "c":
    id_tempo_real()
elif entrada == "i":
    print("Escolha a imagem:")
    print("1: Somente Cebolitos")
    print("2: Somente Cheetos")
    print("3: Somente Doritos")
    print("4: Mix 1")
    print("5: Mix 2")
    entrada = input("Digite o valor desejado:")
    entrada = int(entrada)

    if entrada <= 5:
        id_imagens(entrada)
    else:
        print("Entrada Invalida!")
