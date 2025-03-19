import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def detectar_formas(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def calcular_area(contorno):
    return cv2.contourArea(contorno)


# Função para adicionar texto com Pillow
def putTextPIL(img, text, position, font_size=30, color=(0, 255, 255)):
    pil_img = Image.fromarray(img)  # Converte a imagem do OpenCV para o formato Pillow
    draw = ImageDraw.Draw(pil_img)

    # Usando uma fonte TrueType
    font = ImageFont.truetype("arial.ttf", font_size)  # Caminho para a fonte TTF

    draw.text(position, text, font=font, fill=color)
    return np.array(pil_img)  # Converte de volta para o formato OpenCV


def processar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ultrapassagem_confirmada = False
    colisao_ocorreu = False
    ultimo_contato = False  # Variável para controlar o último contato entre as formas

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output = frame.copy()
        contours = detectar_formas(frame)

        formas = [(contour, calcular_area(contour)) for contour in contours]

        if len(formas) >= 2:
            maior_contorno = max(formas, key=lambda x: x[1])[0]
            menor_contorno = min(formas, key=lambda x: x[1])[0]

            # Desenha o contorno em azul
            cv2.drawContours(output, [maior_contorno], -1, (0, 255, 0), 2)  # Azul (BGR: 255, 0, 0)

            colisao = (cv2.boundingRect(menor_contorno)[0] + cv2.boundingRect(menor_contorno)[2] >
                       cv2.boundingRect(maior_contorno)[0] and
                       cv2.boundingRect(menor_contorno)[0] < cv2.boundingRect(maior_contorno)[0] +
                       cv2.boundingRect(maior_contorno)[2] and
                       cv2.boundingRect(menor_contorno)[1] + cv2.boundingRect(menor_contorno)[3] >
                       cv2.boundingRect(maior_contorno)[1] and
                       cv2.boundingRect(menor_contorno)[1] < cv2.boundingRect(maior_contorno)[1] +
                       cv2.boundingRect(maior_contorno)[3])

            if colisao:
                # Marca que a colisão ocorreu
                colisao_ocorreu = True
                ultimo_contato = True  # Marca que há contato entre as formas
                # Exibe a mensagem de colisão
                output = putTextPIL(output, "COLISÃO DETECTADA", (1000, 100), font_size=40, color=(0, 255, 255))

            elif colisao_ocorreu and not colisao:
                # A colisão foi interrompida, vamos verificar se a ultrapassagem ocorreu
                maior_contorno_x = cv2.boundingRect(maior_contorno)[0]
                menor_contorno_x = cv2.boundingRect(menor_contorno)[0]

                # A ultrapassagem só é confirmada se o quadrado passou a barra
                if maior_contorno_x < menor_contorno_x:
                    ultrapassagem_confirmada = True
                    colisao_ocorreu = False  # Resetando a flag de colisão
                    ultimo_contato = False  # Resetando o estado de contato

            # Exibe a mensagem de "PASSOU A BARREIRA" somente se não houver mais colisão e o contato foi finalizado
            if ultrapassagem_confirmada and not ultimo_contato:
                output = putTextPIL(output, "PASSOU A BARREIRA", (1000, 100), font_size=40, color=(0, 0, 255))

        cv2.imshow('Resultado', output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Caminho do vídeo
video_path = 'q1B.mp4'
processar_video(video_path)
