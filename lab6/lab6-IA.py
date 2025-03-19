import cv2
import numpy as np

def detectar_formas(img):
    # Convertendo a imagem para escala de cinza
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicando a detecção de bordas com Canny
    edges = cv2.Canny(img_gray, 50, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def calcular_area(contorno):
    # Calculando a área de um contorno
    return cv2.contourArea(contorno)


def detectar_colisao(contornos, barra_x, barra_y, barra_altura):
    # Detectando colisão entre o quadrado (forma) e a barra
    for contour in contornos:
        x, y, w, h = cv2.boundingRect(contour)
        # Verificando se o quadrado colidiu com a barra (considerando a barra como uma linha vertical)
        if (x + w > barra_x and x < barra_x + 10) and (y + h > barra_y and y < barra_y + barra_altura):
            return True, (x, y, w, h)  # Retorna as coordenadas do contorno que colidiu
    return False, None


def detectar_ultrapassagem(contorno, linha_referencia_x):
    # Função para verificar se a forma ultrapassou a "barreira" (linha de referência) após colisão com a barra
    x, y, w, h = cv2.boundingRect(contorno)

    # Verifica se a forma ultrapassou a linha de referência ao longo do eixo X
    if x + w > linha_referencia_x:
        return True
    return False


def processar_video(video_path):
    cap = cv2.VideoCapture(video_path)

    colisao_detectada = False  # Flag para controlar quando a colisão foi detectada
    ultrapassou_barreira = False  # Flag para controlar se o quadrado ultrapassou a barreira

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Processamento do quadro
        output = frame.copy()

        # Detecção de formas
        contours = detectar_formas(frame)

        formas = []
        for contour in contours:
            area = calcular_area(contour)
            formas.append((contour, area))

        # R2 - Identificar a forma de maior massa (área)
        if formas:
            maior_area_contorno = max(formas, key=lambda x: x[1])
            maior_area_contorno = maior_area_contorno[0]
            # Desenhando um retângulo verde em torno da maior forma
            x, y, w, h = cv2.boundingRect(maior_area_contorno)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Atualizando a posição da barra com base no contorno da forma (em azul)
            barra_x = x  # Posição x da barra é a mesma do contorno
            barra_y = y + h  # Posição y da barra será logo abaixo do contorno (para não sobrepor)

            # A altura da barra será igual à altura do contorno
            barra_altura = h

        # R3 - Detecção de colisão com a barra
        colisao_detectada = False  # Resetando a flag de colisão para cada quadro
        for contour in formas:
            x, y, w, h = cv2.boundingRect(contour[0])
            if (x + w > barra_x and x < barra_x + 10) and (y + h > barra_y and y < barra_y + barra_altura):
                colisao_detectada = True
                cv2.putText(output, "COLISÃO COM A BARRA DETECTADA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                break

        # R4 - Detecção de ultrapassagem
        if colisao_detectada and not ultrapassou_barreira:  # Verifica se já houve colisão e ainda não ultrapassou a barreira
            linha_referencia_x = output.shape[1] - 100  # Definindo a posição da "barreira" (linha de referência)
            for i in range(len(formas)):
                if detectar_ultrapassagem(formas[i][0], linha_referencia_x):
                    ultrapassou_barreira = True
                    cv2.putText(output, "ULTRAPASSOU A BARREIRA", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Exibindo o quadro com as deteções
        cv2.imshow('Resultado', output)

        # Aguardar tecla 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Caminho para o vídeo
video_path = 'q1B.mp4'

# Processar o vídeo
processar_video(video_path)
