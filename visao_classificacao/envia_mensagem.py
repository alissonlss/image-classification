# Importando as bibliotecas
import pywhatkit as kit
import time

def enviar_mensagem():
    # Número de telefone (com código do país) e mensagem
    numero_telefone = "+55DD9XXXXXXXX"  # Substitua pelo número do destinatário
    mensagem = "Seu pet está fugindo!"
    hora_atual = time.localtime()
    hora = hora_atual.tm_hour
    minuto = hora_atual.tm_min + 1

    if (minuto >= 60):

        minuto = 0

        if (hora == 23):
            hora = 0
        else:
            hora += 1
        
    # Envie a mensagem com a imagem
    kit.sendwhatmsg(numero_telefone, mensagem, hora, minuto)

# Testando
enviar_mensagem()