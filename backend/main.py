import os
import glob
import base64
import shutil
import tempfile
import subprocess

import requests
from groq import Groq
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
    HTTPException,
    Query,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Configuração (tudo via variáveis de ambiente — nada de chave no código)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")

# Limite do free tier do Groq é 25 MB por arquivo. Usamos 24 MB de margem
# de segurança para decidir quando precisamos fatiar.
LIMITE_BYTES = 24 * 1024 * 1024

# Duração de cada pedaço quando o arquivo é grande (5 min em segundos).
# A 16 kHz mono, 5 min de WAV dá ~9,6 MB — bem abaixo do limite de 25 MB
# do Groq e leve na RAM (importa em containers pequenos, ex.: 256 MB).
DURACAO_CHUNK_S = 5 * 60

MODELO = "whisper-large-v3"  # mesma qualidade do seu faster-whisper large-v3

if not GROQ_API_KEY:
    print("⚠️  GROQ_API_KEY não configurada. Defina a variável de ambiente antes de transcrever.")

groq = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Transcritor IA (Groq)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _campo(segmento, chave):
    """Lê um campo do segmento, funcione ele como dict ou como objeto."""
    if isinstance(segmento, dict):
        return segmento.get(chave)
    return getattr(segmento, chave, None)


def transcrever_groq(caminho_audio, offset_segundos=0.0):
    """Manda um arquivo de áudio para o Groq e devolve os segmentos.

    offset_segundos é somado aos tempos para que, ao fatiar, os timestamps
    continuem corretos em relação ao áudio original inteiro.
    """
    with open(caminho_audio, "rb") as fh:
        resposta = groq.audio.transcriptions.create(
            # Passa o handle aberto (não fh.read()) pra não segurar o arquivo
            # inteiro na RAM — o cliente transmite em streaming.
            file=(os.path.basename(caminho_audio), fh),
            model=MODELO,
            language="pt",
            response_format="verbose_json",  # traz segmentos com start/end/text
        )

    segmentos = []
    for s in (resposta.segments or []):
        segmentos.append(
            {
                "start": float(_campo(s, "start") or 0.0) + offset_segundos,
                "end": float(_campo(s, "end") or 0.0) + offset_segundos,
                "text": (_campo(s, "text") or "").strip(),
            }
        )
    return segmentos


def fatiar_audio(caminho):
    """Quebra um áudio grande em pedaços de DURACAO_CHUNK_S.

    Usa o ffmpeg em streaming (segment muxer): ele lê o arquivo aos poucos,
    converte para 16 kHz mono — formato que o Groq usa internamente, sem
    perda de qualidade — e grava cada pedaço como WAV no disco, SEM carregar
    o áudio inteiro na memória (era isso que estourava a RAM antes).

    Retorna uma lista de (caminho_do_pedaco, offset_em_segundos).
    """
    dir_saida = tempfile.mkdtemp(prefix="chunks_")
    padrao = os.path.join(dir_saida, "chunk_%05d.wav")

    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y",
            "-i", caminho,
            "-ar", "16000", "-ac", "1",
            "-f", "segment",
            "-segment_time", str(DURACAO_CHUNK_S),
            "-reset_timestamps", "1",
            padrao,
        ],
        check=True,
        capture_output=True,
    )

    pedacos = []
    for i, caminho_pedaco in enumerate(sorted(glob.glob(os.path.join(dir_saida, "chunk_*.wav")))):
        pedacos.append((caminho_pedaco, float(i * DURACAO_CHUNK_S)))
    return pedacos


def enviar_email_resend(email_destino, nome_original, segmentos):
    """Envia o relatório de transcrição por e-mail via API REST do Resend."""
    if not RESEND_API_KEY:
        print("RESEND_API_KEY não configurada — pulando envio de e-mail.")
        return

    try:
        texto = "RELATÓRIO DE TRANSCRIÇÃO IA\n" + "=" * 40 + "\n\n"
        for s in segmentos:
            texto += f"[{s['start']:.2f}s] {s['text']}\n"

        conteudo_base64 = base64.b64encode(texto.encode("utf-8")).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "from": "Transcrição IA <onboarding@resend.dev>",
            "to": [email_destino],
            "subject": f"Transcrição Concluída: {nome_original}",
            "html": (
                f"<h3>Olá!</h3>"
                f"<p>A transcrição do arquivo <b>{nome_original}</b> foi concluída.</p>"
                f"<p>O relatório segue em anexo.</p>"
            ),
            "attachments": [
                {"filename": "transcricao.txt", "content": conteudo_base64}
            ],
        }

        print(f"Enviando e-mail para {email_destino}...")
        resposta = requests.post(
            "https://api.resend.com/emails", headers=headers, json=payload, timeout=30
        )
        if resposta.status_code in (200, 201):
            print("✅ E-mail enviado com sucesso.")
        else:
            print(f"❌ Falha no Resend (status {resposta.status_code}): {resposta.text}")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Erro ao enviar e-mail: {e}")


def enviar_email_falha(email_destino, nome_original, motivo):
    """Avisa o usuário quando a transcrição falhou (ex.: em modo background)."""
    if not RESEND_API_KEY:
        return
    try:
        headers = {
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "from": "Transcrição IA <onboarding@resend.dev>",
            "to": [email_destino],
            "subject": f"Falha na transcrição: {nome_original}",
            "html": (
                f"<h3>Ops!</h3>"
                f"<p>Não consegui transcrever o arquivo <b>{nome_original}</b>.</p>"
                f"<p>Motivo técnico: {motivo}</p>"
                f"<p>Tente novamente; se o arquivo for muito grande, divida-o.</p>"
            ),
        }
        requests.post(
            "https://api.resend.com/emails", headers=headers, json=payload, timeout=30
        )
    except Exception as e:  # noqa: BLE001
        print(f"❌ Erro ao enviar e-mail de falha: {e}")


def processar(caminho, nome, email):
    """Trabalho pesado (síncrono). Roda em threadpool ou em background.

    Fatia o arquivo se for grande, transcreve cada parte no Groq e,
    opcionalmente, envia o resultado por e-mail.
    """
    try:
        segmentos = []
        if os.path.getsize(caminho) <= LIMITE_BYTES:
            segmentos = transcrever_groq(caminho)
        else:
            print(f"Arquivo grande detectado — fatiando {nome}...")
            pedacos = fatiar_audio(caminho)
            dir_pedacos = os.path.dirname(pedacos[0][0]) if pedacos else None
            try:
                for caminho_pedaco, offset in pedacos:
                    segmentos.extend(transcrever_groq(caminho_pedaco, offset))
            finally:
                if dir_pedacos and os.path.isdir(dir_pedacos):
                    shutil.rmtree(dir_pedacos, ignore_errors=True)

        texto = " ".join(s["text"] for s in segmentos).strip()

        if email:
            enviar_email_resend(email, nome, segmentos)

        return segmentos, texto
    except Exception as e:  # noqa: BLE001
        # Em modo background a exceção morreria silenciosa — avisa por e-mail.
        print(f"❌ Erro ao processar {nome}: {e}")
        if email:
            enviar_email_falha(email, nome, str(e))
        raise
    finally:
        if os.path.exists(caminho):
            os.remove(caminho)
            print(f"🧹 Limpeza: {nome} removido do disco.")


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "modelo": MODELO}


@app.post("/transcrever")
async def handle_transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    email: str = Form(None),
    run_async: bool = Query(False),
):
    print(f"📥 Recebido: {file.filename} | async={run_async} | email={'sim' if email else 'não'}")

    _, ext = os.path.splitext(file.filename or "")
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".webm")
    shutil.copyfileobj(file.file, temp)
    temp.close()

    if run_async:
        if not email:
            os.remove(temp.name)
            raise HTTPException(
                status_code=400,
                detail="Modo assíncrono precisa de um e-mail para notificar.",
            )
        background_tasks.add_task(processar, temp.name, file.filename, email)
        return {
            "status": "async_started",
            "message": "Transcrição despachada. Você receberá o resultado por e-mail.",
        }

    try:
        segmentos, texto = await run_in_threadpool(
            processar, temp.name, file.filename, email
        )
        return {"status": "completed", "segmentos": segmentos, "texto": texto}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


# Execução local: `python main.py` (no Northflank o Dockerfile cuida disso).
if __name__ == "__main__":
    import uvicorn

    porta = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=porta)
