#!/usr/bin/env python3
"""
Real-time speech-to-text using IBM Granite Speech 4.1 2B.
Rich terminal UI with live RMS meter, status panel, and scrollable transcript.
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
import torchaudio
import pyaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

# ── Constants ──────────────────────────────────────────────────────────────────
import os
# Get the absolute path to the project root (3 levels up from src/voiceassistant/main.py)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "models", "granite-2b")

MODEL_ID = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "ibm-granite/granite-speech-4.1-2b"
SAMPLE_RATE   = 16_000
CHUNK_FRAMES  = 1_600           # 100 ms per read
SILENCE_RMS   = 50              # skip inference below this RMS
MAX_LINES     = 20              # transcript history shown on screen

console = Console()


# ── Args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="[bold cyan]Realtime STT[/] — IBM Granite Speech 4.1 2B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", default="transcribe-punct",
                   choices=["transcribe", "transcribe-punct", "translate"],
                   help="ASR task")
    p.add_argument("--lang",     default="English",  help="Translation target language")
    p.add_argument("--keywords", default="",         help="Comma-separated keyword list")
    p.add_argument("--device",   default="auto",     choices=["auto", "cuda", "cpu"])
    p.add_argument("--window",   type=float, default=5.0,  help="Audio window (seconds)")
    p.add_argument("--step",     type=float, default=2.0,  help="Slide step (seconds)")
    p.add_argument("--list-devices", action="store_true",  help="List input devices and exit")
    p.add_argument("--input-device", type=int, default=None, help="PyAudio device index")
    return p.parse_args()


# ── Prompt ─────────────────────────────────────────────────────────────────────
def build_prompt(tokenizer, task, lang, keywords):
    kw = f" Keywords: {keywords}." if keywords else ""
    texts = {
        "transcribe":       f"can you transcribe the speech into a written format?{kw}",
        "transcribe-punct": f"transcribe the speech with proper punctuation and capitalization.{kw}",
        "translate":        f"translate the speech to {lang} with proper punctuation and capitalization.{kw}",
    }
    chat = [{"role": "user", "content": f"<|audio|>{texts[task]}"}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


# ── Model ──────────────────────────────────────────────────────────────────────
def load_model(device_arg):
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    if device == "cuda" and not torch.cuda.is_available():
        console.print("[bold red]ERROR:[/] CUDA requested but not available. "
                      "Check driver compatibility with your torch build.")
        sys.exit(1)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), console=console) as p:
        tid = p.add_task(f"Loading [bold]{MODEL_ID}[/bold] on [bold yellow]{device}[/bold yellow] …")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        model.eval()
        p.update(tid, description=f"[green]Model ready on {device} ({dtype})[/]")

    return model, processor, device


# ── Inference ───────────────────────────────────────────────────────────────────
@torch.inference_mode()
def transcribe_chunk(model, processor, prompt, audio_np, device):
    wav = torch.from_numpy(audio_np).unsqueeze(0).float()
    inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
    n = inputs["input_ids"].shape[-1]
    return processor.tokenizer.batch_decode(
        outputs[0, n:].unsqueeze(0),
        add_special_tokens=False, skip_special_tokens=True,
    )[0].strip()


# ── Audio thread ────────────────────────────────────────────────────────────────
def audio_thread(audio_queue, stop_event, device_index):
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
        input=True, input_device_index=device_index,
        frames_per_buffer=CHUNK_FRAMES,
    )
    try:
        while not stop_event.is_set():
            raw = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            audio_queue.put(pcm)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── RMS bar ─────────────────────────────────────────────────────────────────────
def rms_bar(rms: float, width: int = 30) -> Text:
    level   = min(int(rms / 300 * width), width)
    silence = rms < SILENCE_RMS
    color   = "bright_black" if silence else ("green" if level < width * 0.6 else "yellow" if level < width * 0.85 else "red")
    bar = Text("█" * level + "░" * (width - level), style=color)
    label = Text(f" {rms:5.0f} RMS", style="dim" if silence else "white")
    return bar + label


# ── UI layout ───────────────────────────────────────────────────────────────────
def make_layout(lines: list[tuple[str, str]], rms: float, status: str, args) -> Table:
    # Header info
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan")
    info.add_column(style="white")
    info.add_row("Model",   MODEL_ID)
    info.add_row("Task",    args.task)
    info.add_row("Device",  args.device)
    if args.keywords:
        info.add_row("Keywords", args.keywords)

    header = Panel(info, title="[bold cyan]Granite Speech 4.1 2B — Real-time STT[/]",
                   border_style="cyan", box=box.ROUNDED)

    # Mic meter
    meter = Panel(rms_bar(rms), title="[dim]Microphone[/]",
                  border_style="dim", box=box.SIMPLE)

    # Status
    status_panel = Panel(Text(status, style="bold yellow"),
                         border_style="dim", box=box.SIMPLE)

    # Transcript
    transcript = Text()
    for ts, text in lines[-MAX_LINES:]:
        transcript.append(f"[{ts}] ", style="dim cyan")
        transcript.append(text + "\n", style="bright_white")
    if not lines:
        transcript.append("(waiting for speech …)", style="dim")

    trans_panel = Panel(transcript, title="[bold]Transcript[/]",
                        border_style="green", box=box.ROUNDED, expand=True)

    # Compose
    root = Table.grid()
    root.add_row(header)
    root.add_row(Columns([meter, status_panel]))
    root.add_row(trans_panel)
    return root


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.list_devices:
        pa = pyaudio.PyAudio()
        t = Table("Index", "Name", title="Audio Input Devices", box=box.ROUNDED,
                  show_header=True, header_style="bold cyan")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                t.add_row(str(i), info["name"])
        pa.terminate()
        console.print(t)
        sys.exit(0)

    win_frames  = int(args.window * SAMPLE_RATE)
    step_frames = int(args.step   * SAMPLE_RATE)

    model, processor, device = load_model(args.device)
    args.device = device  # resolve "auto" for display
    prompt = build_prompt(processor.tokenizer, args.task, args.lang, args.keywords)

    audio_buf: deque = deque(maxlen=win_frames)
    audio_q: queue.Queue = queue.Queue()
    stop_evt = threading.Event()
    rms_state = [0.0]
    status_state = ["Listening …"]
    lines: list[tuple[str, str]] = []
    last_text = ""

    t = threading.Thread(
        target=audio_thread, args=(audio_q, stop_evt, args.input_device), daemon=True
    )
    t.start()

    with Live(make_layout(lines, 0.0, "Starting …", args),
              console=console, refresh_per_second=10) as live:
        frames_since_last = 0
        try:
            while True:
                while not audio_q.empty():
                    chunk = audio_q.get_nowait()
                    audio_buf.extend(chunk)
                    rms_state[0] = float(np.sqrt(np.mean(chunk ** 2)) * 32768)

                frames_since_last += CHUNK_FRAMES

                if frames_since_last >= step_frames and len(audio_buf) >= win_frames:
                    frames_since_last = 0
                    audio_np = np.array(audio_buf, dtype=np.float32)
                    rms = float(np.sqrt(np.mean(audio_np ** 2)) * 32768)

                    if rms < SILENCE_RMS:
                        status_state[0] = "[dim]Silence — skipping[/]"
                    else:
                        status_state[0] = "[yellow]Transcribing …[/]"
                        live.update(make_layout(lines, rms_state[0], status_state[0], args))

                        text = transcribe_chunk(model, processor, prompt, audio_np, device)
                        if text and text != last_text:
                            ts = time.strftime("%H:%M:%S")
                            lines.append((ts, text))
                            last_text = text

                        status_state[0] = "Listening …"

                live.update(make_layout(lines, rms_state[0], status_state[0], args))
                time.sleep(0.05)

        except KeyboardInterrupt:
            status_state[0] = "[red]Stopped.[/]"
            live.update(make_layout(lines, 0.0, status_state[0], args))

    stop_evt.set()
    t.join(timeout=2)
    console.print("\n[bold green]Session ended.[/] Goodbye!")


if __name__ == "__main__":
    main()
