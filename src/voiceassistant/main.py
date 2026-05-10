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

import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "models", "granite-2b")

MODEL_ID      = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "ibm-granite/granite-speech-4.1-2b"
SAMPLE_RATE   = 16_000
CHUNK_FRAMES  = 1_600       # 100 ms per read
MAX_LINES     = 20
MAX_UTT_S     = 10.0        # force-flush utterance after this many seconds of speech
SILENCE_PAD_S = 0.3         # keep this many seconds of silence at end of utterance

console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Realtime STT — IBM Granite Speech 4.1 2B")
    p.add_argument("--task",         default="transcribe-punct",
                   choices=["transcribe", "transcribe-punct", "translate"])
    p.add_argument("--lang",         default="English")
    p.add_argument("--keywords",     default="")
    p.add_argument("--device",       default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--silence-rms",  type=float, default=300.0,
                   help="RMS threshold below which audio is considered silence (default 300). "
                        "Run with --calibrate to find a good value for your mic.")
    p.add_argument("--silence-hold", type=float, default=0.6,
                   help="Seconds of silence after speech before the utterance is flushed (default 0.6)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--input-device", type=int, default=None)
    p.add_argument("--calibrate",    action="store_true",
                   help="Run 3-second noise floor calibration and print recommended --silence-rms")
    return p.parse_args()


def build_prompt(tokenizer, task, lang, keywords):
    kw = f" Keywords: {keywords}." if keywords else ""
    texts = {
        "transcribe":       f"can you transcribe the speech into a written format?{kw}",
        "transcribe-punct": f"transcribe the speech with proper punctuation and capitalization.{kw}",
        "translate":        f"translate the speech to {lang} with proper punctuation and capitalization.{kw}",
    }
    chat = [{"role": "user", "content": f"<|audio|>{texts[task]}"}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def load_model(device_arg):
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    if device == "cuda" and not torch.cuda.is_available():
        console.print("[bold red]ERROR:[/] CUDA requested but not available.")
        sys.exit(1)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}"), console=console) as prog:
        tid = prog.add_task(f"Loading [bold]{MODEL_ID}[/bold] on [bold yellow]{device}[/bold yellow] …")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, dtype=dtype, low_cpu_mem_usage=True, device_map=device,
        )
        model.eval()
        prog.update(tid, description=f"[green]Model ready on {device} ({dtype})[/]")

    return model, processor, device


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


def calibrate_noise_floor(device_index, duration=3.0):
    """Record silence and return RMS noise floor."""
    console.print(f"[cyan]Calibrating noise floor for {duration:.0f}s — stay quiet…[/]")
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
        input=True, input_device_index=device_index,
        frames_per_buffer=CHUNK_FRAMES,
    )
    rms_vals = []
    end = time.monotonic() + duration
    while time.monotonic() < end:
        raw = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        rms_vals.append(float(np.sqrt(np.mean(pcm ** 2)) * 32768))
    stream.stop_stream(); stream.close(); pa.terminate()
    floor = float(np.percentile(rms_vals, 95))  # 95th percentile of silence
    recommended = floor * 3.0                    # 3× headroom above noise
    console.print(f"[green]Noise floor (95th pct): {floor:.0f} RMS[/]")
    console.print(f"[bold green]Recommended --silence-rms: {recommended:.0f}[/]")
    return recommended


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


def inference_worker(model, processor, prompt, device, infer_in_q, infer_out_q):
    while True:
        audio_np = infer_in_q.get()
        if audio_np is None:
            break
        t0 = time.perf_counter()
        text = transcribe_chunk(model, processor, prompt, audio_np, device)
        latency = time.perf_counter() - t0
        infer_out_q.put((text, latency))


def rms_bar(rms: float, silence_rms: float, width: int = 28) -> Text:
    level   = min(int(rms / (silence_rms * 6) * width), width)
    silence = rms < silence_rms
    color   = ("bright_black" if silence
                else "green" if level < width * 0.6
                else "yellow" if level < width * 0.85
                else "red")
    bar   = Text("█" * level + "░" * (width - level), style=color)
    label = Text(f" {rms:5.0f} RMS", style="dim" if silence else "white")
    return bar + label


def make_layout(lines, rms, silence_rms, status_text, status_style, args,
                utt_frames, stats) -> Table:
    # Header
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold cyan")
    info.add_column(style="white")
    info.add_row("Model",   os.path.basename(MODEL_ID) if os.path.isdir(MODEL_ID) else MODEL_ID)
    info.add_row("Task",    args.task)
    info.add_row("Device",  args.device)
    info.add_row("Silence", f"{silence_rms:.0f} RMS")
    if args.keywords:
        info.add_row("Keywords", args.keywords)
    header = Panel(info, title="[bold cyan]Granite Speech 4.1 2B — Real-time STT[/]",
                   border_style="cyan", box=box.ROUNDED)

    # Meter + utterance buffer indicator
    utt_s = utt_frames / SAMPLE_RATE
    max_s = MAX_UTT_S
    utt_frac = min(utt_s / max_s, 1.0)
    utt_w = 28
    utt_level = int(utt_frac * utt_w)
    utt_color = "green" if utt_frac < 0.7 else "yellow" if utt_frac < 0.9 else "red"

    meter_grid = Table.grid(padding=(0, 1))
    meter_grid.add_column()
    meter_grid.add_row(Text("MIC ", style="dim") + rms_bar(rms, silence_rms))
    meter_grid.add_row(
        Text("UTT ", style="dim") +
        Text("█" * utt_level + "░" * (utt_w - utt_level), style=utt_color) +
        Text(f" {utt_s:.1f}s", style="dim")
    )
    meter_panel = Panel(meter_grid, border_style="dim", box=box.SIMPLE, title="[dim]Audio[/]")

    # Stats
    elapsed = stats["elapsed"]
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    dur_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    wpm_str = f"{stats['wpm']:.0f}" if stats["wpm"] else "—"
    lat_str = f"{stats['last_latency']*1000:.0f} ms" if stats["last_latency"] else "—"

    stat_grid = Table.grid(padding=(0, 2))
    stat_grid.add_column(style="dim")
    stat_grid.add_column(style="white")
    stat_grid.add_row("Status",   Text(status_text, style=status_style))
    stat_grid.add_row("Duration", dur_str)
    stat_grid.add_row("Words",    str(stats["word_count"]))
    stat_grid.add_row("WPM",      wpm_str)
    stat_grid.add_row("Latency",  lat_str)
    stat_panel = Panel(stat_grid, border_style="dim", box=box.SIMPLE, title="[dim]Stats[/]")

    # Transcript
    transcript = Text()
    for ts, text in lines[-MAX_LINES:]:
        transcript.append(f"[{ts}] ", style="dim cyan")
        transcript.append(text + "\n", style="bright_white")
    if not lines:
        transcript.append("(waiting for speech …)", style="dim")
    trans_panel = Panel(transcript, title="[bold]Transcript[/]",
                        border_style="green", box=box.ROUNDED, expand=True)

    root = Table.grid()
    root.add_row(header)
    root.add_row(Columns([meter_panel, stat_panel]))
    root.add_row(trans_panel)
    return root


def main():
    args = parse_args()

    if args.list_devices:
        pa = pyaudio.PyAudio()
        t = Table("Index", "Name", title="Audio Input Devices",
                  box=box.ROUNDED, header_style="bold cyan")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                t.add_row(str(i), info["name"])
        pa.terminate()
        console.print(t)
        sys.exit(0)

    silence_rms  = args.silence_rms
    silence_hold = args.silence_hold

    if args.calibrate:
        silence_rms = calibrate_noise_floor(args.input_device)
        console.print(f"\nRe-run with: [bold]--silence-rms {silence_rms:.0f}[/]\n")
        sys.exit(0)

    model, processor, device = load_model(args.device)
    args.device = device
    prompt = build_prompt(processor.tokenizer, args.task, args.lang, args.keywords)

    # Utterance-based VAD state
    # Instead of a sliding window, we accumulate frames while speech is active,
    # then flush the whole utterance to the model when silence is detected.
    utt_buf: list       = []      # current utterance frames (list of chunks)
    in_speech           = False
    silence_since       = 0.0     # monotonic time when silence started
    pad_frames          = int(SILENCE_PAD_S * SAMPLE_RATE / CHUNK_FRAMES)  # chunks to keep
    pad_buf: deque      = deque(maxlen=pad_frames)  # rolling pre-speech / tail pad

    audio_q: queue.Queue    = queue.Queue()
    infer_in_q: queue.Queue  = queue.Queue(maxsize=1)
    infer_out_q: queue.Queue = queue.Queue()
    stop_evt = threading.Event()

    lines: list[tuple[str, str]] = []
    rms_state     = [0.0]
    status        = ("Starting …", "yellow")
    inferring     = False
    session_start = time.monotonic()
    stats = {"word_count": 0, "wpm": 0.0, "last_latency": None, "elapsed": 0.0}

    threading.Thread(
        target=audio_thread, args=(audio_q, stop_evt, args.input_device), daemon=True
    ).start()
    threading.Thread(
        target=inference_worker,
        args=(model, processor, prompt, device, infer_in_q, infer_out_q),
        daemon=True,
    ).start()

    def flush_utterance():
        """Send the accumulated utterance buffer to the inference worker."""
        if not utt_buf or inferring:
            return False
        audio_np = np.concatenate(utt_buf, axis=0).astype(np.float32)
        # Clamp to MAX_UTT_S to avoid OOM on very long utterances
        max_frames = int(MAX_UTT_S * SAMPLE_RATE)
        if len(audio_np) > max_frames:
            audio_np = audio_np[-max_frames:]
        try:
            infer_in_q.put_nowait(audio_np)
            return True
        except queue.Full:
            return False

    with Live(
        make_layout(lines, 0.0, silence_rms, "Starting …", "yellow", args, 0, stats),
        console=console, refresh_per_second=10,
    ) as live:
        try:
            while True:
                # Drain audio queue
                while not audio_q.empty():
                    chunk = audio_q.get_nowait()
                    chunk_rms = float(np.sqrt(np.mean(chunk ** 2)) * 32768)
                    rms_state[0] = chunk_rms
                    is_loud = chunk_rms >= silence_rms

                    if is_loud:
                        if not in_speech:
                            # Speech onset — prepend the tail pad for context
                            in_speech = True
                            utt_buf = list(pad_buf) + [chunk]
                        else:
                            utt_buf.append(chunk)
                        silence_since = 0.0
                    else:
                        pad_buf.append(chunk)
                        if in_speech:
                            utt_buf.append(chunk)  # keep trailing silence in utterance
                            if silence_since == 0.0:
                                silence_since = time.monotonic()
                            held = time.monotonic() - silence_since
                            max_held = len(utt_buf) * CHUNK_FRAMES / SAMPLE_RATE >= MAX_UTT_S
                            if held >= silence_hold or max_held:
                                # End of utterance — flush
                                if not inferring:
                                    if flush_utterance():
                                        inferring = True
                                        status = ("Transcribing …", "yellow")
                                utt_buf = []
                                in_speech = False
                                silence_since = 0.0

                stats["elapsed"] = time.monotonic() - session_start

                # Collect inference result
                if not infer_out_q.empty():
                    text, latency = infer_out_q.get_nowait()
                    stats["last_latency"] = latency
                    inferring = False
                    if text:
                        ts = time.strftime("%H:%M:%S")
                        lines.append((ts, text))
                        stats["word_count"] += len(text.split())
                        elapsed_min = stats["elapsed"] / 60
                        stats["wpm"] = stats["word_count"] / elapsed_min if elapsed_min > 1 else 0.0
                    status = ("Listening …", "green")

                # Status update based on VAD state
                if in_speech and status[1] != "yellow":
                    status = ("Speech detected …", "cyan")
                elif not in_speech and not inferring and status[0] not in ("Listening …",):
                    status = ("Listening …", "green")

                utt_frames = sum(len(c) for c in utt_buf)
                live.update(make_layout(
                    lines, rms_state[0], silence_rms, status[0], status[1], args,
                    utt_frames, stats,
                ))
                time.sleep(0.05)

        except KeyboardInterrupt:
            utt_frames = sum(len(c) for c in utt_buf)
            live.update(make_layout(
                lines, 0.0, silence_rms, "Stopped.", "red", args, utt_frames, stats,
            ))

    stop_evt.set()
    infer_in_q.put(None)
    console.print("\n[bold green]Session ended.[/] Goodbye!")


if __name__ == "__main__":
    main()
