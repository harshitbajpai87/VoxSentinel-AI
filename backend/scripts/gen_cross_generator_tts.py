"""Generate a multi-voice, multi-language edge-tts synthetic corpus.

Why: the shipped detector was only ever evaluated on the synthesis methods in
its own training data. To measure generalization we need a synthetic corpus
that spans several TTS voices (and two languages) so that whole voices can be
held out from training.

Design:
  - every clip records the exact voice that produced it (`edge-tts:<voice>`);
  - voices are partitioned into TRAIN / VALID / EVAL groups, and the eval group
    is never generated... no: it is generated here but consumed only by the
    held-out manifest (the split is enforced by the manifest builder);
  - text variety (24 English + 12 Hindi sentences) plus rate/pitch jitter, so
    the classifier cannot memorise one fixed prosody;
  - durations are measured and out-of-range clips (too short/long) are
    reported and excluded by the manifest builder, never silently kept.

Usage:
    python scripts/gen_cross_generator_tts.py --out-dir /tmp/av/generator_v4/tts
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

EN_TEXTS = [
    "Hello, this is a call from your bank fraud department. We have detected a suspicious transaction on your account and need to verify your identity immediately.",
    "Your parcel has arrived at the local depot but the delivery address could not be confirmed. Please call the number on your screen to arrange a new time.",
    "I am calling about the accident you were involved in last month. You may be entitled to compensation for the injuries you suffered.",
    "This is an automated reminder that your electricity connection will be disconnected tonight unless the outstanding balance is paid.",
    "Good morning, I would like to book a table for four people at seven o'clock this evening, if that is still possible.",
    "The weather forecast says it will rain heavily tomorrow, so please remember to carry an umbrella and leave a little earlier.",
    "Please listen carefully because our menu options have recently changed and the previous extension numbers no longer work.",
    "We are pleased to inform you that your application has been approved and the funds will be transferred within three working days.",
    "Can you send me the report before Friday? I need to review the numbers with the finance team before the board meeting.",
    "The train to Chennai has been delayed by forty minutes due to heavy fog in the northern section of the route.",
    "Your subscription will renew automatically next week. If you wish to cancel, please update your preferences in the account settings.",
    "I am sorry, I think you have the wrong number. There is nobody by that name living at this address.",
    "Our records show that your insurance policy expired last month. Please contact your agent to discuss renewal options.",
    "The doctor has reviewed your test results and would like to see you again next Tuesday afternoon for a follow up.",
    "Thank you for calling customer support. All of our agents are currently busy, but your call is important to us.",
    "There is a problem with the payment you submitted yesterday. The card was declined by the issuing bank.",
    "We found your documents near the bus stop and would like to return them. Please tell us where we can meet you.",
    "The annual general meeting will take place on the twenty second of next month in the main conference hall.",
    "I wanted to confirm the timings for tomorrow's interview. The panel will meet you at half past ten in the morning.",
    "Unusual login activity was detected on your account from a device in another city. Please reset your password now.",
    "Would you be able to help me move some furniture this weekend? I will arrange lunch for everyone who comes.",
    "The library will remain closed for maintenance on Monday and Tuesday, and all due dates will be extended by one week.",
    "Our team has prepared a quotation for the renovation work and we can begin as soon as you approve the estimate.",
    "Please note that the office will operate with a holiday schedule next week and responses may be slower than usual.",
]

HI_TEXTS = [
    "नमस्ते, मैं आपके बैंक से बात कर रहा हूँ। आपके खाते में एक संदिग्ध लेनदेन हुआ है और हमें तुरंत पुष्टि करनी है।",
    "आपका पार्सल आज शाम तक पहुँच जाएगा, कृपया घर पर किसी को उपलब्ध रखें।",
    "मुझे बताया गया है कि आपके पिछले महीने के बिजली के बिल का भुगतान नहीं हुआ है।",
    "क्या आप कृपया मुझे अगले सप्ताह की बैठक का समय बता सकते हैं?",
    "आज मौसम बहुत अच्छा है, हम शाम को पार्क में टहलने चल सकते हैं।",
    "आपकी शिकायत दर्ज कर ली गई है और दो कार्य दिवसों में समाधान मिल जाएगा।",
    "मैं आपको याद दिलाना चाहता हूँ कि कल सुबह नौ बजे हमारी परीक्षा है।",
    "कृपया अपना पता और फोन नंबर दोबारा बता दीजिए ताकि हम सही जगह पर डिलीवरी कर सकें।",
    "अस्पताल ने आपकी रिपोर्ट भेज दी है, डॉक्टर कल दोपहर आपसे मिलना चाहते हैं।",
    "हमारी कंपनी ने नया उत्पाद लॉन्च किया है और उस पर पचास प्रतिशत की छूट चल रही है।",
    "ट्रेन चालीस मिनट लेट है, कृपया प्लेटफॉर्म पर ही प्रतीक्षा करें।",
    "आपका आवेदन स्वीकृत हो गया है और राशि तीन कार्य दिवसों में भेज दी जाएगी।",
]

# Voice partitions. EVAL voices are never used by the train/validation manifests.
VOICES: dict[str, list[str]] = {
    "train": [
        "en-US-AriaNeural",
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-IN-NeerjaNeural",
    ],
    "validation": [
        "en-US-EmmaNeural",
        "en-GB-ThomasNeural",
    ],
    "eval": [
        # Unseen voices, unseen language (Hindi) -> hardest generalization case.
        "hi-IN-MadhurNeural",
        "hi-IN-SwaraNeural",
        # Unseen voice in the same locale as a seen voice (en-IN).
        "en-IN-PrabhatNeural",
        # Unseen English voices from locales never seen in training.
        "en-US-AvaNeural",
        "en-AU-NatashaNeural",
    ],
}

CLIPS_PER_VOICE = {"train": 48, "validation": 24, "eval": 24}
RATES = ["-10%", "+0%", "+10%", "+20%"]


def is_hindi(voice: str) -> bool:
    return voice.startswith("hi-")


async def _one(voice: str, index: int, out_path: Path, rng: random.Random,
               sem: asyncio.Semaphore, errors: list[str]) -> None:
    import edge_tts  # noqa: PLC0415

    texts = HI_TEXTS if is_hindi(voice) else EN_TEXTS
    text = texts[index % len(texts)]
    rate = RATES[index % len(RATES)]
    pitch = rng.choice(["-2Hz", "+0Hz", "+2Hz"])
    async with sem:
        try:
            await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(out_path))
        except Exception as exc:  # noqa: BLE001 - recorded, never silently swallowed
            errors.append(f"{voice}#{index}: {type(exc).__name__}: {exc}")


async def main_async(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    errors: list[str] = []
    tasks = []
    planned: list[dict[str, object]] = []
    for split, voices in VOICES.items():
        for voice in voices:
            for index in range(CLIPS_PER_VOICE[split]):
                path = out_dir / f"{voice}_{index:02d}.mp3"
                planned.append({
                    "path": str(path),
                    "voice": voice,
                    "split": split,
                    "generator_id": f"edge-tts:{voice}",
                    "language": "hi" if is_hindi(voice) else "en",
                })
                if path.is_file() and path.stat().st_size > 0 and args.resume:
                    continue
                tasks.append(_one(voice, index, path, rng, sem, errors))
    print(f"Generating {len(tasks)} clips ({len(planned)} planned) into {out_dir} ...")
    await asyncio.gather(*tasks)

    index_path = out_dir.parent / "tts_index.json"
    index_path.write_text(json.dumps({"clips": planned, "errors": errors}, indent=2), encoding="utf-8")
    print(f"Index: {index_path}")
    if errors:
        print(f"WARNING: {len(errors)} generation failures (first 3: {errors[:3]})", file=sys.stderr)
    # Fail loudly if an entire voice is missing: a silently absent eval voice
    # would turn "generalization to unseen voices" into a lie.
    missing = [p["path"] for p in planned
               if not Path(str(p["path"])).is_file() or Path(str(p["path"])).stat().st_size == 0]
    if missing:
        print(f"ERROR: {len(missing)} planned clips are missing/empty.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate multi-voice edge-tts corpus.")
    parser.add_argument("--out-dir", default="/tmp/av/generator_v4/tts")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
