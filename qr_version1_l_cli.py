"""Generate a numeric Version 1-L QR matrix from command-line byte-mode input.

Uses ISO-8859-1, Reed-Solomon error correction, and mask pattern 0.
QR rules and constants: https://www.thonky.com/qr-code-tutorial/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from reedsolo import RSCodec


EXAMPLES = (
    "known",
    "We've succeeded!",
    "~¡256_-_aA&ñ",
    "From α to ɷ...",
    "Sugarplum_Fairy_Nightmare",
)
SIZE = 21
DATA_CODEWORDS = 19
ECC_CODEWORDS = 7
MAX_BYTES = 17


def rows(matrix: np.ndarray) -> str:
    return "\n".join(" ".join(map(str, row)) for row in matrix)


def encode_data(raw: bytes) -> tuple[bytes, list[str]]:
    bits = "0100" + f"{len(raw):08b}" + "".join(f"{byte:08b}" for byte in raw)
    steps = [
        "Mode indicator: 0100 (byte)",
        f"Character count indicator: {len(raw):08b} ({len(raw)} bytes)",
        "Input data bits: " + " ".join(f"{byte:08b}" for byte in raw),
    ]
    bits += "0" * min(4, DATA_CODEWORDS * 8 - len(bits))
    steps.append(f"After terminator: {bits}")
    bits += "0" * (-len(bits) % 8)
    steps.append(f"After byte alignment: {bits}")
    codewords = bytearray(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    for pad in (0xEC, 0x11) * DATA_CODEWORDS:
        if len(codewords) == DATA_CODEWORDS:
            break
        codewords.append(pad)
    steps.append("19 data codewords: " + codewords.hex(" "))
    return bytes(codewords), steps


def format_bits(ecc_level: str, mask_pattern: int) -> int:
    """Build the 15 format bits from the selected ECC level and mask pattern."""
    ecc_codes = {"L": 0b01, "M": 0b00, "Q": 0b11, "H": 0b10}
    if ecc_level not in ecc_codes or not 0 <= mask_pattern <= 7:
        raise ValueError("Invalid QR format parameters")
    value = (ecc_codes[ecc_level] << 3) | mask_pattern
    remainder = value << 10
    for shift in range(4, -1, -1):
        if remainder & (1 << (shift + 10)):
            remainder ^= 0x537 << shift
    return ((value << 10) | remainder) ^ 0x5412


def matrix_for(codewords: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.full((SIZE, SIZE), -1, dtype=np.int8)
    reserved = np.zeros((SIZE, SIZE), dtype=bool)

    def fixed(row: int, col: int, color: int) -> None:
        matrix[row, col] = color
        reserved[row, col] = True

    for top, left in ((0, 0), (0, 14), (14, 0)):
        for dy in range(-1, 8):
            for dx in range(-1, 8):
                row, col = top + dy, left + dx
                if not (0 <= row < SIZE and 0 <= col < SIZE):
                    continue
                inside = 0 <= dy <= 6 and 0 <= dx <= 6
                black = inside and (dy in (0, 6) or dx in (0, 6) or
                                    2 <= dy <= 4 and 2 <= dx <= 4)
                fixed(row, col, int(black))

    for index in range(8, 13):
        fixed(6, index, int(index % 2 == 0))
        fixed(index, 6, int(index % 2 == 0))

    # Format slots are reserved before data placement; version 1 has no alignment pattern.
    for index in (0, 1, 2, 3, 4, 5, 7, 8):
        fixed(8, index, 0)
        fixed(index, 8, 0)
    for index in range(13, SIZE):
        fixed(8, index, 0)
        fixed(index, 8, 0)
    fixed(13, 8, 1)  # Permanent dark module.

    bits = "".join(f"{word:08b}" for word in codewords)
    next_bit = 0
    upward = True
    for right in (20, 18, 16, 14, 12, 10, 8, 5, 3, 1):
        positions = range(SIZE - 1, -1, -1) if upward else range(SIZE)
        for row in positions:
            for col in (right, right - 1):
                if not reserved[row, col]:
                    if next_bit >= len(bits):
                        raise ValueError("More data cells than codeword bits")
                    matrix[row, col] = int(bits[next_bit])
                    next_bit += 1
        upward = not upward
    if next_bit != 208 or (matrix < 0).any():
        raise ValueError(f"Matrix placement incomplete: {next_bit}/208 bits")

    before = matrix.copy()
    mask = np.fromfunction(lambda row, col: (row + col) % 2 == 0,
                           (SIZE, SIZE), dtype=int).astype(bool) & ~reserved
    matrix[mask] ^= 1

    fmt = format_bits("L", 0)
    for index in range(6):
        fixed(index, 8, (fmt >> index) & 1)
    fixed(7, 8, (fmt >> 6) & 1)
    fixed(8, 8, (fmt >> 7) & 1)
    fixed(8, 7, (fmt >> 8) & 1)
    for index in range(9, 15):
        fixed(8, 14 - index, (fmt >> index) & 1)
    for index in range(8):
        fixed(8, SIZE - 1 - index, (fmt >> index) & 1)
    for index in range(8, 15):
        fixed(SIZE - 15 + index, 8, (fmt >> index) & 1)
    fixed(13, 8, 1)
    return before, mask.astype(np.int8), matrix


def generate_report(value: str) -> tuple[str, np.ndarray | None]:
    log = [f"Input: {value!r}", "Version: 1; ECC: L; byte mode; mask: 0",
           "Character set: ISO-8859-1"]
    try:
        raw = value.encode("iso-8859-1", errors="strict")
    except UnicodeEncodeError as error:
        character = value[error.start]
        prefix = value[:error.start].encode("iso-8859-1")
        log += [f"Encoding stopped at index {error.start}: {character!r} "
                f"(U+{ord(character):04X}) is not in ISO-8859-1.",
                f"Successfully encoded prefix ({len(prefix)} bytes): {prefix.hex(' ')}",
                "Prefix bits: " + " ".join(f"{byte:08b}" for byte in prefix),
                "No complete data codewords, ECC, or matrix were produced."]
        return "\n".join(log), None
    log += [f"Input bytes ({len(raw)}): {raw.hex(' ')}",
            "Input bits: " + " ".join(f"{byte:08b}" for byte in raw)]
    if len(raw) > MAX_BYTES:
        log += [f"Capacity stopped: {len(raw)} bytes exceed the "
                f"Version 1-L byte-mode limit of {MAX_BYTES}.",
                "No data codewords, ECC, or matrix were produced."]
        return "\n".join(log), None

    data, steps = encode_data(raw)
    log += steps
    combined = bytes(RSCodec(ECC_CODEWORDS).encode(data))
    if len(combined) != DATA_CODEWORDS + ECC_CODEWORDS:
        raise ValueError("Unexpected Reed-Solomon codeword count")
    log += ["7 ECC codewords (reedsolo): " + combined[DATA_CODEWORDS:].hex(" "),
            "26 combined codewords: " + combined.hex(" ")]
    before, mask, final = matrix_for(combined)
    log += ["Matrix before mask (0=white, 1=black; reserved format slots are 0):",
            rows(before), "Mask 0 on data cells only (1 means toggle):", rows(mask),
            f"Format bits for L/0: {format_bits('L', 0):015b}",
            "Final 21x21 numeric matrix:", rows(final)]
    return "\n".join(log), final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path,
                        help="Write demo_1.txt through demo_5.txt")
    args = parser.parse_args()
    if args.demo_dir is not None:
        args.demo_dir.mkdir(parents=True, exist_ok=True)
        for number, example in enumerate(EXAMPLES, 1):
            path = args.demo_dir / f"demo_{number}.txt"
            report, _ = generate_report(example)
            path.write_text(report + "\n", encoding="utf-8")
            print(path)
    else:
        report, _ = generate_report(input("Text to encode: "))
        print(report)


if __name__ == "__main__":
    main()
