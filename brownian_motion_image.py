"""Genera imágenes de movimiento browniano fraccionario (fBm).

El exponente de Hurst (H) controla la rugosidad del ruido (entre 0 y 1)

Uso:
    python brownian_motion_image.py --width 512 --height 512 --seed 42 --hurst 0.7
"""

import argparse

import numpy as np
from PIL import Image


def generate_fbm(width, height, seed=None, hurst=0.7, scale=1.0, low_freq_cutoff=1e-3):
    """Genera un campo 2D de movimiento browniano fraccionario.

    Parameters
    ----------
    width, height : int
        Dimensiones de la imagen en píxeles.
    seed : int or None
        Semilla para reproducibilidad. None => aleatorio.
    hurst : float
        Exponente de Hurst en (0, 1). Controla la rugosidad del ruido.
    scale : float
        Factor de amplitud aplicado al campo antes de normalizar.
    low_freq_cutoff : float
        Frecuencia mínima considerada, para evitar división por cero y
        limitar el crecimiento de las componentes de muy baja frecuencia.
    """
    rng = np.random.default_rng(seed)

    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    freq = np.sqrt(fx**2 + fy**2)
    freq[freq < low_freq_cutoff] = low_freq_cutoff

    # Densidad espectral de potencia de fBm ~ f^-(2H+2) => amplitud ~ f^-(H+1)
    amplitude = freq ** -(hurst + 1.0)
    amplitude[0, 0] = 0.0 

    white_noise = rng.normal(size=(height, width)) + 1j * rng.normal(size=(height, width))
    field = np.fft.ifft2(white_noise * amplitude).real

    return field * scale


def normalize_to_uint8(field, percentile_clip=0.0):
    """Normaliza un campo float a valores uint8 en [0, 255].

    percentile_clip: si > 0, recorta ese percentil en ambos extremos antes
    de normalizar (útil para aumentar el contraste ignorando outliers).
    """
    if percentile_clip > 0:
        lo, hi = np.percentile(field, [percentile_clip, 100 - percentile_clip])
    else:
        lo, hi = field.min(), field.max()

    if hi - lo < 1e-12:
        return np.zeros_like(field, dtype=np.uint8)

    normalized = np.clip((field - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def save_image(field, path, percentile_clip=0.0):
    gray = normalize_to_uint8(field, percentile_clip=percentile_clip)

    img = Image.fromarray(gray, mode="L")

    img.save(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genera imágenes de movimiento browniano fraccionario (fBm)."
    )
    parser.add_argument("--width", type=int, default=512, help="Ancho de la imagen en píxeles.")
    parser.add_argument("--height", type=int, default=512, help="Alto de la imagen en píxeles.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla para reproducir la misma imagen."
    )
    parser.add_argument(
        "--hurst",
        type=float,
        default=0.7,
        help="Exponente de Hurst en (0,1). Menor = más rugoso, mayor = más suave.",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Factor de amplitud del ruido antes de normalizar."
    )
    parser.add_argument(
        "--percentile-clip",
        type=float,
        default=0.0,
        help="Percentil (0-49) a recortar en ambos extremos para aumentar el contraste.",
    )
    parser.add_argument(
        "--output", type=str, default="brownian_motion.png", help="Ruta del archivo de salida."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.hurst < 1.0):
        raise ValueError("--hurst debe estar en el intervalo abierto (0, 1).")

    field = generate_fbm(
        args.width,
        args.height,
        seed=args.seed,
        hurst=args.hurst,
        scale=args.scale,
    )
    save_image(field, args.output, percentile_clip=args.percentile_clip)

    print(
        f"Imagen guardada en {args.output} "
        f"({args.width}x{args.height}, seed={args.seed}, hurst={args.hurst}, scale={args.scale})"
    )


if __name__ == "__main__":
    main()
