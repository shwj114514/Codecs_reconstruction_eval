import argparse
import os
from pathlib import Path
import typing as tp
import torchaudio
from tqdm import tqdm

def get_filelist(folder: tp.Union[str, os.PathLike], extensions: tp.Optional[tp.List[str]] = None) -> tp.List[str]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    extensions = set(ext.lower() for ext in extensions)
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    return [str(file) for file in path.rglob('*') if file.suffix.lower() in extensions]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    INPUT_FOLDER = "exp_recon/test-clean_flac"
    OUT_FOLDER = "exp_recon/test-clean"
    args = parser.parse_args()
    os.makedirs(OUT_FOLDER, exist_ok=True)
    filelist = get_filelist(INPUT_FOLDER)
    for f in tqdm(filelist):
        waveform, sr = torchaudio.load(f)
        outpath = Path(OUT_FOLDER) / (Path(f).stem + ".wav")
        torchaudio.save(str(outpath), waveform, sr)