import os
import shutil
import sys
from pathlib import Path
import typing as tp

def get_filelist(
    folder: tp.Union[str, os.PathLike],
    extensions: tp.Optional[tp.List[str]] = None
) -> tp.List[str]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    extensions = set(ext.lower() for ext in extensions)
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    return [str(file) for file in path.rglob('*') if file.suffix.lower() in extensions]

if __name__ == '__main__':
    INPUT_FOLDER = "exp_recon/tmp/LibriSpeech/test-clean"
    OUTPUT_FOLDER = "exp_recon/test-clean"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filelist = get_filelist(INPUT_FOLDER)
    print(f"processing filelist = len{len(filelist)} in {INPUT_FOLDER} to {OUTPUT_FOLDER}")
    for file_path in filelist:
        audio_name = str(Path(file_path).name)
        shutil.copyfile(file_path, os.path.join(OUTPUT_FOLDER, audio_name))

