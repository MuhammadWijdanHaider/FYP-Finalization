import librosa
from .contemp import retTempFile

async def audio_processing(file_content, required):
    aud_file = file_content
    if required["mint"]:
        aud_file_r = await retTempFile(file_content, suffixg=required["filename"])
        pass
    else:
        pass
    pass
