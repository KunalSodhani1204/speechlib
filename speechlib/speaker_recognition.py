from speechbrain.pretrained import SpeakerRecognition
import os
from pydub import AudioSegment
from collections import defaultdict
import torch

if torch.cuda.is_available():
    verification = SpeakerRecognition.from_hparams(run_opts={"device":"cuda"}, source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
else:
    verification = SpeakerRecognition.from_hparams(run_opts={"device":"cpu"}, source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

# recognize speaker name
def speaker_recognition(file_name, voices_folder, segments, wildcards, min_confidence=0.8):
    speakers = os.listdir(voices_folder)
    Id_count = defaultdict(int)
    audio = AudioSegment.from_file(file_name, format="wav")
    folder_name = "temp"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    i = 0
    limit = 60  # limit in seconds
    duration = 0

    for segment in segments:
        start = segment[0] * 1000  # start time in milliseconds
        end = segment[1] * 1000  # end time in milliseconds
        clip = audio[start:end]
        i += 1
        temp_file = os.path.join(folder_name, f"{file_name.split('/')[-1].split('.')[0]}_segment{i}.wav")
        clip.export(temp_file, format="wav")

        max_score = 0
        person = "unknown"  # Default assignment for unknown speaker

        for speaker in speakers:
            voices = os.listdir(os.path.join(voices_folder, speaker))
            for voice in voices:
                voice_file = os.path.join(voices_folder, speaker, voice)

                try:
                    # Compare voice file with audio segment
                    score, prediction = verification.verify_files(voice_file, temp_file)
                    prediction = prediction[0].item()
                    score = score[0].item()

                    if prediction and score >= max_score and score >= min_confidence:
                        max_score = score
                        speakerId = speaker.split(".")[0]
                        if speakerId not in wildcards:  # Ensure no conflict with wildcards
                            person = speakerId
                except Exception as err:
                    print("Error occurred while speaker recognition:", err)

        Id_count[person] += 1

        # Delete the temp WAV file after processing
        os.remove(temp_file)

        # Update current prediction and check duration limit
        current_pred = max(Id_count, key=Id_count.get)
        duration += (end - start)
        if duration >= limit and current_pred != "unknown":
            break

    # Determine the most common ID
    most_common_Id = max(Id_count, key=Id_count.get)
    
    # Ensure "unknown" is returned if no valid speaker is identified
    if Id_count[most_common_Id] == 1 and most_common_Id == "unknown":
        return "unknown"

    return most_common_Id
