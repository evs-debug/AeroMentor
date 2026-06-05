import subprocess
import os

DOCUMENTS = {
    "v1": "v1_speed.txt",
    "lift": "lift.txt",
    "drag": "drag.txt",
    "thrust": "thrust.txt",
    "stall": "stall.txt",

    "a320": "a320.txt",
    "a350": "a350.txt",
    "787": "b787.txt",
    "737": "b737.txt",
    "777": "b777.txt",
    "747": "b747.txt",
    "a380": "a380.txt",

    "takeoff": "takeoff.txt",
    "landing": "landing.txt",
    "go around": "go_around.txt",
    "crosswind": "crosswind_landing.txt",

    "ils": "ils.txt",
    "vor": "vor.txt",
    "gps": "gps_navigation.txt",

    "atc": "atc.txt",
    "runway": "runways.txt",
    "taxiway": "taxiways.txt",
    "holding": "holding_pattern.txt",

    "hydraulic": "hydraulics.txt",
    "fly by wire": "fly_by_wire.txt",
    "apu": "apu.txt",
    "landing gear": "landing_gear.txt",

    "engine": "jet_engines.txt",
    "turbofan": "turbofan.txt",
    "turbojet": "turbojet.txt",
    "turboprop": "turboprop.txt",

    "angle of attack": "angle_of_attack.txt",
    "weight": "weight.txt",
    "winglet": "winglets.txt"
}


while True:
    question = input("\nAsk AeroMentor: ")

    if question.lower() == "exit":
        break

    context = ""

    for keyword, filename in DOCUMENTS.items():
        if keyword in question.lower():
            path = os.path.join("data", filename)

            with open(path, "r") as file:
                context = file.read()

            break

    prompt = f"""
You are AeroMentor, an expert aviation instructor and aviation encyclopedia.

Always assume aviation context.

Context:
{context}

User Question:
{question}
"""

    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True,
        text=True
    )

    print("\nAeroMentor:")
    print(result.stdout)