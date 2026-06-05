import subprocess

while True:
    question = input("\nAsk Aviation AI: ")

    if question.lower() == "exit":
        break

    result = subprocess.run(
        ["ollama", "run", "llama3", question],
        capture_output=True,
        text=True
    )

    print("\nAI:")
    print(result.stdout)