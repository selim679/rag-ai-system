from rag.pipeline import generate_answer

while True:
    query = input("\nAsk something: ")

    if query == "exit":
        break

    result = generate_answer(query)

    print("\n ANSWER:")
    print(result["answer"])

    print("\n📚 SOURCES:\n")
    for i, s in enumerate(result["sources"]):
      print(f"[{i+1}] {s[:300]}")

   

