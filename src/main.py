from engine import SpeculativeEngine

def main():
    decoder = SpeculativeEngine()
    
    prompt = "Artificial Intelligence is going to"
    
    text = decoder.generate(prompt, max_new_tokens=30, gamma=4)
    
    print("\n--- Final Output ---")
    print(text)

if __name__ == "__main__":
    main()