from camera import run, run_test

print("=== ASL Recognition ===")
print("")
print("Select mode:")
print("  1 — Normal subtitles mode")
print("  2 — Accuracy self-test mode")
print("")

choice = input("Enter 1 or 2: ").strip()

if choice == "2":
    print("")
    print("=== Self-Test Mode ===")
    print("A letter will appear on screen. Sign it before the timer runs out.")
    print("  ESC — quit")
    print("")
    print("Starting test...")
    run_test()
else:
    print("")
    print("=== Normal Mode ===")
    print("Keyboard shortcuts (click the camera window first):")
    print("  SPACEBAR  — insert a space")
    print("  BACKSPACE — delete last character")
    print("  C         — clear all subtitle text")
    print("  T         — translate subtitle to Ukrainian")
    print("  ESC       — quit")
    print("")
    print("Starting camera...")
    run()
