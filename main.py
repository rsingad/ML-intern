import os
import subprocess

def run_script(script_name):
    print(f"\n{'='*40}")
    print(f"Running: {script_name}")
    print(f"{'='*40}\n")
    try:
        subprocess.run(["./venv/bin/python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
    except FileNotFoundError:
        print(f"Virtual environment or {script_name} not found. Ensure './venv/bin/python' exists.")

def main():
    while True:
        print("\n--- Restaurant Data Analysis Project ---")
        print("1. Predict Restaurant Ratings")
        print("2. Cuisine Classification")
        print("3. Restaurant Recommendation System")
        print("4. Location-based Analysis")
        print("5. Run All Analysis")
        print("q. Quit")
        
        choice = input("\nChoose an option (1-5 or q): ").strip().lower()
        
        if choice == '1':
            run_script("predict_ratings.py")
        elif choice == '2':
            run_script("classify_cuisines.py")
        elif choice == '3':
            run_script("recommend_restaurants.py")
        elif choice == '4':
            run_script("location_analysis.py")
        elif choice == '5':
            run_script("predict_ratings.py")
            run_script("classify_cuisines.py")
            run_script("recommend_restaurants.py")
            run_script("location_analysis.py")
        elif choice == 'q':
            print("Exiting project...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
