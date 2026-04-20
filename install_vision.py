import sys
import ollama

def download_llava():
    print("Initializing download for Llava Vision Model (4.7 GB)...")
    print("This may take several minutes depending on your internet connection.\n")
    
    current_digest = ""
    for response in ollama.pull('llava', stream=True):
        status = response.get('status', '')
        digest = response.get('digest', '')
        total = response.get('total') or 0
        completed = response.get('completed') or 0
        
        if total > 0:
            percent = (completed / total) * 100
            # Print a neat progress percentage
            sys.stdout.write(f"\r{status} [{percent:.1f}%] - {completed/1e6:.1f}MB / {total/1e6:.1f}MB   ")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r{status}                                        ")
            sys.stdout.flush()
            
    print("\n\nLlava successfully installed and ready for Vision Extraction!")

if __name__ == "__main__":
    download_llava()
