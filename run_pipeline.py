import yaml
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def check_checkpoint_1():
    config_path = os.path.join("configs", "experiment.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return False
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
        print(f"--- {config['project']['name']} ---")
        print(f"Project: {config['project']['name']}")
        print(f"Status: {config['project']['status']}")
        print(f"Checkpoint: {config['project']['checkpoint']}")
        print(f"Research Question: {config['research']['research_question'].strip()}")
        print("\nCheckpoint 1 validation successful.")
        print("Note: No data ingestion, model loading, or generation is performed in this phase.")
        return True
    except Exception as e:
        print(f"Error parsing configuration: {e}")
        return False

if __name__ == "__main__":
    success = check_checkpoint_1()
    if not success:
        sys.exit(1)
