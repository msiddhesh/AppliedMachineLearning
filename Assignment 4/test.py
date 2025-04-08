from score import score
import requests
import time
import subprocess
import warnings
import pytest
import subprocess
import time
import requests
warnings.filterwarnings("ignore")

def test_docker():
    """Test that the Docker container works correctly."""
    # Build the Docker image
    subprocess.run(["docker", "build", "-t", "flask-spam-classifier", "."], check=True)
    
    # Run the Docker container
    container = subprocess.Popen(
        ["docker", "run", "-d", "-p", "5000:5000", "--name", "flask-test-container", "flask-spam-classifier"],
        stdout=subprocess.PIPE
    )
    container_id = container.stdout.read().decode('utf-8').strip()
    
    # Wait for the container to start
    time.sleep(3)
    
    try:
        # Test sample request
        sample_text = "Congratulations! You've won a free vacation to Hawaii. Call now!"
        response = requests.post(
            "http://localhost:5000/",
            json={"text": sample_text}
        )
        
        # Check if response is as expected
        assert response.status_code == 200
        result = response.json()
        assert "prediction" in result or "score" in result
        
        # Additional tests for the container
        # Test the home page
        home_response = requests.get("http://localhost:5000/")
        assert home_response.status_code == 200
        
    finally:
        # Stop and remove the container
        subprocess.run(["docker", "stop", container_id], check=True)
        subprocess.run(["docker", "rm", container_id], check=True)