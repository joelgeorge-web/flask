import requests
import time

def check_cbse_results():
  """Checks if the CBSE results page is accessible."""
  url = "https://results.cbse.nic.in/"
  try:
    response = requests.get(url)
    if response.status_code == 200:
      print("The CBSE results page is accessible!")
    else:
      print(f"The page returned status code: {response.status_code}")
  except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

# Simulate refreshing by checking every 5 seconds  
while True:
  check_cbse_results()
  time.sleep(1)
