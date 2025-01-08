import json
import requests

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))
        
        # Define the GitHub API URL
        github_api_url = "https://api.github.com/repos/kpavan2004/driver_demand_prediction_capstone/dispatches"
        
        # Define the headers
        headers = {
            "Authorization": "Bearer ${{ secrets.MLFLOW_TRACKING_URI }}",
            "Content-Type": "application/json"
        }
        
        # Define the payload
        payload = {
            "event_type": "grafana_trigger"
        }
        
        # Make the POST request to GitHub API with a timeout
        response = requests.post(github_api_url, headers=headers, json=payload, timeout=60)
        
        # Log the response
        print(f"GitHub API Response: {response.status_code}")
        print(f"Response Body: {response.text}")
        
        if response.status_code == 204:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "GitHub dispatch event triggered successfully."})
            }
        else:
            return {
                "statusCode": response.status_code,
                "body": json.dumps({"error": "Failed to trigger GitHub dispatch event.", "details": response.text})
            }
    
    except requests.exceptions.Timeout:
        print("Request to GitHub API timed out.")
        return {
            "statusCode": 504,
            "body": json.dumps({"error": "Request to GitHub API timed out."})
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "An exception occurred.", "details": str(e)})
        }
