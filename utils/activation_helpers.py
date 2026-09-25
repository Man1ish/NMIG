import subprocess
import json

class ActivationHelper:
    def __init__(self, wsk_cmd='wsk'):
        self.wsk_cmd = wsk_cmd

    def get_activation(self, activation_id):
        """
        Retrieves and parses activation data for a given activation ID.
        """
        try:
            result = subprocess.run(
                [self.wsk_cmd, 'activation', 'get', activation_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )

            # Extract the part after the first `{` to the last `}` to isolate JSON
            output = result.stdout
            json_start = output.find('{')
            json_str = output[json_start:]
            activation_data = json.loads(json_str)

            return activation_data

        except subprocess.CalledProcessError as e:
            print(f"[Error] wsk CLI failed: {e.stderr}")
            return None
        except json.JSONDecodeError as e:
            print(f"[Error] Failed to parse JSON: {e}")
            return None

    def get_response_result(self, activation_id):
        """
        Extracts the 'response -> result' field from the activation.
        """
        activation = self.get_activation(activation_id)
        if activation and 'response' in activation and 'result' in activation['response']:
            return activation['response']['result']
        else:
            print("Could not extract result from activation.")
            return None


if __name__ == "__main__":
    helper = ActivationHelper()
    activation_id = "4d24b941b9064b42a4b941b9062b4232"
    result = helper.get_activation(activation_id)
    name = result['name']
    inference_time = result['response']['result']['measurement']['inference_time_ms']
    print(inference_time)
