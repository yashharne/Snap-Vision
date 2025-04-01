from flask import Flask, request, jsonify
import torch
import numpy as np
import option
from model import Model
from dataset import Dataset

app = Flask(__name__)

args = option.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model_path = './ckpt/' + args.model_name + 'final.pkl'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@app.route('/snap-video', methods=['POST'])
def snap_video():
    try:
        data = request.get_json(force=True) 
        input_data = data['input']

        input_tensor = torch.tensor(np.array([input_data]), dtype=torch.float32).to(device)

        with torch.no_grad():
            scores, _ = model(input_tensor)
            scores = torch.nn.Sigmoid()(scores).squeeze()
            result = scores.item()

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 