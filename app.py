from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, request, render_template
from flask_cors import CORS
import json

app = Flask(__name__)
# This tells Flask to completely drop its guard and allow requests from anywhere
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": ["GET", "POST", "OPTIONS"]}})

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot',methods=['GET','POST'])
def handle_prompt():
    #check which method is being used
    if request.method == 'GET':
        return "Chatbot endpoint is active!", 200
    try:
        #retrieve prompt data from prompt request
        data = request.get_data(as_text=True)
        data = json.loads(data)
        print(data)
        input_text = data['prompt']

        #join conversation history into string
        history = "\n".join(conversation_history)

        #pass user prompt and history to tokenizer for conversion
        inputs = tokenizer.encode_plus(
            history, 
            input_text,return_tensors="pt"
        )

        #pass inputs to model to be processed
        outputs = model.generate(input_ids=inputs['input_ids'], max_length=60)

        #decode output to create readable response
        response = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        #add this user prompt and assistant response to conversation history 
        #for next iteration
        conversation_history.append(input_text)
        conversation_history.append(response)

        #return response to front end to display to user
        return response
    except Exception as e:
        print(f"CRITICAL BACKEND ERROR: {str(e)}")
        return f"Backend Error: {str(e)}", 500
 

if __name__ == '__main__':
    app.run()