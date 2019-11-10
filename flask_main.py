
from flask import Flask, render_template
from flask_socketio import SocketIO
from inferent import encoder, decoder, voc, searcher, normalizeString, evaluate


def predict(input_sentence):
    try:
        input_sentence = normalizeString(input_sentence)
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        return ' '.join(output_words)

    except KeyError:
        return "Error: Encountered unknown word."

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
socketio = SocketIO(app)

@app.route('/')
def sessions():
    return render_template('chatbox.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    socketio.emit('my response', json, callback=messageReceived)
    bot_json = {'user_name': 'toandaominh1997_bot', 'message': predict(json['message'])}
    socketio.emit('my response', bot_json, callback=messageReceived)

if __name__ == '__main__':
    socketio.run(app, debug=True)