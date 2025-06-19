from flask import Flask, request, send_file
from style_transfer import stylize

app = Flask(__name__)

@app.route('/transfer', methods=['POST'])
def transfer():
    content = request.files['content']
    style = request.files['style']

    content_path = 'content.jpg'
    style_path = 'style.jpg'
    output_path = 'stylized.jpg'

    content.save(content_path)
    style.save(style_path)

    stylize(content_path, style_path, output_path)

    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
