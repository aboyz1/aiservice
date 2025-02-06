from flask import Flask, jsonify, render_template, request, send_file, session
from flask_cors import CORS
from nlp.topic_generation import generate_topics  # Updated import path
from nlp.document_generation import generate_document  # Updated import path
from nlp.chatbot import app as chatbot_app, chatbot_instance  # Updated import path
import os

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.secret_key = "supersecretkey"
CORS(app)

from error_handlers import register_error_handlers
register_error_handlers(app)


@app.route('/ai', methods=['GET'])
def ai():
    """Health check route for AI Service."""
    return jsonify({"message": "Hello from AI Service!"})

@app.route('/topic-generation', methods=['GET', 'POST'])
def topic_generation():
    """Route for topic generation."""
    topics = []
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        num_topics = int(request.form.get('num_topics', 3))
        try:
            topics = generate_topics(user_input, num_topics)
        except Exception as e:
            topics = [f"Error generating topics: {e}"]

    return render_template('topic_generation.html', topics=topics)

@app.route('/document-generator', methods=['GET', 'POST'])
def document_generator():
    """Route to generate a project document."""
    if request.method == 'POST':
        topic = request.form.get('topic')
        chapters_input = request.form.get('chapters')
        output_format = request.form.get('output_format', 'pdf')
        filename = f"generated_project.{output_format}"

        # Split chapters input into a list
        chapters = [chapter.strip() for chapter in chapters_input.split(',') if chapter.strip()]

        try:
            # Generate the document
            generate_document(topic, chapters, output_format=output_format, filename=filename)

            # Return the generated document file as a download
            return send_file(filename, as_attachment=True)
        except Exception as e:
            error_message = f"Error generating document: {e}"
            return render_template('document_generator.html', error=error_message)

    return render_template('document_generator.html')

@app.route('/chat', methods=['POST'])
def chat_with_modification():
    """Chatbot route with integration to modify topics/documents."""
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No input provided."}), 400

    try:
        # Process chatbot response
        response = chatbot_instance.get_response(user_message)
        bot_message = response["response"]

        # Check if the message contains a command to modify topics or documents
        if "modify topic" in user_message.lower():
            modified_topic = user_message.split("modify topic to", 1)[1].strip()
            session["modified_topic"] = modified_topic
            bot_message += f"\nThe topic has been modified to: {modified_topic}"

        elif "modify document" in user_message.lower():
            modified_section = user_message.split("modify document to", 1)[1].strip()
            session["modified_document"] = modified_section
            bot_message += f"\nThe document has been modified accordingly."

        return jsonify({"response": bot_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.before_request
def log_startup():
    """Log server startup status."""
    if not hasattr(app, 'started'):
        app.started = True
        print("Server connected successfully and is ready to handle requests!")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)