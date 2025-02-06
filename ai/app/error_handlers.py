# error_handlers.py (create this in app directory)
from flask import render_template
from werkzeug.exceptions import HTTPException

def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('errors/500.html'), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        app.logger.error(f'Unhandled Exception: {e}')
        return render_template('errors/500.html'), 500