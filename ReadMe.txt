AI Chatbot Project

This project is an AI-powered chatbot web application built with FastAPI and OpenAI’s API. It allows users to interact with a chatbot through a web interface, with responses generated dynamically by the OpenAI model.

Project Features
- AI chatbot integration using OpenAI API.
- FastAPI backend for routing and request handling.
- HTML templates for rendering the user interface.
- Static files support for CSS, JavaScript, and images.
- Automated testing with Pytest.
- Environment variable management for API keys and configuration.

Project Structure
- main.py: Main application file containing the FastAPI app, routes, and chatbot logic.
- test_main.py: Test file for verifying application routes and behavior.
- requirements.txt: Python dependencies.
- logging_config.py: Logging setup module, if used.
- templates/: HTML files used by the chatbot interface.
- static/: CSS, JavaScript, and image assets.
- .env: Stores secrets such as the OpenAI API key.

Technologies Used
- Python
- FastAPI
- OpenAI API
- Jinja2
- Uvicorn
- Pytest
- python-dotenv
- unittest

Installation
1. Create and activate a virtual environment.
2. Install the required packages:
   pip install -r requirements.txt
3. Add your OpenAI API key to a .env file.

Running the Application
Start the app with:

   python -m uvicorn main:app --reload

Running Tests
Run the test suite with:

   pytest

Notes
- The chatbot uses OpenAI to generate responses dynamically.
- Templates are used to display the chatbot interface in the browser.
- Static files are served for styling and frontend behavior.
- The application loads configuration from environment variables.

Summary
This project demonstrates full-stack Python development, API integration, and automated testing. It shows experience building an interactive chatbot application with modern web tools and AI services.