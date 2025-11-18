ReVivid — Image Restoration & Enhancement
=========================================

This repository contains the Streamlit backend app and supporting files for the ReVivid project.
It also expects a separate static frontend (HTML/JS) that communicates with the backend.

Contents
--------
- app.py            : Streamlit application (restore + enhance)
- requirements.txt  : Python packages required
- frontend_stub.html: Placeholder HTML file. Paste the full frontend HTML here (the large file from the chat) and save as reVivid_frontend.html
- README.md         : this file

Quick start (local)
-------------------
1. Create a virtualenv and install dependencies:
   python -m venv venv
   source venv/bin/activate   # mac/linux
   venv\Scripts\activate    # windows
   pip install -r requirements.txt

2. Run the Streamlit app:
   streamlit run app.py

3. (Optional) Open the frontend: save the frontend HTML (from the chat) as `reVivid_frontend.html` and host it with a static server,
   or open it in a browser. By default the frontend expects two API endpoints:
     POST /api/restore   => accepts multipart/form-data: 'image' (file), 'params' (json string)
     POST /api/enhance   => same format
   For a demo you can set `mockMode = true` inside the frontend JS to bypass the API and preview UI behavior.

Notes
-----
- The Streamlit app (app.py) in this bundle already implements restore_image(...) and enhance_image(...).
- If you want a fully integrated deployment, you can convert the frontend static page into a simple Flask/Starlette app that forwards to the Streamlit process,
  or host the frontend on GitHub Pages and run the Streamlit backend on a server that exposes the /api endpoints (Flask wrapper would be needed).
