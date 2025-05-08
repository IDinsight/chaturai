# Local Setup

!!! warning "Fast-moving development ahead!"
    We strive to keep our documentation accurate and up to date. However, our development cycles move quickly, and occasionally the docs may fall slightly behind. If you run into any issues or something doesn’t work as expected, please don’t hesitate to [reach out](../contact_us.md) — we’re here to help!

This guide will you setup the project in your local environment.

## Debugging your FastAPI app on VSCode

1. In your project's `.vscode` folder, create the following `launch.json` file:

    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "FastAPI",
                "type": "debugpy",
                "request": "launch",
                "program": "${workspaceFolder}/backend/src/chaturai/entries/main.py",
                "jinja": true,
                "justMyCode": false,
                "cwd": "${workspaceFolder}/backend",
                "envFile": "${workspaceFolder}/backend/.env",
                "env": {
                    "PATHS_PROJECT_DIR": "${workspaceFolder}",
                    "PYTHONPATH": "${workspaceFolder}/backend/src"
                },
                "console": "integratedTerminal"
            },
        ]
    }
    ```

2. Add break points where you want in the ChaturAI codebase.
3. Go to the debug tab in your Visual Studio Code, and next to RUN AND DEBUG, select FastAPI and hit run.

    ![Next to RUN AND DEBUG, select FastAPI](../images/development/debug_dropdown.png)
