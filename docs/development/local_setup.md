# Setting up your development environment

## Step 1: Fork the repository



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
