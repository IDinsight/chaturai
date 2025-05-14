# Quick Setup with Docker Compose

## Setting up your instance (Ubuntu example)

### Configuring your server

1. Configure your network to accept HTTPS requests
2. Update [`chaturai/cicd/deployment/docker-compose/docker-compose.testing.yml`](https://github.com/IDinsight/chaturai/blob/main/cicd/deployment/docker-compose/docker-compose.testing.yml) with your own logging drivers and any other prod-environment configurations. Make sure your instance has the permissions to write logs.

### Installing dependencies

1. Install Docker Compose.
    1. Install the Docker Engine by following the [official installation steps](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).
    2. Install the Docker Compose plugin following the [official documentation](https://docs.docker.com/compose/install/linux/#install-using-the-repository).
    3. Add docker user
        ```
        sudo usermod -aG docker $USER
        newgrp docker
        ```
2. Install dependencies.
    ```
    sudo apt install git make direnv
    ```
3. Clone the repository and set up the environment variables following the `template.env` files.
    ```
    git clone https://github.com/IDinsight/chaturai.git
    ```
4. Use direnv to export the environment variables.
    ```
    cd chaturai
    direnv allow
    eval "$(direnv hook bash)”
    ```

## Deploy using Docker Compose

In the `chaturai` repository root, run `make prod-run`!

!!! info "Chaturai runs Playwright headlessly inside Docker."
    To run headfully for debuggin purposes, follow the local development setup instructions (currently the best source is the [our repo's readme](https://github.com/IDinsight/chaturai)).
