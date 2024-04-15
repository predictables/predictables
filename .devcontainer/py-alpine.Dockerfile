FROM alpine:3.19

# Working directory is loaded from the .env file in this directory
ARG PROJECT_NAME=alpine
WORKDIR /${PROJECT_NAME}

RUN apk update --no-check-certificate \
    && apk add --no-check-certificate --no-cache \
    zsh \
    git \
    curl \
    bash \
    musl-dev \
    libffi-dev \
    \       
    python3 \
    python3-dev \
    py3-pip \
    py3-setuptools \
    py3-wheel \
    build-base \
    \
    && python3 -m venv .venv \
    && source .venv/bin/activate \
    && pip install --upgrade pip \
    && pip install --no-cache-dir \
    ruff \
    ruff-lsp \
    mypy \
    uv \
    \
    && rm -rf /var/cache/apk/* \
    && rm -rf /root/.cache \
    \
    && git clone -c http.sslVerify=false http://github.com/aaweaver-actuary/dotfiles \
    && mkdir -p ~/bin \
    && cp ./dotfiles/install_dotfiles ~/bin/install_dotfiles \
    && chmod +x ~/dotfiles/bin/* \
    && rm -rf ./dotfiles \
    \
    && install_dotfiles /${PROJECT_NAME} install_oh_my_zsh \
    && chmod +x /${PROJECT_NAME}/install_oh_my_zsh \
    && . /${PROJECT_NAME}/install_oh_my_zsh \
    && rm -rf /${PROJECT_NAME}/install_oh_my_zsh \
    && install_dotfiles ~ .zshrc .zsh_aliases .vimrc \
    \
    && apk del --no-cache \
    python3-dev \
    py3-pip \
    py3-setuptools \
    py3-wheel \
    build-base \
    && apk cache clean \
    && apk cache purge \
    && rm -rf /var/cache/apk/* \
    && rm -rf /root/.cache

SHELL ["/bin/zsh", "-c"]

CMD ["zsh"]