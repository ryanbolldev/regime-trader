DOCKER_COMPOSE ?= docker compose
SERVICE        := regime_trader

.PHONY: build up down logs test shell dashboard restart

build:
	$(DOCKER_COMPOSE) build

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f $(SERVICE)

test:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) pytest

shell:
	$(DOCKER_COMPOSE) run --rm $(SERVICE) bash

dashboard:
	$(DOCKER_COMPOSE) up -d dashboard

restart: down up
