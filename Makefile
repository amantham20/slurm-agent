.PHONY: help build build-no-cache up start down clean logs test test-monitoring test-gpu test-ondemand status shell logs-slurmctld logs-slurmdbd update-slurm reload-slurm install-hook uninstall-hook test-doctor test-doctor-unit version set-version build-all test-all test-version rebuild jobs quick-test run-examples scale-cpu-workers scale-gpu-workers

# Default target
.DEFAULT_GOAL := help

# Supported Slurm versions
SUPPORTED_VERSIONS := 25.05.7 25.11.4
# Read default version from .env.example (source of truth)
DEFAULT_VERSION := $(shell grep '^SLURM_VERSION=' .env.example | cut -d= -f2)

# Auto-detect profiles based on .env configuration
ELASTICSEARCH_HOST := $(shell grep -E '^ELASTICSEARCH_HOST=' .env 2>/dev/null | cut -d= -f2)
GPU_ENABLE := $(shell grep -E '^GPU_ENABLE=' .env 2>/dev/null | cut -d= -f2)
OOD_ENABLE := $(shell grep -E '^OOD_ENABLE=' .env 2>/dev/null | cut -d= -f2)

# Build profile flags
PROFILES :=
ifdef ELASTICSEARCH_HOST
    PROFILES += --profile monitoring
endif
ifeq ($(GPU_ENABLE),true)
    PROFILES += --profile gpu
endif
ifeq ($(OOD_ENABLE),true)
    PROFILES += --profile ondemand
endif
PROFILE_FLAG := $(PROFILES)

# Colors for help output
CYAN := $(shell tput -Txterm setaf 6)
RESET := $(shell tput -Txterm sgr0)

help:  ## Show this help message
	@echo "Slurm Docker Cluster - Available Commands"
	@echo "=========================================="
	@echo ""
	@echo "Cluster Management:"
	@printf "  ${CYAN}%-20s${RESET} %s\n" "build" "Build Docker images"
	@printf "  ${CYAN}%-20s${RESET} %s\n" "build-no-cache" "Build Docker images without cache"
	@printf "  ${CYAN}%-20s${RESET} %s\n" "up" "Start containers"
	@printf "  ${CYAN}%-20s${RESET} %s\n" "down" "Stop containers"
	@printf "  ${CYAN}%-20s${RESET} %s\n" "clean" "Remove containers and volumes"
	@printf "  ${CYAN}%-20s${RESET} %s\n" "scale-cpu-workers" "Scale CPU workers (requires N=...)"
	@printf "  ${CYAN}%-20s${RESET} %s\n" "scale-gpu-workers" "Scale GPU workers (requires N=...)"
	@printf "  ${CYAN}%-20s${RESET} %s\n" "rebuild" "Clean, rebuild, and start"
	@echo ""
	@echo "Quick Commands:"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "jobs" "View job queue"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "status" "Show cluster status"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "logs" "Show all container logs"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "logs-slurmctld" "Show slurmctld logs"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "logs-slurmdbd" "Show slurmdbd logs"
	@echo ""
	@echo "Configuration Management:"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "update-slurm" "Update config files (requires FILES=\"...\")"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "reload-slurm" "Reload Slurm config without restart"
	@echo ""
	@echo "Development & Testing:"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "shell" "Open shell in slurmctld"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "test" "Run test suite"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "test-monitoring" "Run monitoring profile tests"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "test-gpu" "Run GPU profile tests"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "test-ondemand" "Run Open OnDemand profile tests"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "quick-test" "Submit a quick test job"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "run-examples" "Run example jobs"
	@echo ""
	@echo "Multi-Version Support:"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "version" "Show current Slurm version"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "set-version" "Set Slurm version (requires VER=...)"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "build-all" "Build all supported versions"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "test-version" "Test a specific version (requires VER=...)"
	@printf "  ${CYAN}%-15s${RESET} %s\n" "test-all" "Test all supported versions"
	@echo ""
	@echo "Examples:"
	@echo "  make update-slurm FILES=\"slurm.conf slurmdbd.conf\""
	@echo "  make set-version VER=25.05.6"
	@echo "  make scale-cpu-workers N=3"
	@echo "  make scale-gpu-workers N=2"
	@echo "  make test-version VER=25.05.6"
	@echo ""
	@echo "Monitoring:"
	@echo "  Enable:  Set ELASTICSEARCH_HOST=http://elasticsearch:9200 in .env"
	@echo "  Disable: Comment out or remove ELASTICSEARCH_HOST from .env"
	@echo ""
	@echo "GPU Support (NVIDIA):"
	@echo "  Enable:  Set GPU_ENABLE=true in .env (requires nvidia-container-toolkit on host)"
	@echo "  Disable: Set GPU_ENABLE=false or remove GPU_ENABLE from .env"
	@echo ""
	@echo "Open OnDemand:"
	@echo "  Enable:  Set OOD_ENABLE=true in .env"
	@echo "  Disable: Comment out or remove OOD_ENABLE from .env"
	@echo "  Access:  http://localhost:8080 (login: ood@localhost / password)"

build:  ## Build Docker images
	docker compose --progress plain build

build-no-cache:  ## Build Docker images without cache
	docker compose --progress plain build --no-cache

up:  ## Start containers (auto-enables monitoring if ELASTICSEARCH_HOST is set in .env)
	docker compose $(PROFILE_FLAG) up -d

down:  ## Stop containers
	docker compose $(PROFILE_FLAG) down

clean:  ## Remove containers and volumes
	docker compose $(PROFILE_FLAG) down -v

logs:  ## Show container logs
	docker compose logs -f

test:  ## Run test suite
	./test_cluster.sh

test-monitoring:  ## Run monitoring profile test suite
	./test_monitoring.sh

test-gpu:  ## Run GPU profile test suite
	./test_gpu.sh

test-ondemand:  ## Run Open OnDemand profile test suite
	./test_ondemand.sh

status:  ## Show cluster status
	@echo "=== Containers ==="
	@docker compose ps
	@echo ""
	@echo "=== Cluster ==="
	@docker exec slurmctld sinfo 2>/dev/null || echo "Not ready"

shell:  ## Open shell in slurmctld
	docker exec -it slurmctld bash

logs-slurmctld:  ## Show slurmctld logs
	docker compose logs -f slurmctld

logs-slurmdbd:  ## Show slurmdbd logs
	docker compose logs -f slurmdbd

quick-test:  ## Submit a quick test job
	docker exec slurmctld bash -c "cd /data && sbatch --wrap='hostname' && sleep 3 && squeue && cat slurm-*.out 2>/dev/null | tail -5"

run-examples:  ## Run example jobs
	./run_examples.sh

jobs:  ## View job queue
	docker exec slurmctld squeue

update-slurm:  ## Update Slurm config files (usage: make update-slurm FILES="slurm.conf slurmdbd.conf")
	@if [ -z "$(FILES)" ]; then \
		echo "Error: FILES parameter required"; \
		echo "Usage: make update-slurm FILES=\"slurm.conf slurmdbd.conf\""; \
		echo "Available: slurm.conf, slurmdbd.conf, cgroup.conf"; \
		exit 1; \
	fi
	./update_slurmfiles.sh $(FILES)

reload-slurm:  ## Reload Slurm config without restart (after live editing)
	@echo "Reloading Slurm configuration..."
	docker exec slurmctld scontrol reconfigure
	@echo "✓ Configuration reloaded"

install-hook:  ## Install the slurm-doctor jobcomp/script hook into slurmctld
	@echo "==> /opt/slurm-doctor is baked into the image; refreshing it for dev"
	@docker exec slurmctld test -d /opt/slurm-doctor || { \
		echo "ERROR: /opt/slurm-doctor not found. Run 'make build && make up' to bake it in."; exit 1; }
	-docker cp slurm-doctor/. slurmctld:/opt/slurm-doctor/ 2>/dev/null || true
	docker exec slurmctld bash -c 'mkdir -p /opt/slurm-doctor/hooks && \
		ln -sf /opt/slurm-doctor/slurm_doctor/hooks/jobcomp_hook.sh /opt/slurm-doctor/hooks/jobcomp_hook.sh && \
		ln -sf /opt/slurm-doctor/slurm_doctor/hooks/epilog.sh /opt/slurm-doctor/hooks/epilog.sh && \
		chmod -R a+rX /opt/slurm-doctor && chmod 0755 /opt/slurm-doctor/slurm_doctor/hooks/*.sh'
	@echo "==> Ensuring PyYAML is importable as the slurm user"
	docker exec slurmctld bash -c 'python3 -c "import yaml" 2>/dev/null || dnf -y -q install python3.12-pyyaml'
	@echo "==> Creating slurm-writable report + cache dirs"
	docker exec slurmctld bash -c 'mkdir -p /data/jobs/.slurm-doctor/.cache && chown -R slurm:slurm /data/jobs/.slurm-doctor'
	@echo "==> Injecting JobComp settings into /etc/slurm/slurm.conf (backed up)"
	docker exec slurmctld bash -c 'cp /etc/slurm/slurm.conf /etc/slurm/slurm.conf.sd-bak.$$(date +%s); \
		grep -vE "^(JobCompType|JobCompLoc)=" /etc/slurm/slurm.conf | grep -v "slurm-doctor:" > /etc/slurm/slurm.conf.new && \
		mv /etc/slurm/slurm.conf.new /etc/slurm/slurm.conf && \
		cat /opt/slurm-doctor/docker/slurm.conf.snippet >> /etc/slurm/slurm.conf'
	@echo "==> Reconfiguring slurmctld"
	docker exec slurmctld scontrol reconfigure
	@docker exec slurmctld bash -c 'scontrol show config | grep -iE "JobCompType|JobCompLoc"'
	@echo "✓ slurm-doctor hook installed. Failed jobs auto-report to /data/jobs/.slurm-doctor/<jobid>/"

test-doctor:  ## Run slurm-doctor end-to-end tests inside the live cluster
	@echo "==> Refreshing slurm-doctor in slurmctld"
	docker exec slurmctld rm -rf /opt/slurm-doctor
	docker cp slurm-doctor slurmctld:/opt/slurm-doctor
	@echo "==> Ensuring pytest + PyYAML are available in slurmctld"
	docker exec slurmctld bash -c 'python3 -c "import yaml" 2>/dev/null || dnf -y -q install python3.12-pyyaml'
	docker exec slurmctld bash -c 'python3 -c "import pytest" 2>/dev/null || dnf -y -q install python3.12-pytest'
	@echo "==> Running end-to-end tests against the live cluster"
	docker exec -w /opt/slurm-doctor slurmctld python3 -m pytest tests/test_end_to_end.py -v

test-doctor-unit:  ## Run slurm-doctor host unit tests (no cluster needed)
	cd slurm-doctor && python3 -m pytest tests/ --ignore=tests/test_end_to_end.py -q

uninstall-hook:  ## Remove the slurm-doctor hook and restore jobcomp/filetxt
	docker exec slurmctld bash -c 'grep -vE "^(JobCompType|JobCompLoc)=" /etc/slurm/slurm.conf | grep -v "slurm-doctor:" > /etc/slurm/slurm.conf.new && \
		mv /etc/slurm/slurm.conf.new /etc/slurm/slurm.conf && \
		printf "JobCompType=jobcomp/filetxt\nJobCompLoc=/var/log/slurm/jobcomp.log\n" >> /etc/slurm/slurm.conf'
	docker exec slurmctld scontrol reconfigure
	@echo "✓ slurm-doctor hook removed; JobComp restored to jobcomp/filetxt"

scale-cpu-workers:  ## Scale CPU workers (usage: make scale-cpu-workers N=3)
	@if [ -z "$(N)" ]; then \
		echo "Error: N parameter required. Usage: make scale-cpu-workers N=3"; \
		exit 1; \
	fi
	docker compose $(PROFILE_FLAG) up -d --scale cpu-worker=$(N) --no-recreate
	@echo "Waiting for dynamic workers to register..."; \
	sleep 10; \
	LIVE_NODES=$$(docker compose $(PROFILE_FLAG) ps cpu-worker -q 2>/dev/null \
		| while read cid; do \
			docker exec "$$cid" hostname 2>/dev/null; \
		done | sort); \
	SLURM_NODES=$$(docker exec slurmctld scontrol show nodes 2>/dev/null \
		| grep -o 'NodeName=c[0-9]*' | cut -d= -f2 | sort); \
	STALE_NODES=$$(comm -23 <(echo "$$SLURM_NODES") <(echo "$$LIVE_NODES") | paste -sd, -); \
	if [ -n "$$STALE_NODES" ]; then \
		echo "Removing stale dynamic nodes: $$STALE_NODES"; \
		docker exec slurmctld scontrol delete nodename=$$STALE_NODES; \
	fi; \
	docker exec slurmctld sinfo

scale-gpu-workers:  ## Scale GPU workers (usage: make scale-gpu-workers N=2)
	@if [ -z "$(N)" ]; then \
		echo "Error: N parameter required. Usage: make scale-gpu-workers N=2"; \
		exit 1; \
	fi
	docker compose --profile gpu $(PROFILE_FLAG) up -d --scale gpu-worker=$(N) --no-recreate
	@echo "Waiting for dynamic GPU workers to register..."; \
	sleep 10; \
	LIVE_NODES=$$(docker compose --profile gpu $(PROFILE_FLAG) ps gpu-worker -q 2>/dev/null \
		| while read cid; do \
			docker exec "$$cid" hostname 2>/dev/null; \
		done | sort); \
	SLURM_NODES=$$(docker exec slurmctld scontrol show nodes 2>/dev/null \
		| grep -o 'NodeName=g[0-9]*' | cut -d= -f2 | sort); \
	STALE_NODES=$$(comm -23 <(echo "$$SLURM_NODES") <(echo "$$LIVE_NODES") | paste -sd, -); \
	if [ -n "$$STALE_NODES" ]; then \
		echo "Removing stale GPU nodes: $$STALE_NODES"; \
		docker exec slurmctld scontrol delete nodename=$$STALE_NODES; \
	fi; \
	docker exec slurmctld sinfo

# Multi-Version Support Targets

version:  ## Show current Slurm version
	@if [ -f .env ]; then \
		grep SLURM_VERSION .env || echo "SLURM_VERSION not set (default: $(DEFAULT_VERSION))"; \
	else \
		echo "No .env file found (default: $(DEFAULT_VERSION))"; \
	fi

set-version:  ## Set Slurm version (usage: make set-version VER=25.05.6)
	@if [ -z "$(VER)" ]; then \
		echo "Error: VER parameter required. Usage: make set-version VER=25.05.6"; \
		echo "Supported versions: $(SUPPORTED_VERSIONS)"; \
		exit 1; \
	fi
	@echo "SLURM_VERSION=$(VER)" > .env
	@echo "✓ Set SLURM_VERSION=$(VER) in .env"
	@echo "Run 'make rebuild' to rebuild with this version"

build-all:  ## Build Docker images for all supported versions
	@echo "Building all supported Slurm versions..."
	@for version in $(SUPPORTED_VERSIONS); do \
		echo ""; \
		echo "========================================"; \
		echo "Building Slurm $$version"; \
		echo "========================================"; \
		echo "SLURM_VERSION=$$version" > .env; \
		docker compose build || exit 1; \
		echo "✓ Built slurm-docker-cluster:$$version"; \
	done
	@echo ""
	@echo "========================================"; \
	echo "✓ All versions built successfully"; \
	echo "========================================"; \
	docker images | grep slurm-docker-cluster

test-version:  ## Test a specific version (usage: make test-version VER=25.05.6)
	@if [ -z "$(VER)" ]; then \
		echo "Error: VER parameter required. Usage: make test-version VER=25.05.6"; \
		echo "Supported versions: $(SUPPORTED_VERSIONS)"; \
		exit 1; \
	fi
	@echo "========================================"; \
	echo "Testing Slurm $(VER)"; \
	echo "========================================"; \
	echo "SLURM_VERSION=$(VER)" > .env
	@$(MAKE) clean
	@echo "Starting cluster with Slurm $(VER)..."
	@docker compose up -d
	@echo "Waiting for services to start and auto-register..."
	@sleep 20
	@echo "Running test suite..."
	@./test_cluster.sh
	@echo ""
	@echo "✓ Slurm $(VER) tests completed"
	@$(MAKE) clean

test-all:  ## Run test suite against all supported versions
	@echo "Testing all supported Slurm versions..."
	@echo "Supported versions: $(SUPPORTED_VERSIONS)"
	@echo ""
	@for version in $(SUPPORTED_VERSIONS); do \
		echo ""; \
		echo "========================================"; \
		echo "Testing Slurm $$version"; \
		echo "========================================"; \
		$(MAKE) test-version VER=$$version || exit 1; \
	done
	@echo ""
	@echo "========================================"; \
	echo "✓ All version tests passed!"; \
	echo "========================================";

rebuild: clean build up status
