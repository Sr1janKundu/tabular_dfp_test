# Project Structure
```
project/
├── setup_vm.sh
├── bootstrap_project.sh
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── workspace/
```

# Make scripts executable
```bash
chmod +x setup_vm.sh
chmod +x bootstrap_project.sh
```

# Run host setup

Only required once per machine / VM.
```bash
sudo ./setup_vm.sh
```

<!-- # Restart System
```bash
sudo reboot
# for WSL, use wsl --shutdown
``` -->

# Build Development Environment
```bash
docker compose build
```

# Start Container
```bash
docker compose up -d
```

# Enter Container
```bash
docker exec -it dfp_dev bash
```

---
---

# Daily Workflow

```bash
# Start environment:
docker compose up -d
# Enter:
docker exec -it dfp_dev bash
# Stop:
docker compose down
# Rebuild after Dockerfile changes:
docker compose build --no-cache
```