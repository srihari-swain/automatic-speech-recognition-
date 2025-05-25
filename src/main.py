import uvicorn
from src.configs.config_loader import read_base_config


# Load config
config = read_base_config()

if __name__ == "__main__":
    uvicorn.run(
        "src.comms.server.rest_api.api:app",
        host=config["host"],
        port=config["port"],
        reload=config["reload"],
        workers=config["workers"]
    )
